"""Main VMC loop."""
import collections
import logging
import os
from typing import Callable, Dict, Tuple, TypeVar
from vmcnet.utils.checkpoint import RunningEnergyVariance, RunningMetric

import jax
import jax.numpy as jnp
from kfac_ferminet_alpha import utils as kfac_utils

import vmcnet.utils as utils

# represents a pytree or pytree-like object containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
P = TypeVar("P")  # represents a pytree or pytree-like object containing model params
O = TypeVar("O")  # represents optimizer state


def make_metropolis_step(
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
) -> Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]]:
    """Factory to create a function which takes a single metropolis step.

    Following Metropolis-Hastings Markov Chain Monte Carlo, a transition from one data
    state to another is split into proposal and acceptance. When used in a Metropolis
    routine to approximate a stationary distribution P, the proposal and acceptance
    functions should satisfy detailed balance, i.e.,

        proposal_prob_ij * acceptance_ij * P_i = proposal_prob_ji * acceptance_ji * P_j,

    where proposal_prob_ij is the likelihood of proposing the transition from state i to
    state j, acceptance_ij is the likelihood of accepting a transition from state i
    to state j, and P_i is the probability of being in state i.

    Args:
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> proposed_data, key
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> jnp.ndarray: acceptance probabilities
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data

    Returns:
        Callable: function which takes in (data, params, key) and outputs
        (mean acceptance probability, new data, new jax PRNG key split from previous
        one)
    """

    def metrop_step_fn(data, params, key):
        """Take a single metropolis step."""
        key, subkey = jax.random.split(key)
        proposed_data, key = proposal_fn(params, data, key)
        accept_prob = acceptance_fn(params, data, proposed_data)
        move_mask = jax.random.uniform(subkey, shape=accept_prob.shape) < accept_prob
        new_data = update_data_fn(data, proposed_data, move_mask)

        return jnp.mean(accept_prob), new_data, key

    return metrop_step_fn


def walk_data(
    nsteps: int,
    data: D,
    params: P,
    key: jnp.ndarray,
    metrop_step_fn: Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]],
) -> Tuple[jnp.float32, D, jnp.ndarray]:
    """Take multiple Metropolis-Hastings steps.

    This function is roughly equivalent to:
    ```
    accept_sum = 0.0
    for _ in range(nsteps):
        accept_prob, data, key = metropolis_step_fn(data, params, key)
        accept_sum += accept_prob
    return accept_sum / nsteps, data, key
    ```

    but has better tracing/pmap behavior due to the use of jax.lax.scan instead of a
    python for loop. See :func:`~vmcnet.train.vmc.take_metropolis_step`.

    Args:
        nsteps (int): number of steps to take
        data (pytree-like): data to walk (update) with each step
        params (pytree-like): parameters passed to proposal_fn and acceptance_fn, e.g.
            model params
        key (jnp.ndarray): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)

    Returns:
        (jnp.float32, pytree-like, jnp.ndarray): acceptance probability, new data,
            new jax PRNG key split (possibly multiple times) from previous one
    """

    def step_fn(carry, x):
        del x
        accept_prob, data, key = metrop_step_fn(carry[1], params, carry[2])
        return (carry[0] + accept_prob, data, key), None

    out = jax.lax.scan(step_fn, (0.0, data, key), xs=None, length=nsteps)
    accept_sum, data, key = out[0]
    return accept_sum / nsteps, data, key


def make_burning_step(
    metrop_step_fn: Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]],
    pmapped: bool = True,
) -> Callable[[D, P, jnp.ndarray], Tuple[D, jnp.ndarray]]:
    """Factory to create a burning step, which is an optionally pmapped Metropolis step.

    This provides the functionality to optionally apply jax.pmap to a single Metropolis
    step. Only one step is traced so that the first burning step is traced but
    subsequent steps are properly jit-compiled. The acceptance probabilities (which
    typically don't mean much during burning) are thrown away.

    For more about the Metropolis step itself, see
    :func:`~vmcnet.train.vmc.make_metropolis_step`
    and to see it in use, see
    :func:`~vmcnet.train.vmc.vmc_loop`.

    Args:
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)
        pmapped (bool, optional): whether to apply jax.pmap to the burning step.
            Defaults to True.

    Returns:
        Callable: function with signature
            (data, params, key) -> (data, key),
        with jax.pmap optionally applied if pmapped is True. Because it is totally pure,
        the original (data, params, key) buffers are deleted in the pmapped version via
        the `donate_argumns` argument so that XLA is potentially more memory-efficient
        on the GPU. See :func:`jax.pmap`.
    """

    def burning_step(data, params, key):
        _, data, key = metrop_step_fn(data, params, key)
        return data, key

    if not pmapped:
        return burning_step

    return utils.distribute.pmap(burning_step, donate_argnums=(0, 1, 2))


def make_training_step(
    nsteps_per_param_update: int,
    metrop_step_fn: Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]],
    update_param_fn: Callable[[D, P, O], Tuple[P, O, Dict]],
    pmapped: bool = True,
) -> Callable[[D, P, O, jnp.ndarray], Tuple[jnp.float32, D, P, O, Dict, jnp.ndarray]]:
    """Factory to create a training step.

    This provides the functionality to optionally apply jax.pmap to a single training
    step. Only one step is traced so that the first training step is traced but
    subsequent steps are properly jit-compiled.

    The training step consists of two parts:
        1) the walker, which updates the data `nsteps_per_param_update` times
        2) the parameter updates, which occurs once per training step, and is the only
        time `energy_fn` is evaluated during the training step.

    See :func:`~vmcnet.train.vmc.vmc_loop`.

    Args:
        nsteps_per_param_update (int): number of steps to walk data before applying a
            parameter update
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)
        update_param_fn (Callable): function which updates the parameters. Has signature
            (data, params, optimizer_state)
                -> (new_params, optimizer_state, dict: metrics).
            If metrics is not None, it is required to have the entries "energy" and
            "variance" at a minimum.
        pmapped (bool, optional): whether to apply jax.pmap to the burning step.
            Defaults to True.

    Returns:
        Callable: function with signature
            (data, params, optimizer_state, key)
                -> (accept_ratio, data, params, optmizer_state, metrics, key),
        with jax.pmap optionally applied if pmapped is True. Because it is totally pure,
        the original (data, params, optimizer_state, key) buffers are deleted in the
        pmapped version using the `donate_argnums` flag so that XLA is potentially more
        memory-efficient on the GPU.
        See :func:`jax.pmap`.
    """
    nsteps_per_param_update = max(nsteps_per_param_update, 1)

    def training_step(data, params, optimizer_state, key):
        accept_ratio, data, key = walk_data(
            nsteps_per_param_update, data, params, key, metrop_step_fn
        )
        params, optimizer_state, metrics = update_param_fn(
            data, params, optimizer_state
        )
        accept_ratio = utils.distribute.pmean_if_pmap(accept_ratio)
        return accept_ratio, data, params, optimizer_state, metrics, key

    if not pmapped:
        return training_step

    return utils.distribute.pmap(training_step, donate_argnums=(0, 1, 2, 3))


def vmc_loop(
    params: P,
    optimizer_state: O,
    initial_data: D,
    nchains: int,
    nburn: int,
    nepochs: int,
    nsteps_per_param_update: int,
    metrop_step_fn: Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]],
    update_param_fn: Callable[[D, P, O], Tuple[P, O, Dict]],
    key: jnp.ndarray,
    logdir: str = None,
    checkpoint_every: int = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_variance_scale: float = 10.0,
    nhistory_max: int = 200,
    pmapped: bool = True,
) -> Tuple[P, O, D]:
    """Main Variational Monte Carlo loop routine.

    Variational Monte Carlo (VMC) can be generically viewed as minimizing a
    parameterized variational loss stochastically by sampling over a data distribution
    via Monte Carlo sampling. This function implements this idea at a high level, using
    proposal and acceptance functions to sample the data distribution, and passing the
    optimization step to a generic function `update_param_fn`.

    As is custom in VMC, some number of burn-in steps are run before any training
    occurs. Some data update steps are also taken in between each parameter update
    during training. Theoretically, whether these are necessary or helpful is not
    completely clear.

    Practically, the burn-in steps usually lead to higher quality gradient steps once
    the training starts, and seems to be a good idea whenever burning is much cheaper
    than gradient/parameter update calculations. Likewise, inserting a number of
    intermediate data-only update steps between parameter updates during training,
    controlled by `nsteps_per_param_update`, seems to be a good idea in the same regime.

    Args:
        params (pytree-like): model parameters which are trained. If pmapped is True,
            this should already be replicated over all devices.
        optimizer_state (pytree-like): initial state of the optimizer. If pmapped is
            True, this should contain state which has been sharded over all devices.
        initial_data (pytree-like): initial data. If pmapped is True, this should
            contain data which as been sharded over all devices.
        nchains (int): number of parallel MCMC chains being run. This can be difficult
            to infer from data, depending on the structure of data, whether data has
            been pmapped, etc.
        nburn (int): number of data updates to do before training starts. All data
            except for the final data after burning is thrown away.
        nepochs (int): number of parameter updates to do
        nsteps_per_param_update (int): number of data updates to do between each
            parameter update. All data except for the final data after
            nsteps_per_param_update iterations is thrown away.
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)
        update_param_fn (Callable): function which updates the parameters. Has signature
            (data, params, optimizer_state)
                -> (new_params, optimizer_state, dict: metrics).
            If metrics is not None, it is required to have the entries "energy" and
            "variance" at a minimum. If metrics is None, no checkpointing is done.
        key (jnp.ndarray): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn
        logdir (str, optional): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, checkpoints are only saved when the error-adjusted running avg of the
            energy improves. Defaults to None.
        checkpoint_dir (str, optional): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_variance_scale (float, optional): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`. Defaults to 10.0.
        nhistory_max (int, optional): How much history to keep in the running histories
            of the energy and variance. Defaults to 200.
        pmapped (bool, optional): whether to apply jax.pmap to the burning and training
            steps. Defaults to True.

    Returns:
        A tuple of (trained parameters, final optimizer state, final data). These are
        the same structure as (params, optimizer_state, initial_data).
    """
    (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
    ) = utils.checkpoint.initialize_checkpointing_metrics(
        checkpoint_dir, nhistory_max, logdir, checkpoint_every
    )
    data = initial_data

    burning_step = make_burning_step(metrop_step_fn, pmapped=pmapped)
    training_step = make_training_step(
        nsteps_per_param_update, metrop_step_fn, update_param_fn, pmapped=pmapped
    )

    logging.info("Burning for " + str(nburn) + " steps")
    for _ in range(nburn):
        data, key = burning_step(data, params, key)

    for epoch in range(nepochs):
        # for checkpointing; want to save the state that resulted in the best metrics,
        # not the state one step later
        old_params = params
        old_optimizer_state = optimizer_state
        accept_ratio, data, params, optimizer_state, metrics, key = training_step(
            data, params, optimizer_state, key
        )
        if metrics is None:  # don't checkpoint if no metrics to checkpoint
            continue

        metrics["accept_ratio"] = accept_ratio
        if pmapped:
            # Assume all metrics have been collectively reduced to be the same on
            # all devices
            metrics = kfac_utils.get_first(metrics)

        (
            checkpoint_metric,
            checkpoint_str,
        ) = utils.checkpoint.save_metrics_and_handle_checkpoints(
            epoch,
            old_params,
            old_optimizer_state,
            data,
            metrics,
            nchains,
            running_energy_and_variance,
            checkpoint_metric,
            logdir=logdir,
            variance_scale=checkpoint_variance_scale,
            checkpoint_every=None,
            checkpoint_dir=checkpoint_dir,
        )
        utils.checkpoint.log_vmc_loop_state(epoch, metrics, checkpoint_str)

    return params, optimizer_state, data
