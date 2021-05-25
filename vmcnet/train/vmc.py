"""Main VMC loop."""
import collections
import logging
import os
from typing import Callable, Dict, Tuple, TypeVar

import jax
import jax.numpy as jnp
from kfac_ferminet_alpha import utils as kfac_utils

import vmcnet.utils as utils

# to represent a pytree or pytree-like object containing MCMC data, e.g. walker
# positions and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
P = TypeVar("P")  # to represent a pytree or pytree-like object containing model params
O = TypeVar("O")  # to represent optimizer state


def take_metropolis_step(
    data: D,
    params: P,
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
    key: jnp.ndarray,
) -> Tuple[jnp.float32, D, jnp.ndarray]:
    """Split a single step of updating data into proposal and acceptance.

    Following Metropolis-Hastings Markov Chain Monte Carlo, a transition from one data
    state to another is split into proposal and acceptance. When used in a Metropolis
    routine to approximate a stationary distribution P, the proposal and acceptance
    functions should satisfy detailed balance, i.e.,

        proposal_prob_ij * acceptance_ij * P_i = proposal_prob_ji * acceptance_ji * P_j,

    where proposal_prob_ij is the likelihood of proposing the transition from state i to
    state j, acceptance_ij is the likelihood of accepting a transition from state i
    to state j, and P_i is the probability of being in state i.

    Args:
        data (pytree-like): data to update
        params (pytree-like): parameters passed to proposal_fn and acceptance_fn, e.g.
            model params
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> proposed_data, key
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> jnp.ndarray: acceptance probabilities
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data
        key (jnp.ndarray): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn

    Returns:
        (jnp.float32, pytree-like, jnp.ndarray): mean acceptance probability, new data,
            new jax PRNG key split from previous one
    """
    key, subkey = jax.random.split(key)
    proposed_data, key = proposal_fn(params, data, key)
    accept_prob = acceptance_fn(params, data, proposed_data)
    move_mask = jax.random.uniform(subkey, shape=accept_prob.shape) < accept_prob
    new_data = update_data_fn(data, proposed_data, move_mask)

    return jnp.mean(accept_prob), new_data, key


def walk_data(
    nsteps: int,
    data: D,
    params: P,
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
    key: jnp.ndarray,
) -> Tuple[jnp.float32, D, jnp.ndarray]:
    """Take multiple Metropolis-Hastings steps, via take_metropolis_step.

    This function is roughly equivalent to:
    ```
    accept_sum = 0.0
    for _ in range(nsteps):
        accept_prob, data, key = take_metropolis_step(
            data, params, proposal_fn, acceptance_fn, update_data_fn, key
        )
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
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> proposed_data, key
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> jnp.ndarray: acceptance probabilities
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data
        key (jnp.ndarray): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn

    Returns:
        (jnp.float32, pytree-like, jnp.ndarray): acceptance probability, new data,
            new jax PRNG key split (possibly multiple times) from previous one
    """

    def step_fn(carry, x):
        del x
        accept_prob, data, key = take_metropolis_step(
            carry[1], params, proposal_fn, acceptance_fn, update_data_fn, carry[2]
        )
        return (carry[0] + accept_prob, data, key), None

    out = jax.lax.scan(step_fn, (0.0, data, key), xs=None, length=nsteps)
    accept_sum, data, key = out[0]
    return accept_sum / nsteps, data, key


def make_burning_step(
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
    pmapped: bool = True,
) -> Callable[[D, P, jnp.ndarray], Tuple[D, jnp.ndarray]]:
    """Factory to create a burning step.

    This provides the functionality to optionally apply jax.pmap to a single walker
    step. Only one step is traced so that the first burning step is traced but
    subsequent steps are properly jit-compiled.

    For more about the Metropolis step itself, see
    :func:`~vmcnet.train.vmc.take_metropolis_step`
    and to see it in use, see
    :func:`~vmcnet.train.vmc.vmc_loop`.

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
        pmapped (bool, optional): whether to apply jax.pmap to the burning step.
            Defaults to True.

    Returns:
        Callable: function with signature
            (data, params, key) -> (data, key),
        with jax.pmap optionally applied if pmapped is True. Because it is totally pure,
        the original (data, params, key) buffers are deleted in the pmapped version so
        that XLA is potentially more memory-efficient on the GPU. See :func:`jax.pmap`.
    """

    def burning_step(data, params, key):
        _, data, key = take_metropolis_step(
            data, params, proposal_fn, acceptance_fn, update_data_fn, key
        )
        return data, key

    if not pmapped:
        return burning_step

    return utils.distribute.pmap(burning_step, donate_argnums=(0, 1, 2))


def make_training_step(
    nskip: int,
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
    update_param_fn: Callable[[D, P, O], Tuple[P, O, Dict]],
    pmapped: bool = True,
) -> Callable[[D, P, O, jnp.ndarray], Tuple[jnp.float32, D, P, O, Dict, jnp.ndarray]]:
    """Factory to create a training step.

    This provides the functionality to optionally apply jax.pmap to a single training
    step. Only one step is traced so that the first training step is traced but
    subsequent steps are properly jit-compiled.

    The training step consists of two parts:
        1) the walker, which updates the data nskip times (so-called nskip because we
        'skip' these parameter updates). 
        2) the parameter updates, which occurs once per training step, and is the only
        time `energy_fn` is evaluated during the training step.
    
    See :func:`~vmcnet.train.vmc.vmc_loop`.

    Args:
        nskip (int): number of steps to walk data before applying a parameter update
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> (proposed_data, key).
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> jnp.ndarray: acceptance_prob
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data
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
        pmapped version using the donate_argnums flag so that XLA is potentially more
        memory-efficient on the GPU.
        See :func:`jax.pmap`.
    """
    nskip = max(nskip, 1)

    def training_step(data, params, optimizer_state, key):
        accept_ratio, data, key = walk_data(
            nskip, data, params, proposal_fn, acceptance_fn, update_data_fn, key,
        )
        params, optimizer_state, metrics = update_param_fn(
            data, params, optimizer_state
        )
        accept_ratio = utils.distribute.pmean_if_pmap(accept_ratio)
        return accept_ratio, data, params, optimizer_state, metrics, key

    if not pmapped:
        return training_step

    return utils.distribute.pmap(training_step, donate_argnums=(0, 1, 2, 3))


def move_history_window(
    running_energy_avg: jnp.float32,
    running_variance_avg: jnp.float32,
    energy_history: collections.deque,
    variance_history: collections.deque,
    metrics: Dict,
    epoch: int,
    nhistory: int,
) -> Tuple[jnp.float32, jnp.float32]:
    """Append energy/variance to running history, remove oldest if length > nhistory."""
    if nhistory <= 0:
        return running_energy_avg, running_variance_avg

    energy, variance = metrics["energy"], metrics["variance"]
    running_energy_sum = len(energy_history) * running_energy_avg
    running_variance_sum = len(variance_history) * running_variance_avg

    energy_history.append(energy)
    running_energy_sum += energy
    variance_history.append(variance)
    running_variance_sum += variance

    if epoch >= nhistory:
        oldest_energy = energy_history.popleft()
        running_energy_sum -= oldest_energy
        oldest_variance = variance_history.popleft()
        running_variance_sum -= oldest_variance

    return (
        running_energy_sum / len(energy_history),
        running_variance_sum / len(variance_history),
    )


def get_checkpoint_metric(
    energy_running_avg: jnp.float32,
    variance_running_avg: jnp.float32,
    nsamples: int,
    variance_scale: float,
) -> jnp.float32:
    """Get an error-adjusted running average of the energy for checkpointing.
    
    The parameter variance_scale can be tuned and probably should scale linearly with
    some estimate of the integrated autocorrelation. Higher means more allergic to high
    variance, lower means more allergic to high energies.

    Args:
        energy_running_avg (jnp.float32): running average of the energy
        variance_running_avg (jnp.float32): running average of the variance
        nsamples (int): total number of samples reflected in the running averages, equal
            to the number of parallel chains times the length of the history
        variance_scale (float): weight of the variance part of the checkpointing metric.
            The final effect on the variance part is to scale it by
            jnp.sqrt(variance_scale), i.e. to treat it like the integrated
            autocorrelation.

    Returns:
        jnp.float32: error adjusted running average of the energy
    """
    # TODO(Jeffmin): eventually maybe put in some cheap best guess at the IAC?
    if variance_scale <= 0.0 or nsamples <= 0:
        return energy_running_avg

    effective_nsamples = nsamples / variance_scale
    return energy_running_avg + jnp.sqrt(variance_running_avg / effective_nsamples)


def checkpoint(
    epoch: int,
    params: P,
    optimizer_state: O,
    data: D,
    metrics: Dict,
    nchains: int,
    running_energy_avg: jnp.float32,
    running_variance_avg: jnp.float32,
    energy_history: collections.deque[jnp.float32],
    variance_history: collections.deque[jnp.float32],
    checkpoint_metric: jnp.float32,
    logdir: str = None,
    variance_scale: float = 10.0,
    checkpoint_every: int = None,
    checkpoint_dir: str = "checkpoints",
    nhistory: int = 50,
) -> Tuple[jnp.float32, jnp.float32, jnp.float32, str]:
    """Checkpoint the current state of the VMC loop.

    There are two situations to checkpoint:
        1) Regularly, every x epochs, to handle job preemption and track
        parameters/metrics/state over time, and
        2) Whenever a checkpoint metric improves, i.e. the error adjusted running
        average of the energy.

    This function handles both of these cases, as well as writing all current metrics to
    their own files. This currently touches the disk repeatedly, once for each metric,
    which is probably fairly inefficient, especially if called every epoch (as it
    currently is in :func:`~vmcnet.train.vmc.vmc_loop`).

    This is not a *pure* function, as it modifies energy_history and variance_history.

    Args:
        epoch (int): current epoch number
        params (pytree-like): model parameters. Needs to be serializable via `np.savez`
        optimizer_state (pytree-like): running state of the optimizer other than the
            trainable parameters. Needs to be serialiable via `np.savez`
        data (pytree-like): current mcmc data (e.g. position and amplitude data). Needs
            to be serializable via `np.savez`
        metrics (dict): dictionary of metrics. If this is not None, then it must include
            "energy" and "variance". Metrics are currently flattened and written to a
            row of a text file. See :func:`utils.io.write_metric_to_file`.
        nchains (int): number of parallel MCMC chains being run. This can be difficult
            to infer from data, depending on the structure of data, whether data has
            been pmapped, etc.
        running_energy_avg (jnp.float32): running average of energies
        running_variance_avg (jnp.float32): running average of variances
        energy_history (collections.deque[jnp.float32]): running history of energies
        variance_history (collections.deque[jnp.float32]): running history of variances
        checkpoint_metric (jnp.float32): current best error adjusted running average of
            the energy history
        logdir (str, optional): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        variance_scale (float, optional): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`. Defaults to 10.0.
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, checkpoints are only saved when the error-adjusted running avg of the
            energy improves. Defaults to None.
        checkpoint_dir (str, optional): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        nhistory (int, optional): How much history to keep in the running histories of
            the energy and variance. Defaults to 50.

    Returns:
        (jnp.float32, jnp.float32, jnp.float32, str):
        (
            updated running average of the energy,
            updated running average of the variance,
            updated checkpoint metric, and
            string indicating if checkpointing has been done
        )
    """
    checkpoint_str = ""
    if logdir is None or metrics is None:
        # do nothing
        return (
            running_energy_avg,
            running_variance_avg,
            checkpoint_metric,
            checkpoint_str,
        )

    # TODO(Jeffmin): do something more efficient than writing separately to disk for
    # every metric, maybe something like pandas? Also maybe shouldn't be every epoch.
    # Might be better to switch to something like TensorBoard -- need to do research
    for metric, metric_val in metrics.items():
        utils.io.append_metric_to_file(metric_val, logdir, metric)

    if checkpoint_every is not None:
        if (epoch + 1) % checkpoint_every == 0:
            utils.io.save_params(
                os.path.join(logdir, checkpoint_dir),
                str(epoch + 1) + ".npz",
                epoch,
                data,
                params,
                optimizer_state,
            )
        checkpoint_str = checkpoint_str + ", regular ckpt saved"

    running_energy_avg, running_variance_avg = move_history_window(
        running_energy_avg,
        running_variance_avg,
        energy_history,
        variance_history,
        metrics,
        epoch,
        nhistory,
    )
    error_adjusted_running_avg = get_checkpoint_metric(
        running_energy_avg,
        running_variance_avg,
        nchains * len(energy_history),
        variance_scale,
    )
    if error_adjusted_running_avg < checkpoint_metric:
        utils.io.save_params(
            logdir, "checkpoint.npz", epoch, data, params, optimizer_state,
        )
        checkpoint_str = checkpoint_str + ", best weights saved"

    return (
        running_energy_avg,
        running_variance_avg,
        jnp.minimum(error_adjusted_running_avg, checkpoint_metric),
        checkpoint_str,
    )


def log_vmc_loop_state(epoch: int, metrics: Dict, checkpoint_str: str):
    """Log current energy, variance, and accept ratio, w/ optional unclipped values."""
    epoch_str = "Epoch {:5d}".format(epoch + 1)
    energy_str = "Energy: {:.5e}".format(float(metrics["energy"]))
    variance_str = "Variance: {:.5e}".format(float(metrics["variance"]))
    accept_ratio_str = "Accept ratio: {:.5f}".format(float(metrics["accept_ratio"]))

    if "energy_noclip" in metrics:
        energy_str = energy_str + " ({:.5e})".format(float(metrics["energy_noclip"]))

    if "variance_noclip" in metrics:
        variance_str = variance_str + " ({:.5e})".format(
            float(metrics["variance_noclip"])
        )

    info_out = ", ".join([epoch_str, energy_str, variance_str, accept_ratio_str])
    info_out = info_out + checkpoint_str
    logging.info(info_out)


def vmc_loop(
    params: P,
    optimizer_state: O,
    initial_data: D,
    nchains: int,
    nburn: int,
    nepochs: int,
    nskip: int,
    proposal_fn: Callable[[P, D, jnp.ndarray], Tuple[D, jnp.ndarray]],
    acceptance_fn: Callable[[P, D, D], jnp.ndarray],
    update_data_fn: Callable[[D, D, jnp.ndarray], D],
    update_param_fn: Callable[[D, P, O], Tuple[P, O, Dict]],
    key: jnp.ndarray,
    logdir: str = None,
    checkpoint_every: int = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_variance_scale: float = 10.0,
    nhistory: int = 50,
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
    controlled by `nskip`, seems to be a good idea in the same regime.

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
        nskip (int): number of data updates to do between each parameter update. All
            data except for the final data after nskip iterations is thrown away.
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> proposed_data, key
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> jnp.ndarray: acceptance probabilities
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data
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
        nhistory (int, optional): How much history to keep in the running histories of
            the energy and variance. Defaults to 50.
        pmapped (bool, optional): whether to apply jax.pmap to the burning and training
            steps. Defaults to True.

    Returns:
        A tuple of (trained parameters, final optimizer state, final data). These are
        the same structure as (params, optimizer_state, initial_data).
    """
    checkpointdir = None
    if logdir is not None:
        logging.info("Saving to " + logdir)
        os.makedirs(logdir, exist_ok=True)
        if checkpoint_every is not None:
            checkpointdir = utils.io.add_suffix_for_uniqueness(checkpoint_dir, logdir)
            os.makedirs(os.path.join(logdir, checkpointdir), exist_ok=False)

    data = initial_data

    burning_step = make_burning_step(
        proposal_fn, acceptance_fn, update_data_fn, pmapped=pmapped
    )
    training_step = make_training_step(
        nskip,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        update_param_fn,
        pmapped=pmapped,
    )

    logging.info("Burning for " + str(nburn) + " steps")
    for _ in range(nburn):
        data, key = burning_step(data, params, key)

    checkpoint_metric = jnp.inf
    running_energy_avg = 0.0
    running_variance_avg = 0.0
    energy_history: collections.deque[jnp.float32] = collections.deque()
    variance_history: collections.deque[jnp.float32] = collections.deque()
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
            running_energy_avg,
            running_variance_avg,
            checkpoint_metric,
            checkpoint_str,
        ) = checkpoint(
            epoch,
            old_params,
            old_optimizer_state,
            data,
            metrics,
            nchains,
            running_energy_avg,
            running_variance_avg,
            energy_history,
            variance_history,
            checkpoint_metric,
            logdir=logdir,
            variance_scale=checkpoint_variance_scale,
            checkpoint_every=None,
            checkpoint_dir=checkpoint_dir,
            nhistory=nhistory,
        )
        log_vmc_loop_state(epoch, metrics, checkpoint_str)

    return params, optimizer_state, data
