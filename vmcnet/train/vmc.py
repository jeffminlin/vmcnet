"""Main VMC loop."""
import collections
import logging
import os
from typing import Callable, Tuple

import jax.numpy as jnp
import jax
import numpy as np

import vmcnet.utils as utils


def take_metropolis_step(data, params, proposal_fn, acceptance_fn, update_data_fn, key):
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
        (jnp.float32, pytree-like, jnp.ndarray): acceptance probability, new data,
            new jax PRNG key split from previous one
    """
    key, subkey = jax.random.split(key)
    proposed_data, key = proposal_fn(params, data, key)
    accept_prob = acceptance_fn(params, data, proposed_data)
    move_mask = jax.random.uniform(subkey, shape=accept_prob.shape) < accept_prob
    new_data = update_data_fn(data, proposed_data, move_mask)

    return jnp.mean(accept_prob), new_data, key


def walk_data(nsteps, data, params, proposal_fn, acceptance_fn, update_data_fn, key):
    """Take multiple Metropolis-Hastings steps, via take_metropolis_step.

    This function is roughly equivalent to
    
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


def make_burning_step(proposal_fn, acceptance_fn, update_data_fn, pmapped=True):
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

    return jax.pmap(burning_step, donate_argnums=(0, 1, 2))


def make_training_step(
    nskip, proposal_fn, acceptance_fn, update_data_fn, update_param_fn, pmapped=True
):
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
        pmapped version so that XLA is potentially more memory-efficient on the GPU.
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
        return accept_ratio, data, params, optimizer_state, metrics, key

    if not pmapped:
        return training_step

    return jax.pmap(training_step, donate_argnums=(0, 1, 2, 3))


def update_position_and_amplitude(data, proposed_data, move_mask):
    pos_mask = jnp.reshape(move_mask, (-1,) + (len(data["position"].shape) - 1) * (1,))
    new_position = jnp.where(pos_mask, proposed_data["position"], data["position"])
    new_amplitude = jnp.where(move_mask, proposed_data["amplitude"], data["amplitude"])
    return {"position": new_position, "amplitude": new_amplitude}


def move_history_window(energy_history, variance_history, obs, epoch, nhistory):
    energy_history.append(obs["energy"])
    variance_history.append(obs["variance"])
    if epoch >= nhistory:
        energy_history.popleft()
        variance_history.popleft()


def get_checkpoint_metric(energy_history, variance_history, nchains, variance_scale):

    energy_running_avg = jnp.average(energy_history)
    variance_running_avg = jnp.average(variance_history)
    error_adjusted_running_avg = energy_running_avg + variance_scale * jnp.sqrt(
        variance_running_avg / (len(energy_history) * nchains)
    )

    return error_adjusted_running_avg


def checkpoint(
    epoch,
    params,
    optimizer_state,
    data,
    metrics,
    energy_history,
    variance_history,
    checkpoint_metric,
    logdir=None,
    variance_scale=10.0,
    checkpoint_every=None,
    checkpoint_dir="checkpoints",
    nhistory=50,
):
    if logdir is None or metrics is None:
        return checkpoint_metric, ""

    for metric, metric_val in metrics.items():
        utils.io.write_metric_to_file(metric_val, logdir, metric)

    if checkpoint_every is not None:
        if (epoch + 1) % checkpoint_every == 0:
            utils.io.save_params(
                os.path.join(logdir, checkpoint_dir),
                str(epoch + 1) + ".npz",
                data,
                params,
                optimizer_state,
            )

    move_history_window(energy_history, variance_history, metrics, epoch, nhistory)
    checkpoint_str = ""
    error_adjusted_running_avg = get_checkpoint_metric(
        energy_history, variance_history, data.shape[0], variance_scale
    )
    if error_adjusted_running_avg < checkpoint_metric:
        utils.io.save_params(
            logdir, "checkpoint.npz", data, params, optimizer_state,
        )
        checkpoint_str = ", new weights saved"

    return jnp.minimum(error_adjusted_running_avg, checkpoint_metric), checkpoint_str


def log_vmc_loop_state(epoch, metrics, checkpoint_str):
    epoch_str = "Epoch {:5d}".format(epoch + 1)
    energy_str = "Energy: {:.5e}".format(float(metrics["energy"]))
    variance_str = "Variance: {:.5e}".format(float(metrics["variance"]))
    accept_ratio_str = "Accept ratio: {:.5f}".format(float(metrics["accept_ratio"]))

    if "energy_noclip" in metrics:
        energy_str = energy_str + "({:.5e})".format(float(metrics["energy_noclip"]))

    if "variance_noclip" in metrics:
        variance_str = variance_str + " ({:.5e})".format(
            float(metrics["variance_noclip"])
        )

    info_out = ", ".join([epoch_str, energy_str, variance_str, accept_ratio_str])
    info_out = info_out + checkpoint_str
    logging.info(info_out)


def vmc_loop(
    params,
    optimizer_state,
    initial_data,
    nburn,
    nepochs,
    nskip,
    proposal_fn,
    acceptance_fn,
    update_data_fn,
    update_param_fn,
    key,
    logdir=None,
    checkpoint_every=None,
    checkpoint_dir="checkpoints",
    checkpoint_variance_scale=10.0,
    nhistory=50,
    pmapped=True,
):
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

    checkpoint_metric = np.inf
    energy_history = collections.deque()
    variance_history = collections.deque()
    for epoch in range(nepochs):
        # for checkpointing; want to save the state that resulted in the best metrics,
        # not the state one step later
        old_params = params
        old_optimizer_state = optimizer_state
        accept_ratio, data, params, optimizer_state, metrics, key = training_step(
            data, params, optimizer_state, key
        )
        checkpoint_metric, checkpoint_str = checkpoint(
            epoch,
            old_params,
            old_optimizer_state,
            data,
            metrics,
            energy_history,
            variance_history,
            checkpoint_metric,
            logdir=logdir,
            variance_scale=checkpoint_variance_scale,
            checkpoint_every=None,
            checkpoint_dir=checkpoint_dir,
            nhistory=nhistory,
        )
        if metrics is not None:
            metrics["accept_ratio"] = accept_ratio
            log_vmc_loop_state(epoch, metrics, checkpoint_str)

    return params, optimizer_state, data
