"""Main VMC loop."""
from typing import Callable, Dict, Tuple, TypeVar

import jax.numpy as jnp

import vmcnet.utils as utils

# represents a pytree or pytree-like object containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
P = TypeVar("P")  # represents a pytree or pytree-like object containing model params
S = TypeVar("S")  # represents optimizer state


def vmc_loop(
    params: P,
    optimizer_state: S,
    data: D,
    nchains: int,
    nepochs: int,
    walker_fn: Callable[[P, D, jnp.ndarray], Tuple[jnp.float32, D, jnp.ndarray]],
    update_param_fn: Callable[[D, P, S, jnp.ndarray], Tuple[P, S, Dict, jnp.ndarray]],
    key: jnp.ndarray,
    logdir: str = None,
    checkpoint_every: int = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_variance_scale: float = 10.0,
    nhistory_max: int = 200,
) -> Tuple[P, S, D, jnp.ndarray]:
    """Main Variational Monte Carlo loop routine.

    Variational Monte Carlo (VMC) can be generically viewed as minimizing a
    parameterized variational loss stochastically by sampling over a data distribution
    via Monte Carlo sampling. This function implements this idea at a high level, using
    a walker_fn to sample the data distribution, and passing the optimization step to a
    generic function `update_param_fn`.

    Args:
        params (pytree-like): model parameters which are trained
        optimizer_state (pytree-like): initial state of the optimizer
        data (pytree-like): initial data
        nchains (int): number of parallel MCMC chains being run. This can be difficult
            to infer from data, depending on the structure of data, whether data has
            been pmapped, etc.
        nepochs (int): number of parameter updates to do
        walker_fn (Callable): function which does a number of walker steps between each
            parameter update. Has the signature
            (data, params, key) -> (mean accept prob, new data, new key)
        update_param_fn (Callable): function which updates the parameters. Has signature
            (data, params, optimizer_state, key)
                -> (new_params, optimizer_state, dict: metrics, key).
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

    Returns:
        A tuple of (trained parameters, final optimizer state, final data, final key).
        These are the same structure as (params, optimizer_state, initial_data, key).
    """
    (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        checkpoint_writer,
    ) = utils.checkpoint.initialize_checkpointing(
        checkpoint_dir, nhistory_max, logdir, checkpoint_every
    )

    for epoch in range(nepochs):
        # Save state for checkpointing at the start of the epoch for two reasons:
        # 1. To save the model that generates the best energy and variance metrics,
        # rather than the model one parameter UPDATE after the best metrics.
        # 2. To ensure a fully consistent state can be reloading from a checkpoint, and
        # the exact subsequent behavior can be reproduced (if run on same machine).
        old_params = params
        old_state = optimizer_state
        old_data = data
        old_key = key

        accept_ratio, data, key = walker_fn(params, data, key)

        params, optimizer_state, metrics, key = update_param_fn(
            data, params, optimizer_state, key
        )

        if metrics is None:  # don't checkpoint if no metrics to checkpoint
            continue

        metrics["accept_ratio"] = accept_ratio

        (
            checkpoint_metric,
            checkpoint_str,
        ) = utils.checkpoint.save_metrics_and_handle_checkpoints(
            epoch,
            old_params,
            old_state,
            old_data,
            old_key,
            metrics,
            nchains,
            running_energy_and_variance,
            checkpoint_writer,
            checkpoint_metric,
            logdir=logdir,
            variance_scale=checkpoint_variance_scale,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )
        utils.checkpoint.log_vmc_loop_state(epoch, metrics, checkpoint_str)

    checkpoint_writer.close_and_await()
    return params, optimizer_state, data, key
