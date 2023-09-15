"""Main VMC loop."""
from typing import Tuple, Optional

import jax

from vmcnet.mcmc.metropolis import WalkerFn
from vmcnet.updates.params import UpdateParamFn
from vmcnet.utils.checkpoint import CheckpointWriter, MetricsWriter
import vmcnet.utils as utils
from vmcnet.utils.typing import D, GetAmplitudeFromData, P, PRNGKey, S


def vmc_loop(
    params: P,
    optimizer_state: S,
    data: D,
    nchains: int,
    nepochs: int,
    walker_fn: WalkerFn[P, D],
    update_param_fn: UpdateParamFn[P, D, S],
    key: PRNGKey,
    logdir: Optional[str] = None,
    checkpoint_every: Optional[int] = 1000,
    best_checkpoint_every: Optional[int] = 100,
    checkpoint_dir: str = "checkpoints",
    checkpoint_variance_scale: float = 10.0,
    check_for_nans: bool = False,
    record_amplitudes: bool = False,
    get_amplitude_fn: Optional[GetAmplitudeFromData[D]] = None,
    nhistory_max: int = 200,
    is_pmapped=True,
    start_epoch: int = 0,
) -> Tuple[P, S, D, PRNGKey, bool]:
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
        key (PRNGKey): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn
        logdir (str, optional): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, checkpoints are only saved when the error-adjusted running avg of the
            energy improves. Defaults to 1000.
        best_checkpoint_every (int, optional): limit on how often to save best
            checkpoint, even if energy is improving. When the error-adjusted running avg
            of the energy improves, instead of immediately saving a checkpoint, we hold
            onto the data from that epoch in memory, and if it's still the best one when
            we hit an epoch which is a multiple of `best_checkpoint_every`, we save it
            then. This ensures we don't waste time saving best checkpoints too often
            when the energy is on a downward trajectory (as we hope it often is!).
            Defaults to 100.
        checkpoint_dir (str, optional): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_variance_scale (float, optional): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`. Defaults to 10.0.
        check_for_nans (bool, optional): whether to check for nans in the vmc loop. If
            so, then after nans are detected, a checkpoint will be saved and the loop
            will be aborted. Defaults to False.
        nhistory_max (int, optional): How much history to keep in the running histories
            of the energy and variance. Defaults to 200.

    Returns:
        A tuple of (trained parameters, final optimizer state, final data, final key,
        nans_detected). The first four entries are the same structure as
        (params, optimizer_state, initial_data, key).
    """
    (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        best_checkpoint_data,
    ) = utils.checkpoint.initialize_checkpointing(
        checkpoint_dir, nhistory_max, logdir, checkpoint_every
    )
    nans_detected = False

    with CheckpointWriter(
        is_pmapped
    ) as checkpoint_writer, MetricsWriter() as metrics_writer:
        for epoch in range(start_epoch, nepochs):
            # Save state for checkpointing at the start of the epoch for two reasons:
            # 1. To save the model that generates the best energy and variance metrics,
            # rather than the model one parameter UPDATE after the best metrics.
            # 2. To ensure a fully consistent state can be reloaded from a checkpoint, &
            # the exact subsequent behavior can be reproduced (if run on same machine).
            # NOTE: jax deletes the old arrays if we don't make copies.
            old_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            old_state = jax.tree_util.tree_map(lambda x: x.copy(), optimizer_state)
            old_data = data.copy()
            old_key = key.copy()

            accept_ratio, data, key = walker_fn(params, data, key)

            params, data, optimizer_state, metrics, key = update_param_fn(
                params, data, optimizer_state, key
            )

            # Don't checkpoint if no metrics to checkpoint
            if metrics is None:
                continue

            metrics["accept_ratio"] = accept_ratio

            (
                checkpoint_metric,
                checkpoint_str,
                best_checkpoint_data,
                nans_detected,
            ) = utils.checkpoint.save_metrics_and_handle_checkpoints(
                epoch,
                old_params,
                params,
                old_state,
                old_data,
                data,
                old_key,
                metrics,
                nchains,
                running_energy_and_variance,
                checkpoint_writer,
                metrics_writer,
                checkpoint_metric,
                logdir=logdir,
                variance_scale=checkpoint_variance_scale,
                checkpoint_every=checkpoint_every,
                best_checkpoint_every=best_checkpoint_every,
                best_checkpoint_data=best_checkpoint_data,
                checkpoint_dir=checkpoint_dir,
                check_for_nans=check_for_nans,
                record_amplitudes=record_amplitudes,
                get_amplitude_fn=get_amplitude_fn,
            )
            utils.checkpoint.log_vmc_loop_state(epoch, metrics, checkpoint_str)

            if nans_detected:
                break

        utils.checkpoint.finish_checkpointing(
            checkpoint_writer, best_checkpoint_data, logdir
        )

    return params, optimizer_state, data, key, nans_detected
