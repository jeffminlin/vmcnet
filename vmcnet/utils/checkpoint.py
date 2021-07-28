"""Utilities for checkpointing and logging the VMC loop.

Running queues of energy and variance histories are tracked, along with their averages.
Unlike many of the other routines in this package, these are not pure functions, as they
modify the RunningMetrics inside RunningEnergyVariance.
"""
import logging
import os
import queue
import threading
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Generic, NamedTuple, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp

import vmcnet.utils.distribute as distribute
import vmcnet.utils.io as io
from vmcnet.utils.typing import CheckpointData, D, P, S

T = TypeVar("T")

CHECKPOINT_FILE_NAME = "checkpoint.npz"


@dataclass
class RunningMetric:
    """Running history and average of a metric for checkpointing purposes.

    Attributes:
        nhistory_max (int): maximum length of the running history to keep when adding
            new values
        avg (jnp.float32): the running average, should be equal to
            jnp.mean(self.history). Stored here to avoid recomputation when new values
            are added
        history (deque[jnp.float32]): the running history of the metric
    """

    nhistory_max: int
    avg: jnp.float32 = 0.0
    history: deque[jnp.float32] = field(default_factory=deque)

    def move_history_window(self, new_value: jnp.float32):
        """Append new value to running history, remove oldest if length > nhistory_max.

        Args:
            new_value (jnp.float32): new value to insert into the history
        """
        if self.nhistory_max <= 0:
            return

        history_length = len(self.history)
        self.history.append(new_value)

        self_sum = history_length * self.avg
        self_sum += new_value
        history_length += 1

        if history_length >= self.nhistory_max:
            oldest_value = self.history.popleft()
            self_sum -= oldest_value
            history_length -= 1

        self.avg = self_sum / history_length


class RunningEnergyVariance(NamedTuple):
    """Running energy history and variance history, packaged together."""

    energy: RunningMetric
    variance: RunningMetric


class ThreadedWriter(Generic[T]):
    """A simple asynchronous writer to handle file io during training.

    Spins up a thread for the file IO so that it does not block the main line of the
    training procedure. While Python threads do not provide true parallelism of CPU
    computations across cores, they do allow us to write to files and run Jax
    computations simultaneously.
    """

    def __init__(self):
        """Create a new ThreadedWriter."""
        self._thread = threading.Thread(target=self._run_thread)
        self._done = False
        self._queue = queue.Queue()

    @abstractmethod
    def write_out_data(self, directory: str, name: str, data_to_save: T):
        """Abstract method which saves a piece of data pulled from the queue.

        Args:
            directory (str): directory in which to write the checkpoint
            name (str): filename for the checkpoint
            data_to_save (Any): data to save
        """
        pass

    def _run_thread(self):
        while True and not self._done:
            try:
                # Timeout should be long enough to avoid using unnecessarily CPU cycles
                # on this thread, but short enough that we don't mind waiting for the
                # this to time out at the end of the training loop.
                (directory, name, data_to_save) = self._queue.get(timeout=0.5)
                self.write_out_data(directory, name, data_to_save)
            except queue.Empty:
                continue

        # Once the checkpointing is "done", still need to write any remaining items in
        # the queue to disc.
        while not self._queue.empty():
            (directory, name, data_to_save) = self._queue.get()
            self.write_out_data(directory, name, data_to_save)

    def initialize(self):
        """Initialize the ThreadedWriter by starting its internal thread."""
        self._thread.start()

    def save_data(self, directory: str, name: str, data_to_save: T):
        """Queue up data to be written to disc."""
        self._queue.put((directory, name, data_to_save))

    def close_and_await(self):
        """Stop the thread by setting a flag, and return once it gets the message."""
        self._done = True
        self._thread.join()

    def __enter__(self):
        """Enter a ThreadedWriter's context, starting up a thread."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Wait for the thread to finish, then leave the ThreadedWriter's context."""
        self.close_and_await()


class CheckpointWriter(ThreadedWriter[CheckpointData]):
    """A ThreadedWriter for saving checkpoints during training."""

    def write_out_data(
        self, directory: str, name: str, checkpoint_data: CheckpointData
    ):
        """Save checkpoint data.

        Args:
            directory (str): directory in which to write the checkpoint
            name (str): filename for the checkpoint

            checkpoint_data (CheckpointData): checkpoint data which contains:
                epoch (int): epoch at which checkpoint is being saved
                data (pytree or jnp.ndarray): walker data to save
                params (pytree): model parameters to save
                optimizer_state (pytree): optimizer state to save
                key (jnp.ndarray): RNG key, used to reproduce exact behavior from
                    checkpoint
        """
        io.save_vmc_state(directory, name, checkpoint_data)

    def save_data(self, directory: str, name: str, checkpoint_data: CheckpointData):
        """Queue up checkpoint data to be written to disc."""
        checkpoint_data = io.process_checkpoint_data_for_saving(checkpoint_data)
        # Move data to CPU to avoid clogging up GPU memory with queued checkpoints
        checkpoint_data = jax.device_put(checkpoint_data, jax.devices("cpu")[0])
        super().save_data(directory, name, checkpoint_data)


# TODO: Write metrics out more elegantly (maybe to one csv, if we can get a list of
# metrics beforehand, or not quite as frequently perhaps)
class MetricsWriter(ThreadedWriter[Dict]):
    """A ThreadedWriter for saving metrics during training."""

    def write_out_data(self, directory: str, name: str, metrics: Dict):
        """Save metrics to individual text files."""
        del name  # unused, each metric gets its own file
        for metric, metric_val in metrics.items():
            io.append_metric_to_file(metric_val, directory, metric)


def initialize_checkpointing(
    checkpoint_dir: str,
    nhistory_max: int,
    logdir: str = None,
    checkpoint_every: int = None,
) -> Tuple[str, jnp.float32, RunningEnergyVariance, Optional[CheckpointData], bool]:
    """Initialize checkpointing objects.

    A suffix is added to the checkpointing directory if one with the same name already
    exists in the logdir.

    The checkpointing metric (error-adjusted running energy average) is initialized to
    infinity, and empty arrays are initialized in running_energy_and_variance. The
    best checkpoint data is initialized to None, and saved_nan_checkpoint is initialized
    to False.
    """
    if logdir is not None:
        logging.info("Saving to %s", logdir)
        os.makedirs(logdir, exist_ok=True)
        if checkpoint_every is not None:
            checkpoint_dir = io.add_suffix_for_uniqueness(checkpoint_dir, logdir)
            os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=False)

    checkpoint_metric = jnp.inf
    running_energy_and_variance = RunningEnergyVariance(
        RunningMetric(nhistory_max), RunningMetric(nhistory_max)
    )
    best_checkpoint_data = None
    saved_nan_checkpoint = False

    return (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        best_checkpoint_data,
        saved_nan_checkpoint,
    )


def finish_checkpointing(
    checkpoint_writer: CheckpointWriter,
    best_checkpoint_data: CheckpointData = None,
    logdir: str = None,
):
    """Save any final checkpoint data to the CheckpointWriter."""
    if logdir is not None and best_checkpoint_data is not None:
        checkpoint_writer.save_data(logdir, CHECKPOINT_FILE_NAME, best_checkpoint_data)


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


@jax.jit
def _check_for_nans(metrics: Dict, new_params: P) -> bool:
    # Jax logical constructs are used here to enable jitting, which hopefully gives
    # some performance benefit when checkpoint_if_nans is true.
    metrics_nans = jnp.logical_or(
        jnp.isnan(metrics["energy_noclip"]), jnp.isnan(metrics["variance_noclip"])
    )
    new_params = distribute.get_first_if_distributed(new_params)
    params_nans = jnp.any(jnp.isnan(jax.flatten_util.ravel_pytree(new_params)[0]))
    return jnp.logical_or(metrics_nans, params_nans)


def _should_save_nans_checkpoint(
    metrics: Dict,
    new_params: P,
    checkpoint_if_nans: bool,
    only_checkpoint_first_nans: bool,
    saved_nans_checkpoint: bool,
) -> bool:
    # Be sure to check the boolean flags before looking at the metrics and params in
    # order to avoid extra work and also to ensure that no error is thrown in  the eval
    # phase when energy_noclip and variance_noclip will not be recorded as metrics.
    if not checkpoint_if_nans or (only_checkpoint_first_nans and saved_nans_checkpoint):
        return False

    return _check_for_nans(metrics, new_params)


def save_metrics_and_handle_checkpoints(
    epoch: int,
    old_params: P,
    new_params: P,
    optimizer_state: S,
    data: D,
    key: jnp.ndarray,
    metrics: Dict,
    nchains: int,
    running_energy_and_variance: RunningEnergyVariance,
    checkpoint_writer: CheckpointWriter,
    metrics_writer: MetricsWriter,
    checkpoint_metric: jnp.float32,
    logdir: Optional[str] = None,
    variance_scale: float = 10.0,
    checkpoint_every: Optional[int] = None,
    best_checkpoint_every: Optional[int] = None,
    best_checkpoint_data: Optional[CheckpointData[D, P, S]] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_if_nans: bool = False,
    only_checkpoint_first_nans: bool = True,
    saved_nans_checkpoint: bool = False,
) -> Tuple[jnp.float32, str, Optional[CheckpointData[D, P, S]], bool]:
    """Checkpoint the current state of the VMC loop.

    There are two situations to checkpoint:
        1) Regularly, every x epochs, to handle job preemption and track
        parameters/metrics/state over time, and
        2) Whenever a checkpoint metric improves, i.e. the error adjusted running
        average of the energy.

    This is not a pure function, as it modifies the running energy and variance history.

    Args:
        epoch (int): current epoch number
        old_params (pytree-like): model parameters, from before the update function.
            Needs to be serializable via `np.savez`.
        new_params (pytree-like): model parameters, from after the update function.
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
        running_energy_and_variance (RunningEnergyVariance): running history of energies
            and variances
        checkpoint_metric (jnp.float32): current best error adjusted running average of
            the energy history
        best_checkpoint_every (int): limit on how often to save best
            checkpoint, even if energy is improving. When the error-adjusted running avg
            of the energy improves, instead of immediately saving a checkpoint, we hold
            onto the data from that epoch in memory, and if it's still the best one when
            we hit an epoch which is a multiple of `best_checkpoint_every`, we save it
            then. This ensures we don't waste time saving best checkpoints too often
            when the energy is on a downward trajectory (as we hope it often is!).
            Defaults to 100.
        logdir (str, optional): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        variance_scale (float, optional): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`. Defaults to 10.0.
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, checkpoints are only saved when the error-adjusted running avg of the
            energy improves. Defaults to None.
        best_checkpoint_data (CheckpointData, optional): the data needed to save a
            checkpoint for the best energy observed so far.
        checkpoint_dir (str, optional): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_if_nans (bool, optional): whether to save checkpoints when
            nan energy values are recorded. Defaults to False.
        only_checkpoint_first_nans (bool, optional): whether to checkpoint only the
            first time nans are encountered, or every time. Useful to capture a nan
            checkpoint without risking writing too many checkpoints if the optimization
            starts to hit nans most or every epoch after some point. Only relevant if
            checkpoint_if_nans is True. Defaults to True.
        saved_nans_checkpoint (bool, optional): whether a nans checkpoint has already
            been saved. Only relevant if checkpoint_if_nans and
            only_checkpoint_first_nans are both True, and used in that case to decide
            whether to save further nans checkpoints or not. Defaults to False.

    Returns:
        (jnp.float32, str, CheckpointData, bool): best error-adjusted energy average,
        then string indicating if checkpointing has been done, then new best checkpoint
        data (or None), then the updated value of saved_nans_checkpoint.
    """
    checkpoint_str = ""
    if logdir is None or metrics is None:
        # do nothing
        return (
            checkpoint_metric,
            checkpoint_str,
            best_checkpoint_data,
            saved_nans_checkpoint,
        )

    checkpoint_str, saved_nans_checkpoint = save_metrics_and_regular_checkpoint(
        epoch,
        old_params,
        new_params,
        optimizer_state,
        data,
        key,
        metrics,
        logdir,
        checkpoint_writer,
        metrics_writer,
        checkpoint_dir,
        checkpoint_str,
        checkpoint_every,
        checkpoint_if_nans=checkpoint_if_nans,
        only_checkpoint_first_nans=only_checkpoint_first_nans,
        saved_nans_checkpoint=saved_nans_checkpoint,
    )

    (
        checkpoint_str,
        error_adjusted_running_avg,
        new_best_checkpoint_data,
    ) = track_and_save_best_checkpoint(
        epoch,
        old_params,
        optimizer_state,
        data,
        key,
        metrics,
        nchains,
        running_energy_and_variance,
        checkpoint_writer,
        checkpoint_metric,
        logdir,
        variance_scale,
        checkpoint_str,
        best_checkpoint_every,
        best_checkpoint_data,
    )

    return (
        jnp.minimum(error_adjusted_running_avg, checkpoint_metric),
        checkpoint_str,
        new_best_checkpoint_data,
        saved_nans_checkpoint,
    )


def track_and_save_best_checkpoint(
    epoch: int,
    old_params: P,
    optimizer_state: S,
    data: D,
    key: jnp.ndarray,
    metrics: Dict,
    nchains: int,
    running_energy_and_variance: RunningEnergyVariance,
    checkpoint_writer: CheckpointWriter,
    checkpoint_metric: jnp.float32,
    logdir: str,
    variance_scale: float,
    checkpoint_str: str,
    best_checkpoint_every: Optional[int] = None,
    best_checkpoint_data: Optional[CheckpointData[D, P, S]] = None,
) -> Tuple[str, jnp.float32, Optional[CheckpointData[D, P, S]]]:
    """Update running avgs and checkpoint if the error-adjusted energy avg improves.

    Args:
        epoch (int): current epoch number
        old_params (pytree-like): model parameters, from before the update function.
            Needs to be serializable via `np.savez`.
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
        running_energy_and_variance (RunningEnergyVariance): running history of energies
            and variances
        checkpoint_metric (jnp.float32): current best error adjusted running average of
            the energy history
        logdir (str): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        variance_scale (float): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`.
        checkpoint_str (str): string indicating whether checkpointing has previously
            occurred
        best_checkpoint_every (int, optional): limit on how often to save best
            checkpoint, even if energy is improving. When the error-adjusted running avg
            of the energy improves, instead of immediately saving a checkpoint, we hold
            onto the data from that epoch in memory, and if it's still the best one when
            we hit an epoch which is a multiple of `best_checkpoint_every`, we save it
            then. This ensures we don't waste time saving best checkpoints too often
            when the energy is on a downward trajectory (as we hope it often is!).
            Defaults to 100.
        best_checkpoint_data (CheckpointData, optional): the data needed to save a
            checkpoint for the best energy observed so far.

    Returns:
        (str, jnp.float32, CheckpointData): previous checkpointing string with
        additional info if this function did checkpointing, then best error-adjusted
        energy average, then new best checkpoint data, or None.
    """
    if best_checkpoint_every is not None:
        energy, variance = running_energy_and_variance

        energy.move_history_window(metrics["energy"])
        variance.move_history_window(metrics["variance"])
        error_adjusted_running_avg = get_checkpoint_metric(
            energy.avg, variance.avg, nchains * len(energy.history), variance_scale
        )

        if error_adjusted_running_avg < checkpoint_metric:
            best_checkpoint_data = (
                epoch,
                data,
                old_params,
                optimizer_state,
                key,
            )

        should_save_best_checkpoint = (epoch + 1) % best_checkpoint_every == 0
        if should_save_best_checkpoint and best_checkpoint_data is not None:
            checkpoint_writer.save_data(
                logdir, CHECKPOINT_FILE_NAME, best_checkpoint_data
            )
            checkpoint_str = checkpoint_str + ", best weights saved"
            best_checkpoint_data = None
    else:
        error_adjusted_running_avg = checkpoint_metric

    return checkpoint_str, error_adjusted_running_avg, best_checkpoint_data


def save_metrics_and_regular_checkpoint(
    epoch: int,
    old_params: P,
    new_params: P,
    optimizer_state: S,
    data: D,
    key: jnp.ndarray,
    metrics: Dict,
    logdir: str,
    checkpoint_writer: CheckpointWriter,
    metrics_writer: MetricsWriter,
    checkpoint_dir: str,
    checkpoint_str: str,
    checkpoint_every: int = None,
    checkpoint_if_nans: bool = False,
    only_checkpoint_first_nans: bool = True,
    saved_nans_checkpoint: bool = False,
) -> Tuple[str, bool]:
    """Save current metrics to file, and save model state regularly.

    This currently touches the disk repeatedly, once for each metric, which is probably
    fairly inefficient, especially if called every epoch (as it currently is in
    :func:`~vmcnet.train.vmc.vmc_loop`).

    Args:
        epoch (int): current epoch number
        old_params (pytree-like): model parameters, from before the update function.
            Needs to be serializable via `np.savez`.
        new_params (pytree-like): model parameters, from after the update function.
        optimizer_state (pytree-like): running state of the optimizer other than the
            trainable parameters. Needs to be serialiable via `np.savez`
        data (pytree-like): current mcmc data (e.g. position and amplitude data). Needs
            to be serializable via `np.savez`
        metrics (dict): dictionary of metrics. If this is not None, then it must include
            "energy" and "variance". Metrics are currently flattened and written to a
            row of a text file. See :func:`utils.io.write_metric_to_file`.
        checkpoint_str (str): string indicating whether checkpointing has previously
            occurred
        logdir (str): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        checkpoint_dir (str): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, this function doesn't save the model state. Defaults to None.
        checkpoint_if_nans (bool, optional): whether to save checkpoints when
            nan energy values are recorded. Defaults to False.
        only_checkpoint_first_nans (bool, optional): whether to checkpoint only the
            first time nans are encountered, or every time. Useful to capture a nan
            checkpoint without risking writing too many checkpoints if the optimization
            starts to hit nans most or every epoch after some point. Only relevant if
            checkpoint_if_nans is True. Defaults to True.
        saved_nans_checkpoint (bool, optional): whether a nans checkpoint has already
            been saved. Only relevant if checkpoint_if_nans and
            only_checkpoint_first_nans are both True, and used in that case to decide
            whether to save further nans checkpoints or not. Defaults to False.

    Returns:
        (str, bool): previous checkpointing string, with additional info if this
        function did checkpointing; followed by updated value of saved_nans_checkpoint.
    """
    metrics_writer.save_data(logdir, "", metrics)
    checkpoint_data = (epoch, data, old_params, optimizer_state, key)

    if checkpoint_every is not None:
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_writer.save_data(
                os.path.join(logdir, checkpoint_dir),
                str(epoch + 1) + ".npz",
                checkpoint_data,
            )
            checkpoint_str = checkpoint_str + ", regular ckpt saved"

    save_nans_checkpoint = _should_save_nans_checkpoint(
        metrics,
        new_params,
        checkpoint_if_nans,
        only_checkpoint_first_nans,
        saved_nans_checkpoint,
    )

    if save_nans_checkpoint:
        checkpoint_writer.save_data(
            os.path.join(logdir, checkpoint_dir),
            "nans_" + str(epoch + 1) + ".npz",
            checkpoint_data,
        )
        checkpoint_str = checkpoint_str + ", nans ckpt saved"
        saved_nans_checkpoint = True

    return checkpoint_str, saved_nans_checkpoint


def log_vmc_loop_state(epoch: int, metrics: Dict, checkpoint_str: str) -> None:
    """Log current energy, variance, and accept ratio, w/ optional unclipped values."""
    epoch_str = "Epoch %(epoch)5d"
    energy_str = "Energy: %(energy).5e"
    variance_str = "Variance: %(variance).5e"
    accept_ratio_str = "Accept ratio: %(accept_ratio).5f"

    if "energy_noclip" in metrics:
        energy_str = energy_str + " (%(energy_noclip).5e)"

    if "variance_noclip" in metrics:
        variance_str = variance_str + " (%(variance_noclip).5e)"

    info_out = ", ".join([epoch_str, energy_str, variance_str, accept_ratio_str])
    info_out = info_out + checkpoint_str

    logged_metrics = {"epoch": epoch + 1}
    logged_metrics.update(metrics)
    logging.info(info_out, logged_metrics)
