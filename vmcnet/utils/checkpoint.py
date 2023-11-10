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

import chex
import jax
import jax.numpy as jnp

import vmcnet.utils.io as io
from vmcnet.utils.typing import (
    CheckpointData,
    D,
    GetAmplitudeFromData,
    P,
    PRNGKey,
    S,
)

T = TypeVar("T")

BEST_CHECKPOINT_FILE_NAME = "best_checkpoint.npz"
DEFAULT_CHECKPOINT_FILE_NAME = "best_checkpoint.npz"


@dataclass
class RunningMetric:
    """Running history and average of a metric for checkpointing purposes.

    Attributes:
        nhistory_max (int): maximum length of the running history to keep when adding
            new values
        avg (chex.Scalar): the running average, should be equal to
            jnp.mean(self.history). Stored here to avoid recomputation when new values
            are added
        history (deque[chex.Scalar]): the running history of the metric
    """

    nhistory_max: int
    avg: chex.Scalar = 0.0
    history: deque[chex.Scalar] = field(default_factory=deque)

    def move_history_window(self, new_value: chex.Scalar):
        """Append new value to running history, remove oldest if length > nhistory_max.

        Args:
            new_value (chex.Numeric): new value to insert into the history
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

    is_pmapped: bool

    def __init__(self, is_pmapped):
        """Init Checkpoint Writer."""
        super().__init__()
        self.is_pmapped = is_pmapped

    def write_out_data(
        self, directory: str, name: str, checkpoint_data: CheckpointData
    ):
        """Save checkpoint data.

        Args:
            directory (str): directory in which to write the checkpoint
            name (str): filename for the checkpoint

            checkpoint_data (CheckpointData): checkpoint data which contains:
                epoch (int): epoch at which checkpoint is being saved
                data (pytree or Array): walker data to save
                params (pytree): model parameters to save
                optimizer_state (pytree): optimizer state to save
                key (PRNGKey): RNG key, used to reproduce exact behavior from
                    checkpoint
        """
        io.save_vmc_state(directory, name, checkpoint_data)

    def save_data(self, directory: str, name: str, checkpoint_data: CheckpointData):
        """Queue up checkpoint data to be written to disc."""
        checkpoint_data = io.process_checkpoint_data_for_saving(
            checkpoint_data, self.is_pmapped
        )
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
    logdir: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
) -> Tuple[str, chex.Numeric, RunningEnergyVariance, Optional[CheckpointData]]:
    """Initialize checkpointing objects.

    A suffix is added to the checkpointing directory if one with the same name already
    exists in the logdir.

    The checkpointing metric (error-adjusted running energy average) is initialized to
    infinity, and empty arrays are initialized in running_energy_and_variance. The
    best checkpoint data is initialized to None.
    """
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        if checkpoint_every is not None:
            checkpoint_dir = io.add_suffix_for_uniqueness(checkpoint_dir, logdir)
            os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=False)

    checkpoint_metric = jnp.inf
    running_energy_and_variance = RunningEnergyVariance(
        RunningMetric(nhistory_max), RunningMetric(nhistory_max)
    )
    best_checkpoint_data = None

    return (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        best_checkpoint_data,
    )


def finish_checkpointing(
    checkpoint_writer: CheckpointWriter,
    best_checkpoint_data: Optional[CheckpointData] = None,
    logdir: Optional[str] = None,
):
    """Save any final checkpoint data to the CheckpointWriter."""
    if logdir is not None and best_checkpoint_data is not None:
        checkpoint_writer.save_data(
            logdir, BEST_CHECKPOINT_FILE_NAME, best_checkpoint_data
        )


def _add_amplitude_to_metrics_if_requested(
    metrics: Dict,
    data: D,
    record_amplitudes: bool,
    get_amplitude_fn: Optional[GetAmplitudeFromData[D]],
) -> None:
    if record_amplitudes:
        if get_amplitude_fn is None:
            raise ValueError(
                "record_amplitudes set to True, but get_amplitude_fn "
                "function is None."
            )
        amplitudes = get_amplitude_fn(data)
        metrics["amplitude_min"] = jnp.min(amplitudes)
        metrics["amplitude_max"] = jnp.max(amplitudes)


def get_checkpoint_metric(
    energy_running_avg: chex.Numeric,
    variance_running_avg: chex.Numeric,
    nsamples: int,
    variance_scale: float,
) -> chex.Numeric:
    """Get an error-adjusted running average of the energy for checkpointing.

    The parameter variance_scale can be tuned and probably should scale linearly with
    some estimate of the integrated autocorrelation. Higher means more allergic to high
    variance, lower means more allergic to high energies.

    Args:
        energy_running_avg (chex.Numeric): running average of the energy
        variance_running_avg (chex.Numeric): running average of the variance
        nsamples (int): total number of samples reflected in the running averages, equal
            to the number of parallel chains times the length of the history
        variance_scale (float): weight of the variance part of the checkpointing metric.
            The final effect on the variance part is to scale it by
            jnp.sqrt(variance_scale), i.e. to treat it like the integrated
            autocorrelation.

    Returns:
        chex.Numeric: error adjusted running average of the energy
    """
    # TODO(Jeffmin): eventually maybe put in some cheap best guess at the IAC?
    if variance_scale <= 0.0 or nsamples <= 0:
        return energy_running_avg

    effective_nsamples = nsamples / variance_scale
    return energy_running_avg + jnp.sqrt(variance_running_avg / effective_nsamples)


def _check_metrics_for_nans(metrics: Dict) -> chex.Numeric:
    return jnp.logical_or(
        jnp.isnan(metrics["energy_noclip"]), jnp.isnan(metrics["variance_noclip"])
    )


@jax.jit
def _check_for_nans(metrics: Dict, new_params: P) -> chex.Numeric:
    # Jax logical constructs are used here to enable jitting, which hopefully gives
    # some performance benefit for nans checkpointing.
    metrics_nans = _check_metrics_for_nans(metrics)
    params_nans = jnp.any(jnp.isnan(jax.flatten_util.ravel_pytree(new_params)[0]))
    return jnp.logical_or(metrics_nans, params_nans)


# TODO (ggoldsh): encapsulate the numerous settings passed into this function into some
# sort of checkpointing/logging object.
def save_metrics_and_handle_checkpoints(
    epoch: int,
    old_params: P,
    new_params: P,
    optimizer_state: S,
    old_data: D,
    new_data: D,
    key: PRNGKey,
    metrics: Dict,
    nchains: int,
    running_energy_and_variance: RunningEnergyVariance,
    checkpoint_writer: CheckpointWriter,
    metrics_writer: MetricsWriter,
    checkpoint_metric: chex.Numeric,
    logdir: Optional[str] = None,
    variance_scale: float = 10.0,
    checkpoint_every: Optional[int] = None,
    best_checkpoint_every: Optional[int] = None,
    best_checkpoint_data: Optional[CheckpointData[D, P, S]] = None,
    checkpoint_dir: str = "checkpoints",
    check_for_nans: bool = False,
    record_amplitudes: bool = False,
    get_amplitude_fn: Optional[GetAmplitudeFromData[D]] = None,
) -> Tuple[chex.Numeric, str, Optional[CheckpointData[D, P, S]], bool]:
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
        old_data (pytree-like): previous mcmc data (e.g. position and amplitude data).
            Needs to be serializable via `np.savez`
        new_data (pytree-like): new mcmc data (e.g. position and amplitude data). Needs
            to be serializable via `np.savez`
        metrics (dict): dictionary of metrics. If this is not None, then it must include
            "energy" and "variance". Metrics are currently flattened and written to a
            row of a text file. See :func:`utils.io.write_metric_to_file`.
        nchains (int): number of parallel MCMC chains being run. This can be difficult
            to infer from data, depending on the structure of data, whether data has
            been pmapped, etc.
        running_energy_and_variance (RunningEnergyVariance): running history of energies
            and variances
        checkpoint_metric (chex.Numeric): current best error adjusted running average of
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
        check_for_nans (bool, optional): whether to check for nans. Defaults to False.

    Returns:
        (chex.Numeric, str, CheckpointData, bool): best error-adjusted energy
        average, then string indicating if checkpointing has been done, then new best
        checkpoint data (or None), then whether nans were detected.
    """
    checkpoint_str = ""
    if logdir is None or metrics is None:
        # do nothing
        return (
            checkpoint_metric,
            checkpoint_str,
            best_checkpoint_data,
            False,
        )

    _add_amplitude_to_metrics_if_requested(
        metrics, new_data, record_amplitudes, get_amplitude_fn
    )

    (checkpoint_str, nans_detected) = save_metrics_and_regular_checkpoint(
        epoch,
        old_params,
        new_params,
        optimizer_state,
        old_data,
        key,
        metrics,
        logdir,
        checkpoint_writer,
        metrics_writer,
        checkpoint_dir,
        checkpoint_str,
        checkpoint_every,
        check_for_nans,
    )

    (
        checkpoint_str,
        error_adjusted_running_avg,
        new_best_checkpoint_data,
    ) = track_and_save_best_checkpoint(
        epoch,
        old_params,
        optimizer_state,
        old_data,
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
        nans_detected,
    )


def track_and_save_best_checkpoint(
    epoch: int,
    old_params: P,
    optimizer_state: S,
    data: D,
    key: PRNGKey,
    metrics: Dict,
    nchains: int,
    running_energy_and_variance: RunningEnergyVariance,
    checkpoint_writer: CheckpointWriter,
    checkpoint_metric: chex.Numeric,
    logdir: str,
    variance_scale: float,
    checkpoint_str: str,
    best_checkpoint_every: Optional[int] = None,
    best_checkpoint_data: Optional[CheckpointData[D, P, S]] = None,
) -> Tuple[str, chex.Numeric, Optional[CheckpointData[D, P, S]]]:
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
        checkpoint_metric (chex.Numeric): current best error adjusted running average of
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
        (str, chex.Numeric, CheckpointData): previous checkpointing string with
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
                logdir, BEST_CHECKPOINT_FILE_NAME, best_checkpoint_data
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
    key: PRNGKey,
    metrics: Dict,
    logdir: str,
    checkpoint_writer: CheckpointWriter,
    metrics_writer: MetricsWriter,
    checkpoint_dir: str,
    checkpoint_str: str,
    checkpoint_every: Optional[int] = None,
    check_for_nans: bool = False,
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
        check_for_nans (bool, optional): whether to check for nans. Defaults to False.

    Returns:
        (str, bool, bool): previous checkpointing string, with additional info if this
        function did checkpointing; followed by whether the metrics had nans.
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

    nans_detected = False
    if check_for_nans:
        nans_detected = _check_for_nans(metrics, new_params)

    if nans_detected:
        checkpoint_writer.save_data(
            os.path.join(logdir, checkpoint_dir),
            "nans_" + str(epoch + 1) + ".npz",
            checkpoint_data,
        )
        checkpoint_str = checkpoint_str + ", nans ckpt saved"

    return checkpoint_str, nans_detected


def log_vmc_loop_state(epoch: int, metrics: Dict, checkpoint_str: str) -> None:
    """Log current energy, variance, and accept ratio, w/ optional unclipped values."""
    epoch_str = "Epoch %(epoch)5d"
    energy_str = "Energy: %(energy).5e"
    variance_str = "Variance: %(variance).5e"
    accept_ratio_str = "Accept ratio: %(accept_ratio).5f"
    amplitude_str = ""

    if "energy_noclip" in metrics:
        energy_str = energy_str + " (%(energy_noclip).5e)"

    if "variance_noclip" in metrics:
        variance_str = variance_str + " (%(variance_noclip).5e)"

    if "amplitude_min" in metrics:
        amplitude_str = "Min/max amplitude: %(amplitude_min).2f/%(amplitude_max).2f"

    info_out = ", ".join(
        [epoch_str, energy_str, variance_str, accept_ratio_str, amplitude_str]
    )
    info_out = info_out + checkpoint_str

    logged_metrics = {"epoch": epoch + 1}
    logged_metrics.update(metrics)
    logging.info(info_out, logged_metrics)
