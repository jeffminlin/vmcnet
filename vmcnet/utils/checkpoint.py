"""Utilities for checkpointing and logging the VMC loop.

Running queues of energy and variance histories are tracked, along with their averages.
Unlike many of the other routines in this package, these are not pure functions, as they
modify the RunningMetrics inside RunningEnergyVariance.
"""
import logging
import os
from collections import deque
from dataclasses import dataclass, field
import threading
import queue
from typing import Dict, NamedTuple, Optional, Tuple, TypeVar

import jax.numpy as jnp
import vmcnet.utils.io as io

from .types import CheckpointData

# represents a pytree or pytree-like object containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
P = TypeVar("P")  # represents a pytree or pytree-like object containing model params
S = TypeVar("S")  # represents optimizer state

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


class CheckpointWriter:
    """A simple asynchronous writer for saving checkpoints during training.

    Spins up a thread for the file IO so that it does not block the main line of the
    training procedure. While Python threads do not provide true parallelism of CPU
    computations across cores, they do allow us to write to files and run Jax
    computations simultaneously.
    """

    def __init__(self):
        """Create a new CheckpointWriter."""
        self._checkpoint_thread = threading.Thread(target=self._run_checkpoint_thread)
        self._done = False
        self._queue = queue.Queue()

    def _run_checkpoint_thread(self):
        while True and not self._done:
            try:
                # Timeout should be long enough to avoid using unnecessarily CPU cycles
                # on this thread, but short enough that we don't mind waiting for the
                # this to time out at the end of the training loop.
                (directory, name, checkpoint_data) = self._queue.get(timeout=0.5)
                io.save_vmc_state(directory, name, checkpoint_data)
            except queue.Empty:
                continue

        # Once the checkpointing is "done", still need to write any remaining items in
        # the queue to disc.
        while not self._queue.empty():
            (directory, name, checkpoint_data) = self._queue.get()
            io.save_vmc_state(directory, name, checkpoint_data)

    def initialize(self):
        """Initialize the CheckpointWriter by starting its internal thread."""
        self._checkpoint_thread.start()

    def save_checkpoint(
        self,
        directory: str,
        name: str,
        checkpoint_data: CheckpointData,
    ):
        """Queue up a checkpoint to be written to disc.

        Args:
            directory (str): directory in which to write the checkpoint
            name (str): filename for the checkpoint
            epoch (int): epoch at which checkpoint is being saved
            data (pytree or jnp.ndarray): walker data to save
            params (pytree): model parameters to save
            optimizer_state (pytree): optimizer state to save
            key (jnp.ndarray): RNG key, used to reproduce exact behavior from checkpoint
        """
        self._queue.put((directory, name, checkpoint_data))

    def close_and_await(self):
        """Stop the thread by setting a flag, and return once it gets the message."""
        self._done = True
        self._checkpoint_thread.join()


def initialize_checkpointing(
    checkpoint_dir: str,
    nhistory_max: int,
    logdir: str = None,
    checkpoint_every: int = None,
) -> Tuple[str, jnp.float32, RunningEnergyVariance, CheckpointWriter]:
    """Initialize checkpointing objects.

    A suffix is added to the checkpointing directory if one with the same name already
    exists in the logdir.

    The checkpointing metric (error-adjusted running energy average) is initialized to
    infinity, and empty arrays are initialized in running_energy_and_variance.
    """
    if logdir is not None:
        logging.info("Saving to " + logdir)
        os.makedirs(logdir, exist_ok=True)
        if checkpoint_every is not None:
            checkpoint_dir = io.add_suffix_for_uniqueness(checkpoint_dir, logdir)
            os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=False)

    checkpoint_metric = jnp.inf
    running_energy_and_variance = RunningEnergyVariance(
        RunningMetric(nhistory_max), RunningMetric(nhistory_max)
    )

    checkpoint_writer = CheckpointWriter()
    checkpoint_writer.initialize()

    return (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        checkpoint_writer,
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


def save_metrics_and_handle_checkpoints(
    epoch: int,
    params: P,
    optimizer_state: S,
    data: D,
    key: jnp.ndarray,
    metrics: Dict,
    nchains: int,
    running_energy_and_variance: RunningEnergyVariance,
    checkpoint_writer: CheckpointWriter,
    checkpoint_metric: jnp.float32,
    best_checkpoint_every: int,
    logdir: str = None,
    variance_scale: float = 10.0,
    checkpoint_every: int = None,
    best_checkpoint_data: Optional[CheckpointData] = None,
    checkpoint_dir: str = "checkpoints",
) -> Tuple[jnp.float32, str, Optional[CheckpointData]]:
    """Checkpoint the current state of the VMC loop.

    There are two situations to checkpoint:
        1) Regularly, every x epochs, to handle job preemption and track
        parameters/metrics/state over time, and
        2) Whenever a checkpoint metric improves, i.e. the error adjusted running
        average of the energy.

    This is not a pure function, as it modifies the running energy and variance history.

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

    Returns:
        (jnp.float32, str, CheckpointData): best error-adjusted energy average, then
        string indicating if checkpointing has been done, then new best checkpoint data,
        or None.
    """
    checkpoint_str = ""
    if logdir is None or metrics is None:
        # do nothing
        return checkpoint_metric, checkpoint_str, best_checkpoint_data

    checkpoint_str = save_metrics_and_regular_checkpoint(
        epoch,
        params,
        optimizer_state,
        data,
        key,
        metrics,
        logdir,
        checkpoint_writer,
        checkpoint_dir,
        checkpoint_str,
        checkpoint_every,
    )

    (
        checkpoint_str,
        error_adjusted_running_avg,
        new_best_checkpoint_data,
    ) = track_and_save_best_checkpoint(
        epoch,
        params,
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
    )


def track_and_save_best_checkpoint(
    epoch: int,
    params: P,
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
    best_checkpoint_every: int,
    best_checkpoint_data: Optional[CheckpointData],
) -> Tuple[str, jnp.float32, Optional[CheckpointData]]:
    """Update running avgs and checkpoint if the error-adjusted energy avg improves.

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
            params,
            optimizer_state,
            key,
        )

    if (epoch + 1) % best_checkpoint_every == 0 and best_checkpoint_data is not None:
        checkpoint_writer.save_checkpoint(
            logdir, CHECKPOINT_FILE_NAME, best_checkpoint_data
        )
        checkpoint_str = checkpoint_str + ", best weights saved"
        best_checkpoint_data = None

    return checkpoint_str, error_adjusted_running_avg, best_checkpoint_data


def save_metrics_and_regular_checkpoint(
    epoch: int,
    params: P,
    optimizer_state: S,
    data: D,
    key: jnp.ndarray,
    metrics: Dict,
    logdir: str,
    checkpoint_writer: CheckpointWriter,
    checkpoint_dir: str,
    checkpoint_str: str,
    checkpoint_every: int = None,
) -> str:
    """Save current metrics to file, and save model state regularly.

    This currently touches the disk repeatedly, once for each metric, which is probably
    fairly inefficient, especially if called every epoch (as it currently is in
    :func:`~vmcnet.train.vmc.vmc_loop`).

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
        checkpoint_str (str): string indicating whether checkpointing has previously
            occurred
        logdir (str): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        checkpoint_dir (str): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, this function doesn't save the model state. Defaults to None.

    Returns:
        str: previous checkpointing string, with additional info if this function
        did checkpointing
    """
    # TODO(Jeffmin): do something more efficient than writing separately to disk for
    # every metric, maybe something like pandas? Also maybe shouldn't be every epoch.
    # Might be better to switch to something like TensorBoard -- need to do research
    for metric, metric_val in metrics.items():
        io.append_metric_to_file(metric_val, logdir, metric)

    if checkpoint_every is not None:
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_writer.save_checkpoint(
                os.path.join(logdir, checkpoint_dir),
                str(epoch + 1) + ".npz",
                (
                    epoch,
                    data,
                    params,
                    optimizer_state,
                    key,
                ),
            )
            checkpoint_str = checkpoint_str + ", regular ckpt saved"

    return checkpoint_str


def log_vmc_loop_state(epoch: int, metrics: Dict, checkpoint_str: str) -> None:
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
