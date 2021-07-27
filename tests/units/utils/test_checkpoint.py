"""Testing io routines."""
import os
from typing import List

import jax
import jax.numpy as jnp

import vmcnet.utils.checkpoint as checkpoint

from tests.test_utils import make_dummy_data_params_and_key

# These are defined as constants here so that simple equality checks can be done
# between mocker.call_args_list values and reconstructed lists of mocker.call() values.
# This doesn't feel ideal but I haven't yet found a better way to make these tests work
# without going deep into the structures of the arguments to compare individual values.
(_, PARAMS, KEY) = make_dummy_data_params_and_key()
DATA = {"position": jnp.arange(jax.local_device_count() * 2)}
OPT_STATE = {"momentum": 2.0}


def _get_fake_checkpoint_data(epoch: int):
    return (epoch, DATA, PARAMS, OPT_STATE, KEY)


def _get_fake_filepaths():
    log_dir = "/fake/directory"
    checkpoint_dir_name = "checkpoints"
    checkpoint_dir = os.path.join(log_dir, checkpoint_dir_name)
    return log_dir, checkpoint_dir_name, checkpoint_dir


def test_three_calls_to_threaded_writer(mocker):
    """Test saving three fake checkpoints using the asynchronous ThreadedWriter."""
    # Create fake data
    log_dir, _, _ = _get_fake_filepaths()
    file_name1 = "checkpoint_file1.npz"
    file_name2 = "checkpoint_file2.npz"
    file_name3 = "checkpoint_file3.npz"
    checkpoint_data = _get_fake_checkpoint_data(0)

    # Mock out save_vmc_state method
    mock_write_out_data = mocker.patch(
        "vmcnet.utils.checkpoint.ThreadedWriter.write_out_data"
    )

    # Initialize checkpoint writer and save three checkpoints
    with checkpoint.ThreadedWriter() as threaded_writer:
        threaded_writer.save_data(log_dir, file_name1, checkpoint_data)
        threaded_writer.save_data(log_dir, file_name2, checkpoint_data)
        threaded_writer.save_data(log_dir, file_name3, checkpoint_data)

    expected_calls = [
        mocker.call(log_dir, file_name1, checkpoint_data),
        mocker.call(log_dir, file_name2, checkpoint_data),
        mocker.call(log_dir, file_name3, checkpoint_data),
    ]
    assert mock_write_out_data.call_args_list == expected_calls


def test_save_best_checkpoint(mocker):
    """Test saving best checkpoints using the asynchronous CheckpointWriter."""
    # Create fake data
    log_dir, _, _ = _get_fake_filepaths()
    (_, data, params, opt_state, key) = _get_fake_checkpoint_data(0)
    running_energy_and_variance = checkpoint.RunningEnergyVariance(
        checkpoint.RunningMetric(50), checkpoint.RunningMetric(50)
    )
    # Create checkpoint writer and mock out metrics and checkpointing
    mock_get_metrics = mocker.patch("vmcnet.utils.checkpoint.get_checkpoint_metric")

    # Pull out simple helper function since only epoch, best_metric, and
    # best_checkpoint_data will be changed in the loop below.
    def track_and_save_best(
        epoch, checkpoint_writer, checkpoint_metric, best_checkpoint_data
    ):
        best_checkpoint_every = 3
        return checkpoint.track_and_save_best_checkpoint(
            epoch,
            params,
            opt_state,
            data,
            key,
            {"energy": 1.0, "variance": 1.0},
            10,
            running_energy_and_variance,
            checkpoint_writer,
            checkpoint_metric,
            log_dir,
            1.0,
            "",
            best_checkpoint_every,
            best_checkpoint_data,
        )

    best_checkpoint_data = None
    per_epoch_avg = [2, 1, 3, 2, 1, 0.5, 3, 2, 3]
    # checkpoint_metric is running min of previous per_epoch_avgs
    checkpoint_metric = [jnp.inf, 2, 1, 1, 1, 1, 0.5, 0.5, 0.5]

    # Run 9 "epochs", substituting the mocked metrics in place of real ones
    with checkpoint.CheckpointWriter() as checkpoint_writer:
        mock_save_checkpoint = mocker.patch.object(checkpoint_writer, "save_data")
        for epoch in range(9):
            mock_get_metrics.return_value = per_epoch_avg[epoch]
            (_, _, best_checkpoint_data,) = track_and_save_best(
                epoch, checkpoint_writer, checkpoint_metric[epoch], best_checkpoint_data
            )

    # Checkpoints should only be saved from epochs 1 and 5
    expected_calls = [
        mocker.call(
            log_dir, checkpoint.CHECKPOINT_FILE_NAME, _get_fake_checkpoint_data(1)
        ),
        mocker.call(
            log_dir, checkpoint.CHECKPOINT_FILE_NAME, _get_fake_checkpoint_data(5)
        ),
    ]
    assert mock_save_checkpoint.call_args_list == expected_calls


def _test_nans_checkpointing(
    mocker,
    checkpoint_if_nans: bool,
    only_checkpoint_first_nans: bool,
    metric_nans: List[bool],
    new_param_nans: List[bool],
    expected_calls: List,
) -> None:
    (_, data, old_params, opt_state, key) = _get_fake_checkpoint_data(0)
    log_dir, checkpoint_dir_name, _ = _get_fake_filepaths()

    # Pull out simple helper function since only a few args will change during the test
    def save_regular_checkpoints(
        epoch,
        checkpoint_writer,
        metrics_writer,
        metrics,
        new_params,
        saved_nans_checkpoint,
    ):
        return checkpoint.save_metrics_and_regular_checkpoint(
            epoch,
            old_params,
            new_params,
            opt_state,
            data,
            key,
            metrics,
            log_dir,
            checkpoint_writer,
            metrics_writer,
            checkpoint_dir_name,
            "",
            checkpoint_if_nans=checkpoint_if_nans,
            only_checkpoint_first_nans=only_checkpoint_first_nans,
            saved_nans_checkpoint=saved_nans_checkpoint,
        )

    no_nans_met = {"energy_noclip": 2.0, "variance_noclip": 3.0}
    nans_met = {"energy_noclip": jnp.nan, "variance_noclip": 4.0}
    metrics_list = [nans_met if is_nan else no_nans_met for is_nan in metric_nans]

    no_nans_params = old_params
    nans_params = {
        "params": {"a": jnp.ones((2, 2)), "b": jnp.array([2.0, 3.0, jnp.nan])}
    }
    new_params_list = [
        nans_params if is_nan else no_nans_params for is_nan in new_param_nans
    ]

    with checkpoint.MetricsWriter() as metrics_writer:
        with checkpoint.CheckpointWriter() as checkpoint_writer:
            mock_save_checkpoint = mocker.patch.object(checkpoint_writer, "save_data")
            mocker.patch.object(metrics_writer, "save_data")
            saved_nans_checkpoint = False

            for epoch in range(len(metrics_list)):
                _, saved_nans_checkpoint = save_regular_checkpoints(
                    epoch,
                    checkpoint_writer,
                    metrics_writer,
                    metrics_list[epoch],
                    new_params_list[epoch],
                    saved_nans_checkpoint,
                )

    assert mock_save_checkpoint.call_args_list == expected_calls


def test_nans_checkpointing_when_off(mocker):
    """Test no checkpoints are written if nans checkpointing is off."""
    metric_nans = [False, False, True, False, False, True]
    new_param_nans = [True, False, False, False, False, True]
    expected_calls = []
    checkpoint_if_nans = False
    only_checkpoint_first_nans = True
    _test_nans_checkpointing(
        mocker,
        checkpoint_if_nans,
        only_checkpoint_first_nans,
        metric_nans,
        new_param_nans,
        expected_calls,
    )


def test_checkpoint_first_nans_for_metric_nans(mocker):
    """Test checkpointing first nans only, when first nans come from metrics."""
    _, _, checkpoint_dir = _get_fake_filepaths()
    metric_nans = [False, False, True, False, False, True]
    new_param_nans = [False, False, False, False, True, False]
    expected_calls = [
        mocker.call(checkpoint_dir, "nans_3.npz", _get_fake_checkpoint_data(2))
    ]
    checkpoint_if_nans = True
    only_checkpoint_first_nans = True
    _test_nans_checkpointing(
        mocker,
        checkpoint_if_nans,
        only_checkpoint_first_nans,
        metric_nans,
        new_param_nans,
        expected_calls,
    )


def test_checkpoint_first_nans_for_param_nans(mocker):
    """Test checkpointing first nans only, when first nans come from params."""
    _, _, checkpoint_dir = _get_fake_filepaths()
    metric_nans = [False, False, False, True, False, True]
    new_param_nans = [False, True, True, False, True, False]
    expected_calls = [
        mocker.call(checkpoint_dir, "nans_2.npz", _get_fake_checkpoint_data(1))
    ]
    checkpoint_if_nans = True
    only_checkpoint_first_nans = True
    _test_nans_checkpointing(
        mocker,
        checkpoint_if_nans,
        only_checkpoint_first_nans,
        metric_nans,
        new_param_nans,
        expected_calls,
    )


def test_nans_checkpointing_when_checkpointing_all_nans(mocker):
    """Test all relevant checkpoints are written if checkpointing all nans."""
    _, _, checkpoint_dir = _get_fake_filepaths()
    metric_nans = [False, False, True, False, False, True]
    new_param_nans = [False, True, True, False, True, False]
    expected_calls = [
        mocker.call(checkpoint_dir, "nans_2.npz", _get_fake_checkpoint_data(1)),
        mocker.call(checkpoint_dir, "nans_3.npz", _get_fake_checkpoint_data(2)),
        mocker.call(checkpoint_dir, "nans_5.npz", _get_fake_checkpoint_data(4)),
        mocker.call(checkpoint_dir, "nans_6.npz", _get_fake_checkpoint_data(5)),
    ]
    checkpoint_if_nans = True
    only_checkpoint_first_nans = False
    _test_nans_checkpointing(
        mocker,
        checkpoint_if_nans,
        only_checkpoint_first_nans,
        metric_nans,
        new_param_nans,
        expected_calls,
    )


def test_metrics_saved_to_their_own_files(mocker):
    """Test that appending metrics to files is called properly from a MetricsWriter."""
    directory = "/fake/directory"
    metrics_list = [
        {
            "energy": -1.234 * i,
            "variance": 5.678 * i,
        }
        for i in range(1, 4)
    ]

    mock_append_metrics = mocker.patch("vmcnet.utils.io.append_metric_to_file")

    with checkpoint.MetricsWriter() as metrics_writer:
        metrics_writer.save_data(directory, "", metrics_list[0])
        metrics_writer.save_data(directory, "", metrics_list[1])
        metrics_writer.save_data(directory, "", metrics_list[2])

    expected_calls = [
        mocker.call(-1.234, directory, "energy"),
        mocker.call(5.678, directory, "variance"),
        mocker.call(-1.234 * 2, directory, "energy"),
        mocker.call(5.678 * 2, directory, "variance"),
        mocker.call(-1.234 * 3, directory, "energy"),
        mocker.call(5.678 * 3, directory, "variance"),
    ]

    assert mock_append_metrics.call_args_list == expected_calls
