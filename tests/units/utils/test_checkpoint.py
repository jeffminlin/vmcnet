"""Testing io routines."""
import os
from typing import List, Tuple

import jax
import jax.numpy as jnp

import vmcnet.utils.checkpoint as checkpoint
from vmcnet.utils.typing import CheckpointData

from tests.test_utils import assert_pytree_allclose, make_dummy_data_params_and_key


def _get_fake_checkpoint_data(epoch: int):
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(jax.local_device_count() * 2)}
    opt_state = {"momentum": 2.0}

    return (epoch, data, params, opt_state, key)


def _assert_checkpoint_save_calls_equal(
    call_args_list: List, expected_calls: List[Tuple[str, str, CheckpointData]]
):
    assert len(call_args_list) == len(expected_calls)
    for i in range(len(call_args_list)):
        call = call_args_list[i]
        expected_call = expected_calls[i]
        assert call.args[0] == expected_call[0]
        assert call.args[1] == expected_call[1]
        assert_pytree_allclose(call.args[2], expected_call[2])


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
        (log_dir, file_name1, checkpoint_data),
        (log_dir, file_name2, checkpoint_data),
        (log_dir, file_name3, checkpoint_data),
    ]
    _assert_checkpoint_save_calls_equal(
        mock_write_out_data.call_args_list, expected_calls
    )


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
    with checkpoint.CheckpointWriter(is_pmapped=False) as checkpoint_writer:
        mock_save_checkpoint = mocker.patch.object(checkpoint_writer, "save_data")
        for epoch in range(9):
            mock_get_metrics.return_value = per_epoch_avg[epoch]
            (
                _,
                _,
                best_checkpoint_data,
            ) = track_and_save_best(
                epoch, checkpoint_writer, checkpoint_metric[epoch], best_checkpoint_data
            )

    # Checkpoints should only be saved from epochs 1 and 5
    expected_calls = [
        (log_dir, checkpoint.BEST_CHECKPOINT_FILE_NAME, _get_fake_checkpoint_data(1)),
        (log_dir, checkpoint.BEST_CHECKPOINT_FILE_NAME, _get_fake_checkpoint_data(5)),
    ]
    _assert_checkpoint_save_calls_equal(
        mock_save_checkpoint.call_args_list, expected_calls
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
