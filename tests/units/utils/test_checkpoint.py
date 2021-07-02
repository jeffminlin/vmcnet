"""Testing io routines."""

import jax
import jax.numpy as jnp
import vmcnet.utils.checkpoint as checkpoint
from tests.test_utils import make_dummy_data_params_and_key


def test_three_checkpoints(mocker):
    """Test saving three checkpoints using the asynchronous CheckpointWriter."""
    # Create fake data
    directory = "/fake/directory"
    file_name1 = "checkpoint_file1.npz"
    file_name2 = "checkpoint_file2.npz"
    file_name3 = "checkpoint_file3.npz"
    epoch = 0
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(jax.local_device_count() * 2)}
    opt_state = {"momentum": 2.0}
    checkpoint_data = (epoch, data, params, opt_state, key)

    # Create checkpoint writer and mock out save_vmc_state method
    checkpoint_writer = checkpoint.CheckpointWriter()
    mock_save_vmc = mocker.patch("vmcnet.utils.io.save_vmc_state")

    # Initialize checkpoint writer and save three checkpoints
    checkpoint_writer.initialize()
    checkpoint_writer.save_data(directory, file_name1, checkpoint_data)
    checkpoint_writer.save_data(directory, file_name2, checkpoint_data)
    checkpoint_writer.save_data(directory, file_name3, checkpoint_data)
    checkpoint_writer.close_and_await()

    expected_calls = [
        mocker.call(directory, file_name1, checkpoint_data),
        mocker.call(directory, file_name2, checkpoint_data),
        mocker.call(directory, file_name3, checkpoint_data),
    ]
    assert mock_save_vmc.call_args_list == expected_calls


def test_save_best_checkpoint(mocker):
    """Test saving three checkpoints using the asynchronous CheckpointWriter."""
    # Create fake data
    directory = "/fake/directory"
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(jax.local_device_count() * 2)}
    opt_state = {"momentum": 2.0}
    running_energy_and_variance = checkpoint.RunningEnergyVariance(
        checkpoint.RunningMetric(50), checkpoint.RunningMetric(50)
    )

    def get_checkpoint_data(epoch):
        return (epoch, data, params, opt_state, key)

    # Create checkpoint writer and mock out metrics and checkpointing
    checkpoint_writer = checkpoint.CheckpointWriter()
    mock_save_checkpoint = mocker.patch.object(checkpoint_writer, "save_data")
    mock_get_metrics = mocker.patch("vmcnet.utils.checkpoint.get_checkpoint_metric")

    # Pull out simple helper function since only epoch, best_metric, and
    # best_checkpoint_data will be changed in the loop below.
    def track_and_save_best(epoch, checkpoint_metric, best_checkpoint_data):
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
            directory,
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
    for i in range(9):
        mock_get_metrics.return_value = per_epoch_avg[i]
        (
            _,
            best_metric,
            best_checkpoint_data,
        ) = track_and_save_best(i, checkpoint_metric[i], best_checkpoint_data)

    # Checkpoints should only be saved from epochs 1 and 5
    expected_calls = [
        mocker.call(directory, checkpoint.CHECKPOINT_FILE_NAME, get_checkpoint_data(1)),
        mocker.call(directory, checkpoint.CHECKPOINT_FILE_NAME, get_checkpoint_data(5)),
    ]
    assert mock_save_checkpoint.call_args_list == expected_calls
