"""Testing io routines."""

import unittest.mock as mock

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

    # Create checkpoint writer and mock out save_vmc_state method
    checkpoint_writer = checkpoint.CheckpointWriter()
    save_vmc_stub = mocker.patch("vmcnet.utils.io.save_vmc_state")

    # Initialize checkpoint writer and save three checkpoints
    checkpoint_writer.initialize()
    checkpoint_writer.save_checkpoint(
        directory, file_name1, epoch, data, params, opt_state, key
    )
    checkpoint_writer.save_checkpoint(
        directory, file_name2, epoch, data, params, opt_state, key
    )
    checkpoint_writer.save_checkpoint(
        directory, file_name3, epoch, data, params, opt_state, key
    )
    checkpoint_writer.close_and_await()

    expected_calls = [
        mock.call(directory, file_name1, epoch, data, params, opt_state, key),
        mock.call(directory, file_name2, epoch, data, params, opt_state, key),
        mock.call(directory, file_name3, epoch, data, params, opt_state, key),
    ]
    assert save_vmc_stub.call_args_list == expected_calls
