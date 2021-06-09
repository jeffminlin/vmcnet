"""Testing io routines."""
import jax
import jax.numpy as jnp
import numpy as np
import vmcnet.utils.distribute as distribute
import vmcnet.utils.io as io

from ..test_utils import make_dummy_data_params_and_key, assert_pytree_equal


def test_save_and_reload_vmc_state(tmp_path):
    """Test round-trip of vmc state to and from disk."""
    # Set up log directory within pytest tmp_dir
    subdir = "logs"
    directory = tmp_path / subdir
    file_name = "checkpoint_file.npz"

    # Create dummy vmc state and distribute it across fake devices
    epoch = 0
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(jax.local_device_count() * 2)}
    opt_state = {"momentum": 2.0}
    (data, params, opt_state, key) = distribute.distribute_vmc_state(
        data, params, opt_state, key
    )

    # Save the vmc state to file, then reload it and redistribute it
    io.save_vmc_state(directory, file_name, epoch, data, params, opt_state, key)
    (
        restored_epoch,
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = io.reload_vmc_state(directory, file_name)
    (
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = distribute.distribute_vmc_state_from_checkpoint(
        restored_data, restored_params, restored_opt_state, restored_key
    )

    # Verify that restored data is same as original data
    np.testing.assert_equal(restored_epoch, epoch)
    assert_pytree_equal(restored_data, data)
    assert_pytree_equal(restored_params, params)
    assert_pytree_equal(restored_opt_state, opt_state)
    np.testing.assert_allclose(restored_key, key)
