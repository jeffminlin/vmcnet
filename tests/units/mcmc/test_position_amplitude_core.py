"""Test core routines for position amplitude data."""
import jax.numpy as jnp
import numpy as np
import vmcnet.mcmc as mcmc

from ..utils import make_dummy_data_params_and_key, dummy_model_apply


def test_gaussian_proposal_with_nonzero_step_width():
    """Test that a model_apply is passed through correctly when making a proposal_fn."""
    std_move = 0.3
    proposal_fn = (
        mcmc.position_amplitude_core.make_position_amplitude_gaussian_proposal(
            dummy_model_apply, lambda _: std_move
        )
    )
    positions, params, key = make_dummy_data_params_and_key()
    # use the "wrong" amplitudes here so we can make sure the "right" ones come out of
    # the proposal
    amplitudes = jnp.array([-1, -1, -1, -1])
    data = mcmc.position_amplitude_core.make_position_amplitude_data(
        positions, amplitudes, None
    )

    new_data, _ = proposal_fn(params, data, key)
    np.testing.assert_allclose(new_data.walker_data.amplitude, jnp.array([0, 1, 2, 3]))


def test_update_position_amplitude():
    """Test that the mask modification is working for the position update."""
    pos = jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    proposed_pos = jnp.array([[1, 1], [2, 2], [3, 4], [4, 3]])
    amplitude = jnp.array([-1, -1, -1, -1])
    proposed_amplitude = jnp.array([-1, -2, -3, -4])
    old_metadata_value = 3
    new_metadata_value = 4

    data = mcmc.position_amplitude_core.make_position_amplitude_data(
        pos, amplitude, None
    )
    proposed_data = mcmc.position_amplitude_core.make_position_amplitude_data(
        proposed_pos, proposed_amplitude, old_metadata_value
    )

    move_mask = jnp.array([True, False, False, True])

    update_position_amplitude = (
        mcmc.position_amplitude_core.make_position_amplitude_update(
            lambda old_val, _2: new_metadata_value
        )
    )
    updated_data = update_position_amplitude(data, proposed_data, move_mask)

    ((position, amplitude), move_metadata) = updated_data
    np.testing.assert_allclose(position, jnp.array([[1, 1], [0, 0], [0, 0], [4, 3]]))
    np.testing.assert_allclose(amplitude, jnp.array([-1, -1, -1, -4]))
    np.testing.assert_allclose(move_metadata, new_metadata_value)


def test_distribute_position_amplitude_data():
    """Test proper distribution of position, amplitude, and metadata"""
    pos = jnp.array([0, 1, 2, 3])
    amplitude = jnp.array([-1, -2, -3, -4])
    metadata = jnp.array([2, 3])

    data = mcmc.position_amplitude_core.make_position_amplitude_data(
        pos, amplitude, metadata
    )

    data = mcmc.position_amplitude_core.distribute_position_amplitude_data(data)

    for device_index in range(4):
        ((position, amplitude), move_metadata) = data
        # Position and amplitude are distributed across devices
        np.testing.assert_equal(position[device_index], jnp.array([device_index]))
        np.testing.assert_equal(amplitude[device_index], jnp.array([-device_index - 1]))
        # Metadata is replicated across devices
        np.testing.assert_array_equal(move_metadata[device_index], jnp.array([2, 3]))
