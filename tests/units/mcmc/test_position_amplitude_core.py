"""Test core routines for position amplitude data."""
import jax.numpy as jnp
import numpy as np
import vmcnet.mcmc as mcmc

from .test_metropolis import _make_dummy_data_params_and_key, _dummy_model_apply


def test_gaussian_proposal_with_nonzero_step_width():
    """Test that a model_apply is passed through correctly when making a proposal_fn."""
    std_move = 0.3
    proposal_fn = (
        mcmc.position_amplitude_core.make_position_amplitude_gaussian_proposal(
            _dummy_model_apply, lambda _: std_move
        )
    )
    positions, params, key = _make_dummy_data_params_and_key()
    # use the "wrong" amplitudes here so we can make sure the "right" ones come out of
    # the proposal
    amplitudes = jnp.array([-1, -1, -1, -1])
    data = mcmc.position_amplitude_core.PositionAmplitudeData(
        positions, amplitudes, None
    )

    new_data, _ = proposal_fn(params, data, key)
    np.testing.assert_allclose(new_data.amplitude, jnp.array([0, 1, 2, 3]))


def test_update_gaussian_position_amplitude():
    """Test that the mask modification is working for the position update."""
    pos = jnp.array([0, 0, 0, 0])
    proposed_pos = jnp.array([1, 2, 3, 4])
    amplitude = jnp.array([-1, -1, -1, -1])
    proposed_amplitude = jnp.array([-1, -2, -3, -4])
    std_move = 0.3
    mean_accept_prob = 0.5
    old_metadata_value = 3
    new_metadata_value = 4

    data = mcmc.position_amplitude_core.PositionAmplitudeData(pos, amplitude, None)
    proposed_data = mcmc.position_amplitude_core.PositionAmplitudeData(
        proposed_pos, proposed_amplitude, old_metadata_value
    )

    move_mask = jnp.array([True, False, False, True])

    update_position_amplitude = (
        mcmc.position_amplitude_core.make_position_amplitude_update(
            lambda old_val, _2: new_metadata_value
        )
    )
    updated_data = update_position_amplitude(data, proposed_data, move_mask)

    np.testing.assert_allclose(updated_data.position, jnp.array([1, 0, 0, 4]))
    np.testing.assert_allclose(updated_data.amplitude, jnp.array([-1, -1, -1, -4]))
    np.testing.assert_allclose(updated_data.move_metadata, new_metadata_value)


def test_distribute_position_amplitude_data():
    """Test proper distribution of position, amplitude, and metadata"""
    pos = jnp.array([0, 1, 2, 3])
    amplitude = jnp.array([-1, -2, -3, -4])
    metadata = jnp.array([2, 3])

    data = mcmc.position_amplitude_core.PositionAmplitudeData(pos, amplitude, metadata)

    data = mcmc.position_amplitude_core.distribute_position_amplitude_data(data)

    for device_index in range(4):
        # Position and amplitude are distributed across devices
        np.testing.assert_equal(data.position[device_index], jnp.array([device_index]))
        np.testing.assert_equal(
            data.amplitude[device_index], jnp.array([-device_index - 1])
        )
        # Metadata is replicated across devices
        np.testing.assert_array_equal(
            data.move_metadata[device_index], jnp.array([2, 3])
        )
