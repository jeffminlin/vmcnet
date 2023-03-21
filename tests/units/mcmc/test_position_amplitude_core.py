"""Test core routines for position amplitude data."""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc.position_amplitude_core as pacore
from vmcnet.utils.typing import Array

from tests.test_utils import make_dummy_data_params_and_key, dummy_model_apply


def test_gaussian_proposal_with_nonzero_step_width():
    """Test that a model_apply is passed through correctly when making a proposal_fn."""
    std_move = 0.3
    proposal_fn = pacore.make_position_amplitude_gaussian_proposal(
        dummy_model_apply, lambda _: std_move
    )
    positions, params, key = make_dummy_data_params_and_key()
    # use the "wrong" amplitudes here so we can make sure the "right" ones come out of
    # the proposal
    amplitudes = jnp.array([-1, -1, -1, -1])
    data = pacore.make_position_amplitude_data(positions, amplitudes, None)

    new_data, _ = proposal_fn(params, data, key)
    np.testing.assert_allclose(
        new_data["walker_data"]["amplitude"], jnp.array([0, 1, 2, 3])
    )


def _get_data_for_test_update() -> (
    Tuple[
        pacore.PositionAmplitudeData,
        pacore.PositionAmplitudeData,
        Array,
        Array,
        Array,
    ]
):
    pos = jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    proposed_pos = jnp.array([[1, 1], [2, 2], [3, 4], [4, 3]])
    amplitude = jnp.array([-1, -1, -1, -1])
    proposed_amplitude = jnp.array([-1, -2, -3, -4])
    original_metadata_value = 2
    proposed_metadata_value = 3

    data = pacore.make_position_amplitude_data(pos, amplitude, original_metadata_value)
    proposed_data = pacore.make_position_amplitude_data(
        proposed_pos, proposed_amplitude, proposed_metadata_value
    )
    move_mask = jnp.array([True, False, False, True])
    expected_position = jnp.array([[1, 1], [0, 0], [0, 0], [4, 3]])
    expected_amplitude = jnp.array([-1, -1, -1, -4])

    return (data, proposed_data, move_mask, expected_position, expected_amplitude)


def test_update_position_amplitude():
    """Test mask application and metadata update."""
    (
        data,
        proposed_data,
        move_mask,
        expected_position,
        expected_amplitude,
    ) = _get_data_for_test_update()

    updated_metadata_value = 4

    update_position_amplitude = pacore.make_position_amplitude_update(
        lambda old_val, _: updated_metadata_value
    )
    updated_data = update_position_amplitude(data, proposed_data, move_mask)

    (position, amplitude, move_metadata) = pacore.to_pam_tuple(updated_data)
    np.testing.assert_allclose(position, expected_position)
    np.testing.assert_allclose(amplitude, expected_amplitude)
    np.testing.assert_allclose(move_metadata, updated_metadata_value)


def test_update_position_amplitude_no_metadata_update_fn():
    """Test that proposed metadata is returned directly when no update fn is given."""
    (
        data,
        proposed_data,
        move_mask,
        expected_position,
        expected_amplitude,
    ) = _get_data_for_test_update()

    update_position_amplitude = pacore.make_position_amplitude_update()
    updated_data = update_position_amplitude(data, proposed_data, move_mask)

    (position, amplitude, move_metadata) = pacore.to_pam_tuple(updated_data)
    np.testing.assert_allclose(position, expected_position)
    np.testing.assert_allclose(amplitude, expected_amplitude)
    np.testing.assert_allclose(move_metadata, proposed_data["move_metadata"])


def test_distribute_position_amplitude_data():
    """Test proper distribution of position, amplitude, and metadata."""
    ndevices = jax.local_device_count()
    pos = jnp.arange(ndevices)
    amplitude = -1 * jnp.arange(ndevices) - 1
    metadata = jnp.array([2, 3])

    data = pacore.make_position_amplitude_data(pos, amplitude, metadata)

    data = pacore.distribute_position_amplitude_data(data)

    for device_index in range(ndevices):
        (position, amplitude, move_metadata) = pacore.to_pam_tuple(data)
        # Position and amplitude are distributed across devices
        np.testing.assert_equal(position[device_index], jnp.array([device_index]))
        np.testing.assert_equal(amplitude[device_index], jnp.array([-device_index - 1]))
        # Metadata is replicated across devices
        np.testing.assert_array_equal(move_metadata[device_index], jnp.array([2, 3]))
