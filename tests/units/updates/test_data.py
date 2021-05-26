"""Test data update routines."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.updates as updates


def test_update_position_and_amplitude():
    """Test that the mask modification is working for the position update."""
    pos = jnp.array([0, 0, 0, 0])
    proposed_pos = jnp.array([1, 2, 3, 4])
    amplitude = jnp.array([-1, -1, -1, -1])
    proposed_amplitude = jnp.array([-1, -2, -3, -4])

    data = updates.data.PositionAmplitudeData(pos, amplitude)
    proposed_data = updates.data.PositionAmplitudeData(proposed_pos, proposed_amplitude)

    move_mask = jnp.array([True, False, False, True])

    updated_data = updates.data.update_position_and_amplitude(
        data, proposed_data, move_mask
    )

    np.testing.assert_allclose(updated_data.position, jnp.array([1, 0, 0, 4]))
    np.testing.assert_allclose(updated_data.amplitude, jnp.array([-2, -1, -1, -5]))
