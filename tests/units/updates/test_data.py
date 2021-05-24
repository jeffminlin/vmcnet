"""Test data update routines."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.updates as updates


def test_update_position_and_amplitude():
    """Test that the mask modification is working for the position update."""
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)

    nbatch = 10
    nparticles = 7
    ndim = 3

    pos = jax.random.normal(keys[0], shape=(nbatch, nparticles, ndim))
    proposed_pos = jax.random.normal(keys[1], shape=(nbatch, nparticles, ndim))
    amplitude = jax.random.normal(keys[2], shape=(nbatch,))
    proposed_amplitude = jax.random.normal(keys[3], shape=(nbatch,))

    data = updates.data.PositionAmplitudeData(pos, amplitude)
    proposed_data = updates.data.PositionAmplitudeData(proposed_pos, proposed_amplitude)
    move_mask = jnp.array(
        [True, False, True, True, False, True, False, False, False, False]
    )
    inverted_mask = jnp.invert(move_mask)
    updated_data = updates.data.update_position_and_amplitude(
        data, proposed_data, move_mask
    )

    for field in data._fields:
        # updated data should be equal to the proposed data where move_mask is True
        np.testing.assert_allclose(
            getattr(updated_data, field)[move_mask, ...],
            getattr(proposed_data, field)[move_mask, ...],
        )
        # updated data should be equal to the original data where move_mask is False
        np.testing.assert_allclose(
            getattr(updated_data, field)[inverted_mask, ...],
            getattr(data, field)[inverted_mask, ...],
        )
