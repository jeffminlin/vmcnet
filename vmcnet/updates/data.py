"""Routines for updating data given some proposed data and accepted walker moves."""
from typing import NamedTuple
import jax.numpy as jnp


class PositionAmplitudeData(NamedTuple):
    """NamedTuple of data holding positions and wavefunction amplitudes."""

    position: jnp.ndarray
    amplitude: jnp.ndarray


def update_position_and_amplitude(
    data: PositionAmplitudeData,
    proposed_data: PositionAmplitudeData,
    move_mask: jnp.ndarray,
) -> PositionAmplitudeData:
    """Update a data dict which holds both position and amplitude info.

    Holding both particle position and wavefn amplitude in the data can be advantageous
    to avoid recalculating amplitudes in some routines, e.g. acceptance probabilities.
    This function takes a mask of approved MCMC walker moves `move_mask` and accepts
    those proposed moves from `proposed_data`, for both positions and amplitudes.

    The moves in `move_mask` are applied along the first axis of the position data, and
    should be the same shape as the amplitude data (one-dimensional jnp.ndarray).

    Args:
        data (PositionAmplitudeData): original data, before the move
        proposed_data (PositionAmplitudeData): proposed data, some of which will be
            moved to, wherever `move_mask` is True.
        move_mask (jnp.ndarray[bool]): a boolean array of the same shape as
            data.amplitude which is True where the proposed moves should be accepted.

    Returns:
        PositionAmplitudeData: mix of original and proposed data for both position and
        amplitude, equal to the original data where `move_mask` is False, and equal to
        the proposed data where `move_mask` is True.
    """
    pos_mask = jnp.reshape(move_mask, (-1,) + (len(data.position.shape) - 1) * (1,))
    new_position = jnp.where(pos_mask, proposed_data.position, data.position)
    new_amplitude = jnp.where(move_mask, proposed_data.amplitude, data.amplitude)
    return PositionAmplitudeData(new_position, new_amplitude)
