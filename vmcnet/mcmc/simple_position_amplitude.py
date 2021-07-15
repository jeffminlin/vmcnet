"""Helper functions for position amplitude data with fixed-width gaussian steps."""

from typing import Callable, Tuple, TypedDict

import jax.numpy as jnp

from .position_amplitude_core import (
    make_position_amplitude_data,
    make_position_amplitude_gaussian_metropolis_step,
    PositionAmplitudeWalkerData,
)
from vmcnet.utils.typing import P, ModelApply


class SimplePositionAmplitudeData(TypedDict):
    """TypedDict of positions, amplitudes, and nothing else."""

    walker_data: PositionAmplitudeWalkerData
    move_metadata: None


SPAData = SimplePositionAmplitudeData


def make_simple_position_amplitude_data(position: jnp.ndarray, amplitude: jnp.ndarray):
    """Create SimplePositionAmplitudeData from position and amplitude.

    Args:
        position (jnp.ndarray): the particle positions
        amplitude (jnp.ndarray): the wavefunction amplitudes

    Returns:
        SPAData
    """
    return make_position_amplitude_data(position, amplitude, None)


def make_simple_pos_amp_gaussian_step(
    model_apply: ModelApply[P],
    std_move: jnp.float32,
    logabs: bool = True,
) -> Callable[[P, SPAData, jnp.ndarray], Tuple[jnp.float32, SPAData, jnp.ndarray],]:
    """Create metropolis step for PositionAmplitudeData with fixed gaussian step width.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        std_move: the standard deviation of the gaussian step
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        Callable: function which does a metropolis step. Has the signature
            (params, PositionAmplitudeData, key)
            -> (mean acceptance probability, PositionAmplitudeData, new_key)
    """
    return make_position_amplitude_gaussian_metropolis_step(
        model_apply, lambda _: std_move, None, logabs
    )
