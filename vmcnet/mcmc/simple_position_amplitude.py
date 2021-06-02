"""Helper functions for position amplitude data with fixed-width gaussian
steps and no metadata."""
from typing import Callable, TypeVar

import jax.numpy as jnp

from .position_amplitude_core import (
    make_position_amplitude_gaussian_metropolis_step,
    PositionAmplitudeData,
)

# Represents a pytree or pytree-like object containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
# Represents a pytree or pytree-like object containing model params
P = TypeVar("P")


class SimplePositionAmplitudeData(PositionAmplitudeData):
    """NamedTuple of data holding positions and wavefunction amplitudes, and nothing else."""

    move_metadata: None


def make_simple_position_amplitude_data(position: jnp.ndarray, amplitude: jnp.ndarray):
    """Creates SimplePositionAmplitudeData by filling in the move_metadata field with a None value

    Args:
        position (jnp.ndarray): the particle positions
        amplitude (jnp.ndarray): the wavefunction amplitudes

    Returns:
        SimplePositionAmplitudeData
    """
    return SimplePositionAmplitudeData(position, amplitude, None)


def make_simple_pos_amp_gaussian_step(
    model_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    std_move: jnp.float32,
    logabs: bool = True,
):
    """Creates a simple metropolis step for PositionAmplitudeData with fixed gaussian step width

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
