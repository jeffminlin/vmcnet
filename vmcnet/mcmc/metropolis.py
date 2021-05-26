"""Proposal and acceptance fns for Metropolis-Hastings Markov-Chain Monte Carlo."""
from typing import Callable, Tuple, TypeVar

import jax
import jax.numpy as jnp

from vmcnet.updates.data import PositionAmplitudeData

P = TypeVar("P")  # to represent a pytree or pytree-like object containing model params


def gaussian_proposal(
    positions: jnp.ndarray, std_move: jnp.float32, key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simple symmetric gaussian proposal in all positions at once.

    Args:
        positions (jnp.ndarray): original positions
        std_move (jnp.float32): standard deviation of the moves
        key (jnp.ndarray): an array with shape (2,) representing a jax PRNG key

    Returns:
        (jnp.ndarray, jnp.ndarray): (new positions, new key split from previous)
    """
    key, subkey = jax.random.split(key)
    return positions + std_move * jax.random.normal(subkey, shape=positions.shape), key


def make_position_and_amplitude_gaussian_proposal(
    model_eval: Callable[[P, jnp.ndarray], jnp.ndarray], std_move: jnp.float32
) -> Callable[
    [P, PositionAmplitudeData, jnp.ndarray], Tuple[PositionAmplitudeData, jnp.ndarray]
]:
    """Factory to make a gaussian proposal on PositionAmplitudeData.

    Args:
        model_eval (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        std_move (jnp.float32): standard deviation of the proposed moves

    Returns:
        Callable: proposal function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, key) -> (PositionAmplitudeData, key).
    """

    def proposal_fn(params, data, key):
        proposed_position, key = gaussian_proposal(data.position, std_move, key)
        proposed_amplitude = model_eval(params, proposed_position)
        return PositionAmplitudeData(proposed_position, proposed_amplitude), key

    return proposal_fn


def metropolis_symmetric_acceptance(
    amplitude: jnp.ndarray, proposed_amplitude: jnp.ndarray, logabs: bool = True
) -> jnp.ndarray:
    """Standard Metropolis acceptance ratio for a symmetric proposal function.

    The general Metropolis-Hastings choice of acceptance ratio for moves from state i to
    state j is given by

        accept_ij = min(1, (P_j * proposal_prob_ji) / (P_i * proposal_prob_ij)).

    When proposal_prob is symmetric (assumed in this function), this simply reduces to
    accept_ij = min(1, P_j / P_i). Some care is taken to avoid numerical overflow and
    division by zero.

    The inputs are wavefunction amplitudes psi or log(|psi|), so the probability P_i
    refers to |psi(i)|^2.

    Args:
        amplitude (jnp.ndarray): one-dimensional array of wavefunction amplitudes for
            the current state, or log wavefunction amplitudes if logabs is True
        proposed_amplitude (jnp.ndarray): one-dimensional array of wavefunction
            amplitudes for the proposed state, or log wavefunction amplitudes if logabs
            is True
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        jnp.ndarray: one-dimensional array of acceptance ratios for the Metropolis
        algorithm
    """
    if not logabs:
        prob_old = jnp.square(amplitude)
        prob_new = jnp.square(proposed_amplitude)
        ratio = prob_new / prob_old
        # safe division by zero
        ratio = jnp.where(
            jnp.logical_or(prob_old < prob_new, prob_old == 0.0),
            jnp.ones_like(ratio),
            ratio,
        )
        return ratio

    log_prob_old = 2.0 * amplitude
    log_prob_new = 2.0 * proposed_amplitude
    # avoid overflow if log_prob_new - log_prob_old is large
    return jnp.where(
        log_prob_new > log_prob_old,
        jnp.ones_like(log_prob_new),
        jnp.exp(log_prob_new - log_prob_old),
    )


def make_position_and_amplitude_metrpolis_symmetric_acceptance(
    logabs: bool = True,
) -> Callable[[P, PositionAmplitudeData, PositionAmplitudeData], jnp.ndarray]:
    """Factory to make a Metropolis acceptance function on PositionAmplitudeData.

    Args:
        logabs (bool, optional): whether amplitudes provided to `acceptance_fn`
            represent psi (logabs = False) or log|psi| (logabs = True). Defaults to
            True.

    Returns:
        Callable: acceptance function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, PositionAmplitudeData) -> accept_ratio
    """

    def acceptance_fn(params, data, proposed_data):
        del params
        return metropolis_symmetric_acceptance(
            data.amplitude, proposed_data.amplitude, logabs=logabs
        )

    return acceptance_fn
