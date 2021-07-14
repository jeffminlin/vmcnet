"""Test Metropolis routines."""
import jax.numpy as jnp
import numpy as np
import vmcnet.mcmc as mcmc

from tests.test_utils import (
    make_dummy_data_params_and_key,
    make_dummy_metropolis_fn,
)


def test_symmetric_acceptance():
    """Test acceptance probabilities in the symmetric proposal case."""
    old_amplitudes = jnp.array([1.0, -jnp.pi, 0.0, 1e-5])
    new_amplitudes = jnp.array([0.0, 1.0, 3.0, 4.0])

    log_old_amplitudes = jnp.log(jnp.abs(old_amplitudes))
    log_new_amplitudes = jnp.log(jnp.abs(new_amplitudes))

    expected_accept_prob = jnp.array([0.0, 1.0 / jnp.square(jnp.pi), 1.0, 1.0])

    acceptance_prob = mcmc.metropolis.metropolis_symmetric_acceptance(
        old_amplitudes, new_amplitudes, logabs=False
    )
    acceptance_prob_from_log = mcmc.metropolis.metropolis_symmetric_acceptance(
        log_old_amplitudes, log_new_amplitudes, logabs=True
    )

    np.testing.assert_allclose(acceptance_prob, expected_accept_prob)
    np.testing.assert_allclose(
        acceptance_prob_from_log, expected_accept_prob, rtol=1e-6
    )


def test_metropolis_step():
    """Test the acceptance probability and data update for a single Metropolis step."""
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()

    accept_prob, new_data, _ = metrop_step_fn(params, data, key)

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, jnp.array([1, 0, 3, 0]))


def test_walk_data():
    """Test a few Metropolis steps.

    Test that taking a few Metropolis steps is equivalent to skipping to the end and
    taking one big step. Specifically, with a single proposal fn which adds a constant
    array at each step, test that taking a few steps is equivalent to adding that
    multiple of the proposal array directly (where the moves are accepted).
    """
    nsteps = 6
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()
    accept_prob, new_data, _ = mcmc.metropolis.walk_data(
        nsteps, params, data, key, metrop_step_fn
    )

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, jnp.array([nsteps, 0, 3 * nsteps, 0]))
