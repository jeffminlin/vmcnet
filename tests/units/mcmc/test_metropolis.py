"""Test Metropolis routines."""
from jax._src.lax.lax import exp
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc as mcmc


def test_symmetric_acceptance():
    old_amplitudes = jnp.array([1.0, -jnp.pi, 0.0, 1e-5])
    new_amplitudes = jnp.array([0.0, 1.0, 3.0, 4.0])

    log_old_amplitudes = jnp.log(jnp.abs(old_amplitudes))
    log_new_amplitudes = jnp.log(jnp.abs(new_amplitudes))

    expected_accept_prob = jnp.array([0.0, 1.0 / jnp.pi ** 2, 1.0, 1.0])

    acceptance_prob = mcmc.metropolis.metropolis_symmetric_acceptance(
        old_amplitudes, new_amplitudes, logabs=False
    )
    acceptance_prob_from_log = mcmc.metropolis.metropolis_symmetric_acceptance(
        log_old_amplitudes, log_new_amplitudes, logabs=True
    )

    np.testing.assert_allclose(acceptance_prob, expected_accept_prob)
    np.testing.assert_allclose(acceptance_prob_from_log, expected_accept_prob)
