"""Test routines for computing statistics related to MCMC time series."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import vmcnet.mcmc.statistics as statistics


def test_autocorr_per_chain():
    """Test autocorrelation on alternating chains of +-1 and +-2"""
    samples = jnp.tile(
        jnp.array(
            [
                [1, 2],
                [-1, -2],
            ]
        ),
        (10, 1),
    )

    autocorr = statistics.per_chain_autocorr_fast(samples)

    # Autocorrelation should be +1 for even distances and -1 for odd distances
    expected_autocorr = jnp.tile(jnp.array([[1.0, 1.0], [-1.0, -1.0]]), (10, 1))
    np.testing.assert_allclose(autocorr, expected_autocorr, 1e-5)
