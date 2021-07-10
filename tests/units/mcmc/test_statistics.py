"""Test routines for computing statistics related to MCMC time series."""

import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.mcmc.statistics as statistics


@pytest.mark.slow
def test_alternating_autocorr_per_chain():
    """Test per chain autocorrelation on alternating chains of +-n."""
    ntiles = 100
    samples = jnp.tile(
        jnp.array(
            [
                [1, 2],
                [-1, -2],
            ]
        ),
        (ntiles, 1),
    )

    autocorr = statistics.per_chain_autocorr_fast(samples)

    # Autocorrelation should be +1 for even distances and -1 for odd distances
    expected_autocorr = jnp.tile(jnp.array([[1.0, 1.0], [-1.0, -1.0]]), (ntiles, 1))
    np.testing.assert_allclose(autocorr, expected_autocorr, 1e-5)


@pytest.mark.slow
def test_alternating_multi_chain_autocorr():
    """Test multi chain autocorrelation on alternating chains of +-n."""
    nsamples = 20
    ntiles = nsamples // 2
    samples = jnp.tile(
        jnp.array(
            [
                [1, 2, 3],
                [-1, -2, -3],
            ]
        ),
        (ntiles, 1),
    )

    autocorr = statistics.multi_chain_autocorr(samples)

    # For even distances, expect perfect correlation of 1.0. For odd distances, a quirk
    # in the estimation formulas makes the value come out to not quite -1.0.
    expected_even = 1.0
    expected_odd = -1.0 - 2 / (nsamples - 1)
    expected_autocorr = jnp.tile(jnp.array([expected_even, expected_odd]), ntiles)
    np.testing.assert_allclose(autocorr, expected_autocorr, 1e-5)


def test_tau_exp_decay():
    """Test tau calculation on decaying exponential autocorrelation curve."""
    decay_factor = 0.9
    autocorr = decay_factor ** jnp.arange(1000)
    tau = statistics.tau(autocorr)

    infinite_exponential_sum = 1 / (1 - decay_factor)
    expected_tau = -1 + 2 * infinite_exponential_sum
    np.testing.assert_allclose(tau, expected_tau, 1e-6)
