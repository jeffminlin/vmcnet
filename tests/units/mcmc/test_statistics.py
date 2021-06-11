"""Test routines for computing statistics related to MCMC time series."""
import numpy.random as random
import numpy as np
import vmcnet.mcmc.statistics as statistics

nsamples = 100000
nchains = 5


def test_constant_samples_no_variance():
    """Test that samples from a constant distribution produce 0 variance."""
    constant_samples = np.ones((nsamples, nchains))

    var_estimate = statistics.multi_chain_variance_estimate(constant_samples)

    np.testing.assert_allclose(var_estimate, 0)


def test_independent_samples():
    """Test statistics on a chain with independent samples."""
    independent_samples = random.randn(nsamples, nchains)

    autocorr_curve = statistics.multi_chain_autocorr(independent_samples)
    tau = statistics.tau(autocorr_curve)

    # Autocorrelation curve should return sample variance at first index and
    # 0 everywhere else, due to independence of samples.
    np.testing.assert_allclose(autocorr_curve[0], 1, atol=1e-1)
    np.testing.assert_allclose(autocorr_curve[1:], 0, atol=1e-1)
    # Autocorrelation time should be 1 as each sample is independent.
    np.testing.assert_allclose(tau, 1, 1e-1)


def test_correlated_samples():
    """Test statistics on a chain with exponentially decaying autocorrelation."""
    decay_factor = 0.9
    independent_samples = random.randn(nsamples, nchains)
    correlated_samples = np.ones_like(independent_samples)
    # Construct simple correlated chains where each sample is constructed by multiplying
    # The previous sample by a decay factor and then adding a new independent sample.
    for chain in range(nchains):
        correlated_samples[0][chain] = independent_samples[0][chain]
        for i in range(1, len(correlated_samples)):
            correlated_samples[i][chain] = (
                correlated_samples[i - 1][chain] * decay_factor
                + independent_samples[i][chain]
            )

    autocorr_curve = statistics.multi_chain_autocorr(correlated_samples)
    tau = statistics.tau(autocorr_curve)

    # Test beginning of autocorrelation curve against a decaying exponential.
    autocorr_to_check = 100
    expected_autocorr_head = decay_factor ** np.arange(autocorr_to_check)
    np.testing.assert_allclose(
        autocorr_curve[0:autocorr_to_check], expected_autocorr_head, atol=1e-1
    ),
    # Test autocorrelation time against the infinite sum of the decaying exponential.
    infinite_exponential_sum = 1 / (1 - decay_factor)
    expected_tau = -1 + 2 * infinite_exponential_sum
    np.testing.assert_allclose(tau, expected_tau, atol=1.0)
