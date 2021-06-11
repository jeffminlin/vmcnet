"""Test routines for computing statistics related to MCMC time series."""
import numpy.random as random
import numpy as np
import vmcnet.mcmc.statistics as statistics


def _get_sample_size():
    return (100000, 5)


def test_constant_samples_no_variance():
    """Test that samples from a constant distribution produce 0 variance."""
    (nsamples, nchains) = _get_sample_size()
    constant_samples = np.ones((nsamples, nchains))

    var_estimate = statistics.multi_chain_variance_estimate(constant_samples)

    np.testing.assert_allclose(var_estimate, 0)


def test_independent_samples():
    """Test statistics on a chain with independent samples."""
    (nsamples, nchains) = _get_sample_size()
    independent_samples = random.randn(nsamples, nchains)

    autocorr_curve = statistics.multi_chain_autocorr(independent_samples)
    tau = statistics.tau(autocorr_curve)

    # Autocorrelation curve should return sample variance at first index and
    # 0 everywhere else, due to independence of samples.
    np.testing.assert_allclose(autocorr_curve[0], 1, atol=1e-1)
    np.testing.assert_allclose(autocorr_curve[1:], 0, atol=1e-1)
    # Autocorrelation time should be 1 as each sample is independent.
    np.testing.assert_allclose(tau, 1, 1e-1)


def _construct_correlated_samples(nsamples, nchains, decay_factor):
    """Construct chains of simple correlated samples.

    Each sample is constructed by multiplying the previous sample by a constant decay
    factor and then adding a new independent gaussian sample.
    """
    independent_samples = random.randn(nsamples, nchains)
    correlated_samples = independent_samples

    def get_correlated_sample(prev_corr_sample, curr_ind_sample):
        return prev_corr_sample * decay_factor + curr_ind_sample

    for chain in range(nchains):
        for i in range(1, len(correlated_samples)):
            correlated_samples[i][chain] = get_correlated_sample(
                correlated_samples[i - 1][chain], independent_samples[i][chain]
            )

    return correlated_samples


def test_correlated_samples():
    """Test statistics on sample chains with exponentially decaying autocorrelation."""
    (nsamples, nchains) = _get_sample_size()
    decay_factor = 0.9
    correlated_samples = _construct_correlated_samples(nsamples, nchains, decay_factor)

    autocorr_curve = statistics.multi_chain_autocorr(correlated_samples)
    tau = statistics.tau(autocorr_curve)

    # Test beginning of autocorrelation curve against a decaying exponential.
    nautocorr_to_check = 100
    expected_autocorr = decay_factor ** np.arange(nautocorr_to_check)
    np.testing.assert_allclose(
        autocorr_curve[0:nautocorr_to_check], expected_autocorr, atol=1e-1
    ),
    # Test autocorrelation time against the infinite sum of the decaying exponential.
    infinite_exponential_sum = 1 / (1 - decay_factor)
    expected_tau = -1 + 2 * infinite_exponential_sum
    np.testing.assert_allclose(tau, expected_tau, atol=1.0)
