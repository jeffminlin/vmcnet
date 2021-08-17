"""Routines for computing statistics related to MCMC time series."""

from typing import Dict, Optional, Tuple

import numpy as np


def per_chain_autocorr_fast(
    samples: np.ndarray, cutoff: Optional[int] = None
) -> np.ndarray:
    """Calculate autocorrelation curve per chain using FFT.

    See Sokal, Alan D., "Monte Carlo Methods in Statistical Mechanics: Foundations
    and New Algorithms," pp. 13-16, September 1996.
    http://staff.ustc.edu.cn/~yjdeng/lecturenotes/cargese_1996.pdf

    Args:
        samples (np.ndarray): samples of shape (N, M) where N is num-samples-per-chain
            and M is num-chains.
        cutoff (int): hard cut-off for the length of returned curve.

    Returns:
        np.ndarray: autcorrelation curve per chain, of shape (C, M), where
            C = min(N, cutoff)
    """
    N = len(samples)
    if cutoff is None:
        cutoff = N
    else:
        cutoff = min(cutoff, N)
    centered_samples = samples - np.mean(samples, axis=0, keepdims=True)

    # Calculate autocorrelation curve efficiently as the inverse Fourier transform
    # of the power spectral density of the series.
    fvi = np.fft.fft(centered_samples, n=(2 * N), axis=0)
    G = np.real(np.fft.ifft(fvi * np.conjugate(fvi), axis=0))[:cutoff]
    # Divide (i)th term by (n-i) to account for the number of available samples.
    normalization_factors = N - np.arange(cutoff)
    G /= np.expand_dims(normalization_factors, -1)
    # Divide by C(0) to get the normalized autocorrelation curve.
    G /= G[:1]
    return G


def multi_chain_autocorr_and_variance(
    samples: np.ndarray, cutoff: Optional[int] = None
) -> Tuple[np.ndarray, np.float32]:
    """Calculate multi-chain autocorrelation curve with cutoff and multi-chain variance.

    The variance estimate here is the population variance, sum_i (x_i - mu)^2 / N,
    and *not* the sample variance, sum_i (x_i - mu)^2 / (N - 1).

    See Stan Reference Manual, Version 2.27, Section 16.3-16-4
    https://mc-stan.org/docs/2_27/reference-manual

    Args:
        samples (np.ndarray): samples of shape (N, M) where N is num-samples-per-chain
            and M is num-chains.
        cutoff (int): hard cut-off for the length of returned curve.

    Returns:
        (np.ndarray, np.float32): combined autcorrelation curve using data from all
        chains, of length C, where C = min(N, cutoff); overall variance estimate
    """
    N = len(samples)

    per_chain_autocorr = per_chain_autocorr_fast(samples, cutoff)
    per_chain_var = np.var(samples, axis=0, ddof=1)
    autocorrelation_term = np.mean(per_chain_var * per_chain_autocorr, axis=1)

    between_chain_term = np.var(np.mean(samples, axis=0), ddof=1)
    within_chain_term = np.mean(per_chain_var)
    overall_var_estimate = within_chain_term * (N - 1) / N + between_chain_term

    return (
        1 - (within_chain_term - autocorrelation_term) / overall_var_estimate,
        overall_var_estimate,
    )


def tau(autocorr_curve: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation, with automatic truncation.

    Uses Geyer's initial minimum sequence for estimating the integrated autocorrelation.
    This is a consistent overestimate of the asymptotic IAC.

    See Geyer, Charles J. 2011. “Introduction to Markov Chain Monte Carlo.” In Handbook
    of Markov Chain Monte Carlo, edited by Steve Brooks, Andrew Gelman, Galin L. Jones,
    and Xiao-Li Meng, 3–48. Chapman; Hall/CRC.

    Also see Charles J. Geyer. "Practical Markov Chain Monte Carlo." Statist. Sci.
    7 (4) 473 - 483, November, 1992. https://doi.org/10.1214/ss/1177011137

    Args:
        autocorr_curve (np.ndarray): 1D array containing the series of autocorrelation
            values over which to calculate the autocorrelation time.
    Returns:
        (np.ndarray): a single estimate of the autocorrelation time, packaged
            as an array.
    """
    # Cut off the last sample if necessary to get an even number of samples.
    nsamples = autocorr_curve.shape[0]
    even_nsamples = int(2 * np.floor(nsamples / 2))
    even_length_curve = autocorr_curve[:even_nsamples]

    # Create new curve containing sums of adjacent pairs from the initial curve.
    paired_curve = even_length_curve.reshape(
        (even_nsamples // 2, 2) + tuple(autocorr_curve.shape[1:])
    )
    sums_of_pairs = np.sum(paired_curve, axis=1)

    # The new curve is always positive in theory. In practice, the first negative term
    # can be used as a sentinel to decide when to cut off the calculation.
    negative_indices = np.nonzero(sums_of_pairs < 0)[0]
    if len(negative_indices) > 0:
        first_negative_idx = negative_indices[0]
    else:
        first_negative_idx = len(sums_of_pairs)
    positive_sums_curve = sums_of_pairs[:first_negative_idx]

    # Final estimate is based on the sum of the monotonically decreasing curve created
    # by taking the running minimum of the positive_sums_curve.
    monotonic_min_curve = np.minimum.accumulate(positive_sums_curve)
    return -1.0 + 2.0 * np.sum(monotonic_min_curve, axis=0)


def get_stats_summary(samples: np.ndarray) -> Dict[str, np.float32]:
    """Get a summary of the stats (mean, var, std_err, iac) for a collection of samples.

    Args:
        samples (np.ndarray): samples of shape (N, M) where N is num-samples-per-chain
            and M is num-chains.

    Returns:
        dictionary: a summary of the statistics, with keys "average", "variance",
        "std_err", and "integrated_autocorrelation"
    """
    # Nested mean may be more numerically stable than single mean
    average = np.mean(np.mean(samples, axis=-1), axis=-1)
    autocorr_curve, variance = multi_chain_autocorr_and_variance(samples)
    iac = tau(autocorr_curve)
    std_err = np.sqrt(iac * variance / np.size(samples))
    eval_statistics = {
        "average": average,
        "variance": variance,
        "std_err": std_err,
        "integrated_autocorrelation": iac,
    }

    return eval_statistics
