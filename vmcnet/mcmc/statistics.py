"""Routines for computing statistics related to MCMC time series."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def per_chain_autocorr_fast(
    samples: jnp.ndarray, cutoff: Optional[int] = None
) -> jnp.ndarray:
    """Calculate autocorrelation curve per chain using FFT.

    See Sokal, Alan D., "Monte Carlo Methods in Statistical Mechanics: Foundations
    and New Algorithms," pp. 13-16, September 1996.
    http://staff.ustc.edu.cn/~yjdeng/lecturenotes/cargese_1996.pdf

    Args:
        samples (jnp.ndarray): samples of shape (N, M) where N is num-samples-per-chain
            and M is num-chains.
        cutoff (int): hard cut-off for the length of returned curve.

    Returns:
        jnp.ndarray: autcorrelation curve per chain, of shape (C, M), where
            C = min(N, cutoff)
    """
    N = len(samples)
    if cutoff is None:
        cutoff = N
    else:
        cutoff = min(cutoff, N)
    centered_samples = samples - jnp.mean(samples, axis=0, keepdims=True)

    # Calculate autocorrelation curve efficiently as the inverse Fourier transform
    # of the power spectral density of the series.
    fvi = jnp.fft.fft(centered_samples, n=(2 * N), axis=0)
    G = jnp.real(jnp.fft.ifft(fvi * jnp.conjugate(fvi), axis=0))[:cutoff]
    # Divide (i)th term by (n-i) to account for the number of available samples.
    normalization_factors = N - jnp.arange(cutoff)
    G /= jnp.expand_dims(normalization_factors, -1)
    # Divide by C(0) to get the normalized autocorrelation curve.
    G /= G[:1]
    return G


def multi_chain_autocorr_and_variance(
    samples: jnp.ndarray, cutoff: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.float32]:
    """Calculate multi-chain autocorrelation curve with cutoff and multi-chain variance.

    The variance estimate here is the population variance, sum_i (x_i - mu)^2 / N,
    and *not* the sample variance, sum_i (x_i - mu)^2 / (N - 1).

    See Stan Reference Manual, Version 2.27, Section 16.3-16-4
    https://mc-stan.org/docs/2_27/reference-manual

    Args:
        samples (jnp.ndarray): samples of shape (N, M) where N is num-samples-per-chain
            and M is num-chains.
        cutoff (int): hard cut-off for the length of returned curve.

    Returns:
        (jnp.ndarray, jnp.float32): combined autcorrelation curve using data from all
        chains, of length C, where C = min(N, cutoff); overall variance estimate
    """
    N = len(samples)

    per_chain_autocorr = per_chain_autocorr_fast(samples, cutoff)
    per_chain_var = jnp.var(samples, axis=0, ddof=1)
    autocorrelation_term = jnp.mean(per_chain_var * per_chain_autocorr, axis=1)

    between_chain_term = jnp.var(jnp.mean(samples, axis=0), ddof=1)
    within_chain_term = jnp.mean(per_chain_var)
    overall_var_estimate = within_chain_term * (N - 1) / N + between_chain_term

    return (
        1 - (within_chain_term - autocorrelation_term) / overall_var_estimate,
        overall_var_estimate,
    )


def tau(autocorr_curve: jnp.ndarray) -> jnp.ndarray:
    """Integrated autocorrelation, with automatic truncation.

    Uses Geyer's initial minimum sequence for estimating the integrated autocorrelation.
    This is a consistent overestimate of the asymptotic IAC.

    See Geyer, Charles J. 2011. “Introduction to Markov Chain Monte Carlo.” In Handbook
    of Markov Chain Monte Carlo, edited by Steve Brooks, Andrew Gelman, Galin L. Jones,
    and Xiao-Li Meng, 3–48. Chapman; Hall/CRC.

    Also see Charles J. Geyer. "Practical Markov Chain Monte Carlo." Statist. Sci.
    7 (4) 473 - 483, November, 1992. https://doi.org/10.1214/ss/1177011137

    Args:
        autocorr_curve (jnp.ndarray): 1D array containing the series of autocorrelation
            values over which to calculate the autocorrelation time.
    Returns:
        (jnp.ndarray): a single estimate of the autocorrelation time, packaged
            as an array.
    """
    # Cut off the last sample if necessary to get an even number of samples.
    nsamples = autocorr_curve.shape[0]
    even_nsamples = int(2 * jnp.floor(nsamples / 2))
    even_length_curve = autocorr_curve[:even_nsamples]

    # Create new curve containing sums of adjacent pairs from the initial curve.
    paired_curve = even_length_curve.reshape(
        (even_nsamples // 2, 2) + tuple(autocorr_curve.shape[1:])
    )
    sums_of_pairs = jnp.sum(paired_curve, axis=1)

    # The new curve is always positive in theory. In practice, the first negative term
    # can be used as a sentinel to decide when to cut off the calculation.
    negative_indices = jnp.nonzero(sums_of_pairs < 0)[0]
    if len(negative_indices) > 0:
        first_negative_idx = negative_indices[0]
    else:
        first_negative_idx = len(sums_of_pairs)
    positive_sums_curve = sums_of_pairs[:first_negative_idx]

    def accumulate_minima(carry_min, new_slice):
        new_minima = jnp.minimum(carry_min, new_slice)
        return new_minima, new_minima

    # Final estimate is based on the sum of the monotonically decreasing curve created
    # by taking the running minimum of the positive_sums_curve.
    _, monotonic_min_curve = jax.lax.scan(
        accumulate_minima,
        positive_sums_curve[0],
        positive_sums_curve,
    )
    return -1.0 + 2.0 * jnp.sum(monotonic_min_curve, axis=0)
