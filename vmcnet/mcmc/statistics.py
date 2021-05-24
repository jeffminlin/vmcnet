"""Compute MCMC statistics to evaluate the quality of the Markov-Chain equilibration."""
import jax.numpy as jnp


def per_chain_autocorr_fast(samples, kappa):
    """Autocorrelation curve per chain with cutoff kappa.

    samples is (N, M) where N is num-samples-per-chain and M is num-chains.
    kappa is the cutoff, which is one greater than the number of lags to compute. The
        actual cutoff returned is min(kappa, N)
    """
    centered_samples = samples - jnp.mean(samples, axis=0, keepdims=True)
    N = len(samples)
    cutoff = min(kappa, N)

    fvi = jnp.fft.fft(centered_samples, n=(2 * N), axis=0)
    # G is the autocorrelation curve
    G = jnp.real(jnp.fft.ifft(fvi * jnp.conjugate(fvi), axis=0))[:cutoff]
    G = jnp.swapaxes(jnp.swapaxes(G, 0, -1) / (N - jnp.arange(cutoff)), 0, -1)
    G /= G[:1]
    return G


def multi_chain_autocorr(samples, kappa):
    """Multi-chain autocorrelation curve with cutoff kappa.

    samples is (N, M) where N is num-samples-per-chain and M is num-chains.
    kappa is the cutoff, which is one greater than the number of lags to compute. The
        actual cutoff returned is min(kappa, N)
    """
    per_chain_autocorr = per_chain_autocorr_fast(samples, kappa)
    within_chain_var = jnp.var(samples, axis=0, ddof=1)
    multi_chain_var = multi_chain_variance(samples)
    return (
        1
        - (
            jnp.mean(within_chain_var)
            - jnp.mean(within_chain_var * per_chain_autocorr, axis=1)
        )
        / multi_chain_var
    )


def multi_chain_variance(samples):
    """Compute a multi-chain variance estimate from M chains.

    Input is (N, M) where N is num-samples-per-chain and M is num-chains.
    """
    within_chain_avgs = jnp.mean(samples, axis=0)
    cross_chain_variance = jnp.var(within_chain_avgs, ddof=1)
    return jnp.var(samples, ddof=0) + cross_chain_variance


def tau(autocorr_curve):
    """Integrated autocorrelation, with automatic truncation.

    autcorr_curve has shape (cutoff, ...), where cutoff is one greater than the number
        of lags that has been computed.

    Uses Geyer's initial minimum sequence for estimating the integrated autocorrelation.
    This is a consistent overestimate of the asymptotic IAC.

    See Geyer, Charles J. 2011. “Introduction to Markov Chain Monte Carlo.” In Handbook
    of Markov Chain Monte Carlo, edited by Steve Brooks, Andrew Gelman, Galin L. Jones,
    and Xiao-Li Meng, 3–48. Chapman; Hall/CRC.
    """
    cutoff = autocorr_curve.shape[0]
    even_cutoff = int(2 * jnp.floor(cutoff / 2))
    paired_curves = autocorr_curve[:even_cutoff].reshape(
        (even_cutoff // 2, 2) + tuple(autocorr_curve.shape[1:])
    )

    # theoretically positive curve, but needs to be truncated when terms are not > 0
    positive_curve = jnp.sum(paired_curves, axis=1)
    negative_indices = jnp.nonzero(positive_curve < 0)[0]
    if len(negative_indices) > 0:
        first_negative_idx = negative_indices[0]
    else:
        first_negative_idx = len(positive_curve)
    truncated_positive_curve = positive_curve[:first_negative_idx]
    minimum_curve = jnp.minimum.accumulate(truncated_positive_curve, axis=0)

    return -1.0 + 2.0 * jnp.sum(minimum_curve, axis=0)
