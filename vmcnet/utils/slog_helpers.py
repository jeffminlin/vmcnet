"""Helper functions for dealing with (sign, logabs) data."""
from typing import Optional

import jax.numpy as jnp

from .typing import SLArray, ArrayList, SLArrayList
from .kfac import register_batch_dense


def array_to_slog(x: jnp.ndarray) -> SLArray:
    """Converts a regular array into (sign, logabs) form.

    Args:
        x (jnp.ndarray): input data.

    Returns:
        (SLArray): data in form (sign(x), log(abs(x)))
    """
    return (jnp.sign(x), jnp.log(jnp.abs(x)))


def array_from_slog(x: SLArray) -> jnp.ndarray:
    """Converts an slog data tuple into a regular array.

    Args:
        x (SLArray): input data in slog form. This data looks like
        (sign(z), log(abs(z))) for some z which represents the underlying data.

    Returns:
        (jnp.ndarray): the data as a single, regular array. In other words, the z
        such that x = (sign(z), log(abs(z)))
    """
    return x[0] * jnp.exp(x[1])


def array_list_to_slog(x: ArrayList) -> SLArrayList:
    """Map an ArrayList to SLArrayList form.

    Args:
        x (ArrayList): input data as a regular spin-split array.

    Returns:
        (SLArrayList): same data with each array transformed to slog form.
    """
    return [array_to_slog(arr) for arr in x]


def array_list_from_slog(x: SLArrayList) -> ArrayList:
    """Map a SLArrayList to ArrayList form.

    Args:
        x (SLArrayList): input data as a list of slog arrays.

    Returns:
        (ArrayList): same data with slog tuples transformed to single arrays.
    """
    return [array_from_slog(slog) for slog in x]


def slog_multiply(x: SLArray, y: SLArray) -> SLArray:
    """Computes the product of two slog array tuples, as another slog array tuple.

    Signs are multiplied and logs are added.
    """
    (sx, lx) = x
    (sy, ly) = y
    return (sx * sy, lx + ly)


def slog_linear_comb(
    x: SLArray,
    weights: Optional[jnp.ndarray] = None,
    axis: int = 0,
    register_kfac: bool = True,
) -> SLArray:
    """Stably compute sum(x_i * w_ij) + b_j in log domain.

    In order to avoid overflow when computing

        log(abs(sum_i(sign_i * w_ij * exp(log_i)))),

    the largest exp(log_i) is divided out from all the values and added back in after
    the outer log, i.e.

        log(abs(sum_i(sign_i * w_ij * exp(log_i - max)))) + max.

    This trick also avoids the underflow issue of when all vals are small enough that
    exp(log_i) is approximately 0 for all i.

    Args:
        x (SLArray): slog array with shape (..., d, ...), where d is the size of the
            given axis.
        weights (jnp.ndarray, optional): weights of a linear transformation to apply to
            the given axis, with shape (d, d'). If not provided, a simple sum is taken
            instead, equivalent to (d, 1) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum and max. Defaults to 0.
        register_kfac (bool, optional): if weights are not None, whether to register the
            linear part of the computation with KFAC. Defaults to True.

    Returns:
        (SLArray): sign of linear combination, log of linear
        combination. Both outputs have shape (..., d', ...), where d' = 1 if weights is
        None, and d' = weights.shape[1] otherwise.
    """
    (signs, logs) = x
    max_log = jnp.max(logs, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(logs - max_log)
    if weights is not None:
        # swap axis and -1 to conform to jnp.dot and register_batch_dense api
        terms_divided_by_max = jnp.swapaxes(terms_divided_by_max, axis, -1)
        transformed_divided_by_max = jnp.dot(terms_divided_by_max, weights)
        if register_kfac:
            transformed_divided_by_max = register_batch_dense(
                transformed_divided_by_max, terms_divided_by_max, weights, None
            )

        # swap axis and -1 back after the contraction and registration
        transformed_divided_by_max = jnp.swapaxes(transformed_divided_by_max, axis, -1)
    else:
        transformed_divided_by_max = jnp.sum(
            terms_divided_by_max, axis=axis, keepdims=True
        )

    signs = jnp.sign(transformed_divided_by_max)
    logs = jnp.log(jnp.abs(transformed_divided_by_max)) + max_log
    return signs, logs


def sum_sl_array_list(x: SLArrayList) -> SLArray:
    """Take the sum of a list of SLArrays which are all of the same shape."""
    stacked_slog_vals = (jnp.stack([a[0] for a in x]), jnp.stack([a[1] for a in x]))
    return slog_linear_comb(stacked_slog_vals)


def slog_sum(x: SLArray, y: SLArray):
    """Take the sum of two SLArrays which are of the same shape."""
    return sum_sl_array_list([x, y])
