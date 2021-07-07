"""Helper functions for dealing with (sign, logabs) data."""

import jax.numpy as jnp

from .log_linear_exp import log_linear_exp
from .typing import SLArray, ArrayList, SLArrayList


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


def slog_sum_over_axis(x: SLArray, axis: int = 0) -> SLArray:
    """Take the sum of a single slog array over a specified axis."""
    signs, logs = log_linear_exp(x[0], x[1], axis=axis)
    return (jnp.squeeze(signs, axis=axis), jnp.squeeze(logs, axis=axis))


def slog_array_list_sum(x: SLArrayList) -> SLArray:
    """Take the sum of a list of SLArrays which are all of the same shape."""
    stacked_vals = slog_array_list_stack(x)
    return slog_sum_over_axis(stacked_vals)


def slog_array_list_stack(x: SLArrayList, axis: int = 0) -> SLArray:
    """Stack a list of SLArrays which are all of the same shape."""
    return (
        jnp.stack([a[0] for a in x], axis=axis),
        jnp.stack([a[1] for a in x], axis=axis),
    )


def slog_array_list_concat(x: SLArrayList, axis: int = 0) -> SLArray:
    """Concat a list of SLArrays of the same shape except on the specified axis."""
    return (
        jnp.concatenate([a[0] for a in x], axis=axis),
        jnp.concatenate([a[1] for a in x], axis=axis),
    )


def slog_sum(x: SLArray, y: SLArray) -> SLArray:
    """Take the sum of two SLArrays which are of the same shape."""
    return slog_array_list_sum([x, y])


def slog_ones_like(x: SLArray) -> SLArray:
    """Generate array of ones matching input shape, in slog form."""
    return (jnp.ones_like(x[0]), jnp.zeros_like(x[0]))


def slog_flip_sign(x: SLArray) -> SLArray:
    """Flip the sign of an SLArray value."""
    return (-x[0], x[1])
