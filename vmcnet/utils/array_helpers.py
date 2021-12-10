"""Helper functions for dealing with (sign, logabs) data."""

import jax.numpy as jnp

from .log_linear_exp import log_linear_exp
from .typing import Array, SLArray, ArrayList, SLArrayList


def negate_array(x: Array) -> Array:
    """Multiplies an array by -1 in a type-safe way.

    Mypy does not otherwise think the negation of an array results in an array,
    which can mess up type-checking.
    """
    return -x  # type: ignore


def add_arrays(x: Array, y: Array) -> Array:
    """Adds two arrays in a type-safe way.

    Mypy does not otherwise think the sum of two arrays results in an array,
    which can mess up type-checking.
    """
    return x + y  # type: ignore


def subtract_arrays(x: Array, y: Array) -> Array:
    """Subtracts two arrays in a typeful way.

    Mypy otherwise assigns no type to the subtraction of two arrays,
    which can mess up type-checking.
    """
    return x - y


def multiply_arrays(x: Array, y: Array) -> Array:
    """Multiplies two arrays in a type-safe way.

    Mypy does not otherwise think the product of two arrays results in an array,
    which can mess up type-checking.
    """
    return x * y  # type: ignore


def array_to_slog(x: Array) -> SLArray:
    """Converts a regular array into (sign, logabs) form.

    Args:
        x (Array): input data.

    Returns:
        (SLArray): data in form (sign(x), log(abs(x)))
    """
    return (jnp.sign(x), jnp.log(jnp.abs(x)))


def array_from_slog(x: SLArray) -> Array:
    """Converts an slog data tuple into a regular array.

    Args:
        x (SLArray): input data in slog form. This data looks like
        (sign(z), log(abs(z))) for some z which represents the underlying data.

    Returns:
        (Array): the data as a single, regular array. In other words, the z
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
    return (multiply_arrays(sx, sy), add_arrays(lx, ly))


def slog_sum_over_axis(x: SLArray, axis: int = 0) -> SLArray:
    """Take the sum of a single slog array over a specified axis."""
    signs, logs = log_linear_exp(x[0], x[1], axis=axis)
    return (jnp.squeeze(signs, axis=axis), jnp.squeeze(logs, axis=axis))


def slog_array_list_sum(x: SLArrayList) -> SLArray:
    """Take the sum of a list of SLArrays which are all of the same shape."""
    stacked_vals = (jnp.stack([a[0] for a in x]), jnp.stack([a[1] for a in x]))
    return slog_sum_over_axis(stacked_vals)


def slog_sum(x: SLArray, y: SLArray) -> SLArray:
    """Take the sum of two SLArrays which are of the same shape."""
    return slog_array_list_sum([x, y])
