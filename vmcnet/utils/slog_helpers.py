"""Helper functions for dealing with structured values."""

import jax.numpy as jnp

from .typing import SLArray, SpinSplitArray, SpinSplitSLArray


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


def spin_split_array_to_slog(x: SpinSplitArray) -> SpinSplitSLArray:
    """Map a SpinSplitArray to SpinSplitSLArray form.

    Args:
        x (SpinSplitArray): input data as a regular spin-split array.

    Returns:
        (SpinSplitSLArray): same data with each array transformed to slog form.
    """
    return [array_to_slog(arr) for arr in x]


def spin_split_array_from_slog(x: SpinSplitSLArray) -> SpinSplitArray:
    """Map a SpinSplitSLArray to SpinSplitArray form.

    Args:
        x (SpinSplitSLArray): input data as a spin-split slog array.

    Returns:
        (SpinSplitSLArray): same data with slog tuples transformed to single arrays.
    """
    return [array_from_slog(slog) for slog in x]


def slog_multiply(x: SLArray, y: SLArray) -> SLArray:
    """Computes the product of two slog array tuples, as another slog array tuple.

    Signs are multiplied and logs are added.
    """
    (sx, lx) = x
    (sy, ly) = y
    return (sx * sy, lx + ly)


# TODO (ggoldsh): once Jeffmin's PR $22 is merged, can merge this with the more general
# log_linear_exp fn implemented there.
def slog_sum(x: SLArray, axis: int = 0) -> SLArray:
    """Computes the sum of the input array along the given axis, in slog form.

    This operation is equivalent to converting the data from slog form to a normal
    array, calling jnp.sum() on the converted data, and then converting the data back
    to slog form. However, the calculation implemented here is more stable.

    Args:
        x (SLArray): the input data, in slog form.
        axis (int): the axis along which to sum.
    """
    (signs, logs) = x
    max_val = jnp.max(logs, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(logs - max_val)
    sum_terms_divided_by_max = jnp.sum(terms_divided_by_max, axis=axis)
    return (
        jnp.sign(sum_terms_divided_by_max),
        jnp.log(jnp.abs(sum_terms_divided_by_max)) + jnp.squeeze(max_val, axis=axis),
    )
