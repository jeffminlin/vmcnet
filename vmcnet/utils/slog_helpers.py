"""Helper functions for dealing with structured values."""
from typing import Sequence

import jax.numpy as jnp

from .typing import SLArray


def array_to_slog(x: jnp.ndarray) -> SLArray:
    """Converts an slog array tuple back into regular array form."""
    return (jnp.sign(x), jnp.log(jnp.abs(x)))


def array_from_slog(x: SLArray) -> jnp.ndarray:
    """Converts a regular array into slog form."""
    return x[0] * jnp.exp(x[1])


def slog_multiply(x: SLArray, y: SLArray) -> SLArray:
    """Computes the product of two slog array values."""
    (sx, lx) = x
    (sy, ly) = y
    return (sx * sy, lx + ly)


def slog_sum(inputs: SLArray, axis: int = 0) -> SLArray:
    """Stably compute log(abs(sum_i(sign_i * exp(vals_i)))) along an axis."""
    (signs, logs) = inputs
    max_val = jnp.max(logs, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(logs - max_val)
    sum_terms_divided_by_max = jnp.sum(terms_divided_by_max, axis=axis)
    return (
        jnp.sign(sum_terms_divided_by_max),
        jnp.log(jnp.abs(sum_terms_divided_by_max)) + jnp.squeeze(max_val, axis=axis),
    )
