"""Helper function for log sum exp trick with weights."""
from typing import Optional

import jax.numpy as jnp

from vmcnet.utils.typing import Array, SLArray


def log_linear_exp(
    signs: Array,
    vals: Array,
    weights: Optional[Array] = None,
    axis: int = 0,
) -> SLArray:
    """Stably compute sign and log(abs(.)) of sum_i(sign_i * w_ij * exp(vals_i)) + b_j.

    In order to avoid overflow when computing

        log(abs(sum_i(sign_i * w_ij * exp(vals_i)))),

    the largest exp(val_i) is divided out from all the values and added back in after
    the outer log, i.e.

        log(abs(sum_i(sign_i * w_ij * exp(vals_i - max)))) + max.

    This trick also avoids the underflow issue of when all vals are small enough that
    exp(val_i) is approximately 0 for all i.

    Args:
        signs (Array): array of signs of the input x with shape (..., d, ...),
            where d is the size of the given axis
        vals (Array): array of log|abs(x)| with shape (..., d, ...), where d is
            the size of the given axis
        weights (Array, optional): weights of a linear transformation to apply to
            the given axis, with shape (d, d'). If not provided, a simple sum is taken
            instead, equivalent to (d, 1) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum and max. Defaults to 0.

    Returns:
        (SLArray): sign of linear combination, log of linear
        combination. Both outputs have shape (..., d', ...), where d' = 1 if weights is
        None, and d' = weights.shape[1] otherwise.
    """
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(vals - max_val)
    if weights is not None:
        # swap axis and -1 to conform to jnp.dot api
        terms_divided_by_max = jnp.swapaxes(terms_divided_by_max, axis, -1)
        transformed_divided_by_max = jnp.dot(terms_divided_by_max, weights)

        # swap axis and -1 back after the contraction
        transformed_divided_by_max = jnp.swapaxes(transformed_divided_by_max, axis, -1)
    else:
        transformed_divided_by_max = jnp.sum(
            terms_divided_by_max, axis=axis, keepdims=True
        )

    signs = jnp.sign(transformed_divided_by_max)
    vals = jnp.log(jnp.abs(transformed_divided_by_max)) + max_val
    return signs, vals
