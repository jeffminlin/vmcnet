"""Helper function for log sum exp trick with weights."""
from typing import Optional

import jax.numpy as jnp

from vmcnet.utils.typing import SLArray
from vmcnet.utils.kfac import register_batch_dense


def log_linear_exp(
    signs: jnp.ndarray,
    vals: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: int = 0,
    register_kfac: bool = True,
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
        signs (jnp.ndarray): array of signs of the input x with shape (..., d, ...),
            where d is the size of the given axis
        vals (jnp.ndarray): array of log|abs(x)| with shape (..., d, ...), where d is
            the size of the given axis
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
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    # The below handles the case where there is exactly one inf value in a given slice.
    # In that case, the subtraction of vals - max_val will be inf - inf at the maximal
    # index, which will give a nan. However, we can safely replace it with a zero as we
    # have actually pulled the inf term outside the sum anyway.
    value_is_inf = vals == jnp.inf
    max_val_is_neg_inf = max_val == -jnp.inf
    single_inf_in_slice = jnp.count_nonzero(value_is_inf, axis=axis, keepdims=True) == 1
    value_is_only_inf_in_slice = jnp.logical_and(value_is_inf, single_inf_in_slice)
    diffs_with_max = jnp.where(
        jnp.logical_or(value_is_only_inf_in_slice, max_val_is_neg_inf),
        0,
        vals - max_val,
    )

    terms_divided_by_max = signs * jnp.exp(diffs_with_max)
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
    vals = jnp.log(jnp.abs(transformed_divided_by_max)) + max_val
    return signs, vals
