"""Antisymmetry parts to compose into a model."""
import functools
from typing import Tuple

import jax
import jax.numpy as jnp


def _istupleofarrays(x) -> bool:
    return isinstance(x, tuple) and all(isinstance(x_i, jnp.ndarray) for x_i in x)


def slogdet_product(xs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the (sign, log) of the product of determinants of the leaves of a pytree.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        (jnp.ndarray, jnp.ndarray): the product of the sign_dets and the sum of the
        log_dets over all leaves of the pytree xs
    """
    slogdets = jax.tree_map(jnp.linalg.slogdet, xs)

    slogdet_leaves, _ = jax.tree_flatten(slogdets, _istupleofarrays)
    sign_prod, log_prod = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdet_leaves
    )
    return sign_prod, log_prod


def logdet_product(xs) -> jnp.ndarray:
    """Compute the log|prod_x det(x)| of the leaves x of a pytree (throwing away sign).

    Because we don't need to carry sign, the logic can be made slightly simpler and
    we can avoid a few computations.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        jnp.ndarray: the sum of the log_dets over all leaves of the pytree xs, which is
        equal to the log of the product of the dets over all leaves of xs
    """
    logdets = jax.tree_map(lambda x: jnp.linalg.slogdet(x)[1], xs)
    log_prod = functools.reduce(lambda a, b: a + b, jax.tree_leaves(logdets))
    return log_prod
