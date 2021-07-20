"""Helper functions for pytrees."""
import functools

import jax
import jax.numpy as jnp

from vmcnet.utils.typing import PyTree


def tree_sum(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_prod(tree1: PyTree, tree2: PyTree) -> PyTree:
    """Leaf-wise product of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a * b, tree1, tree2)


def tree_reduce_l1(xs: PyTree) -> jnp.ndarray:
    """L1 norm of a pytree as a flattened vector."""
    return functools.reduce(
        lambda a, b: jnp.sum(jnp.abs(a)) + jnp.sum(jnp.abs(b)), jax.tree_leaves(xs)
    )
