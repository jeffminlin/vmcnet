"""Helper functions for pytrees."""
import chex
import jax
import jax.numpy as jnp
import jax.flatten_util

from vmcnet.utils.typing import Array, PyTree, T


def tree_sum(tree1: T, tree2: T) -> T:
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_diff(tree1: T, tree2: T) -> T:
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a - b, tree1, tree2)


def tree_dist(tree1: T, tree2: T, mode="squares") -> Array:
    """Distance between two pytrees with the same structure."""
    dT = tree_diff(tree1, tree2)
    if mode == "squares":
        return tree_inner_product(dT, dT)
    raise ValueError(f"Unknown mode {mode}")


def tree_prod(tree1: T, tree2: T) -> T:
    """Leaf-wise product of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a * b, tree1, tree2)


def multiply_tree_by_scalar(tree: T, scalar: chex.Numeric) -> T:
    """Multiply all leaves of a pytree by a scalar."""
    return jax.tree_map(lambda x: scalar * x, tree)


def tree_inner_product(tree1: T, tree2: T) -> Array:
    """Inner product of two pytrees with the same structure."""
    leaf_inner_prods = jax.tree_map(lambda a, b: jnp.sum(a * b), tree1, tree2)
    return jnp.sum(jax.flatten_util.ravel_pytree(leaf_inner_prods)[0])


def tree_reduce_l1(xs: PyTree) -> chex.Numeric:
    """L1 norm of a pytree as a flattened vector."""
    concat_xs, _ = jax.flatten_util.ravel_pytree(xs)
    return jnp.sum(jnp.abs(concat_xs))
