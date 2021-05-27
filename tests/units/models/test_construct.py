"""Test model construction."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models


def test_compose_negative_with_three_body_antisymmetry():
    """Test a simple three-body antisymmetry composed with the *(-1) equivariance."""

    def equivariance(xs):
        return jax.tree_map(lambda a: -a, xs)

    def three_body_antisymmetry(x):
        return (
            (x[..., 0] - x[..., 1]) * (x[..., 0] - x[..., 2]) * (x[..., 1] - x[..., 2])
        )

    def product_antisymmetry(xs):
        antisymmetry_tree = jax.tree_map(three_body_antisymmetry, xs)
        return functools.reduce(lambda a, b: a * b, jax.tree_leaves(antisymmetry_tree))

    model = models.construct.ComposedModel([equivariance, product_antisymmetry])

    x = [
        jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),  # antisymmetry gives [-2, -2]
        jnp.array([[1, 3, 5], [7, 10, 13]]),  # antisymmetry gives [-16, -54]
        jnp.array([[1, 0, -1, 6], [-1, 0, 4, 2]]),  # antisymmetry gives [2, -20]
    ]

    output = model.apply({}, x)

    np.testing.assert_allclose(output, jnp.array([2 * 16 * -2, 2 * 54 * 20]))
