"""Test model antisymmetries."""
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models


def test_slogdet_product():
    """Test that slogdet outputs are combined correctly over a pytree."""
    m1 = jnp.array(
        [
            [
                [1, 2],
                [3, 4],
            ],  # det = -2
            [
                [1, 2],
                [4, 3],
            ],  # det = -5
        ]
    )

    m2 = jnp.array(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],  # det = 0
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],  # det = -1
        ]
    )

    matrix_pytree = {0: m1, 1: (m2, m1)}
    sign_prod, log_prod = models.antisymmetry.slogdet_product(matrix_pytree)

    np.testing.assert_allclose(sign_prod, jnp.array([0, -1]))
    np.testing.assert_allclose(log_prod, jnp.array([-jnp.inf, jnp.log(25)]))
