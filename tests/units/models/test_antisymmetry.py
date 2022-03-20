"""Test model antisymmetries."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.models as models
from vmcnet.utils.slog_helpers import array_to_slog, slog_sum_over_axis


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


def test_lexicographic_order_of_perms():
    """Test getting all permutations and their signs for three things."""
    x = jnp.array([[1], [2], [3]])
    perm_layer = models.antisymmetry.ParallelPermutations(len(x))

    lexicographic_perms = [
        [[1], [2], [3]],
        [[1], [3], [2]],
        [[2], [1], [3]],
        [[2], [3], [1]],
        [[3], [1], [2]],
        [[3], [2], [1]],
    ]
    lexicographic_signs = [1, -1, -1, 1, 1, -1]

    key = jax.random.PRNGKey(0)
    (perms, signs), _ = perm_layer.init_with_output(key, x)

    np.testing.assert_array_equal(perms, lexicographic_perms)
    np.testing.assert_array_equal(signs, lexicographic_signs)


def _vandermonde_product(x):
    """Compute x_0^0 * x_1^1 * x_2^2 * ... * x_(n-1)^(n-1), x_i = x[..., i]."""
    vandermonde_powers = jnp.stack(
        [jnp.power(x[..., i], i) for i in range(x.shape[-1])], axis=-1
    )
    return jnp.prod(vandermonde_powers, axis=-1, keepdims=True)


def _diagonal_product(flattened_square_matrix, n):
    """Get product of diagonal entries of a flattened square matrix."""
    assert flattened_square_matrix.shape[-1] == n**2
    square_matrix = jnp.reshape(
        flattened_square_matrix, flattened_square_matrix.shape[:-1] + (n, n)
    )
    diags = jnp.diagonal(square_matrix, axis1=-2, axis2=-1)
    return jnp.prod(diags, axis=-1, keepdims=True)


@pytest.mark.slow
def test_factorized_antisymmetrize_vandermonde_product():
    """Test factorized antisym can be a product of vandermonde dets."""
    xs = [jnp.array([[3], [-2], [4]]), jnp.array([[1], [2], [3], [4]])]
    vandermonde_dets = [
        ((-2) - 3) * (4 - 3) * (4 - (-2)),
        (2 - 1) * (3 - 1) * (4 - 1) * (3 - 2) * (4 - 2) * (4 - 3),
    ]
    det_product = vandermonde_dets[0] * vandermonde_dets[1]
    slogdet_product = array_to_slog(det_product)

    for logabs in [False, True]:
        split_layer = models.antisymmetry.FactorizedAntisymmetrize(
            [_vandermonde_product, _vandermonde_product], logabs=logabs
        )

        key = jax.random.PRNGKey(0)
        output, _ = split_layer.init_with_output(key, xs)

        if logabs:
            chex.assert_shape(output, ((1,), (1,)))
            np.testing.assert_allclose(
                slog_sum_over_axis(output, axis=-1), slogdet_product
            )
        else:
            chex.assert_shape(output, (1,))
            np.testing.assert_allclose(output, det_product)


@pytest.mark.slow
def test_generic_antisymmetrize_product():
    """Check generic antisymmetrize can make a product of determinants."""
    x_matrices = [
        jnp.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 2],
            ]
        ),
        jnp.array(
            [
                [0, 0, 1, 1],
                [1, 2, 3, 4],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
            ]
        ),
    ]
    dets = [jnp.linalg.det(x) for x in x_matrices]
    det_product = dets[0] * dets[1]
    slogdet_product = array_to_slog(det_product)

    def fn_to_antisymmetrize(concat_flattened_matrices):
        prod_0 = _diagonal_product(concat_flattened_matrices[..., :9], 3)
        prod_1 = _diagonal_product(concat_flattened_matrices[..., 9:], 4)
        return prod_0 * prod_1

    for logabs in [False, True]:
        composed_layer = models.antisymmetry.GenericAntisymmetrize(
            fn_to_antisymmetrize,
            logabs=logabs,
        )

        key = jax.random.PRNGKey(0)
        output, _ = composed_layer.init_with_output(key, x_matrices)

        if logabs:
            np.testing.assert_allclose(output, slogdet_product, rtol=1e-6)
        else:
            np.testing.assert_allclose(output, det_product, rtol=1e-6)
