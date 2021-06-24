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


def _get_initial_pos_and_hyperparams():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    ion_pos = jax.random.normal(subkey, (5, 3))

    key, subkey = jax.random.split(key)
    init_pos = jax.random.normal(subkey, shape=(20, 6, 3))

    spin_split = (2,)  # 2 up, 4 down
    ndense_list = ((64, 8), (64, 8), (32, 8), (32, 4), (12,), (14, 3), (13,))

    return key, ion_pos, init_pos, spin_split, ndense_list


def test_ferminet_can_be_constructed():
    """Make sure basic construction of the FermiNet does not raise exceptions."""
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    for cyclic_spins in [True, False]:
        log_psi = models.construct.FermiNet(
            spin_split,
            ndense_list,
            3,
            models.weights.get_kernel_initializer("orthogonal"),
            models.weights.get_kernel_initializer("xavier_normal"),
            models.weights.get_kernel_initializer("glorot_uniform"),
            models.weights.get_kernel_initializer("kaiming_uniform"),
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("zeros"),
            models.weights.get_bias_initializer("normal"),
            models.weights.get_bias_initializer("uniform"),
            jnp.tanh,
            ion_pos=ion_pos,
            cyclic_spins=cyclic_spins,
        )

        key, subkey = jax.random.split(key)
        params = log_psi.init(subkey, init_pos)
        jax.jit(log_psi.apply)(params, init_pos)


def test_split_antisymmetry_can_be_constructed():
    """Check construction of SplitBruteForceAntisymmetryWithDecay does not fail."""
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    for cyclic_spins in [True, False]:
        log_psi = models.construct.SplitBruteForceAntisymmetryWithDecay(
            spin_split,
            ndense_list,
            32,
            3,
            models.weights.get_kernel_initializer("orthogonal"),
            models.weights.get_kernel_initializer("xavier_normal"),
            models.weights.get_kernel_initializer("kaiming_uniform"),
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("zeros"),
            models.weights.get_bias_initializer("normal"),
            models.weights.get_bias_initializer("uniform"),
            jnp.tanh,
            jnp.tanh,
            ion_pos=ion_pos,
            cyclic_spins=cyclic_spins,
        )

        key, subkey = jax.random.split(key)
        params = log_psi.init(subkey, init_pos)
        jax.jit(log_psi.apply)(params, init_pos)


def test_composed_antisymmetry_can_be_constructed():
    """Check construction of ComposedBruteForceAntisymmetryWithDecay does not fail."""
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    for cyclic_spins in [True, False]:
        log_psi = models.construct.ComposedBruteForceAntisymmetryWithDecay(
            spin_split,
            ndense_list,
            32,
            3,
            models.weights.get_kernel_initializer("orthogonal"),
            models.weights.get_kernel_initializer("xavier_normal"),
            models.weights.get_kernel_initializer("kaiming_uniform"),
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("zeros"),
            models.weights.get_bias_initializer("normal"),
            models.weights.get_bias_initializer("uniform"),
            jnp.tanh,
            jnp.tanh,
            ion_pos=ion_pos,
            cyclic_spins=cyclic_spins,
        )

        key, subkey = jax.random.split(key)
        params = log_psi.init(subkey, init_pos)
        jax.jit(log_psi.apply)(params, init_pos)
