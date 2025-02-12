"""Test equivariant model parts."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.models as models
import vmcnet.physics as physics

from .utils import get_elec_hyperparams, get_input_streams_from_hyperparams


def test_electron_electron_add_norm():
    """Test that adding the norm normally gives the same output."""
    elec_pos = jnp.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ]
    )
    target_input_2e = physics.potential.compute_displacements(elec_pos, elec_pos)
    target_input_2e = jnp.concatenate(
        [target_input_2e, jnp.linalg.norm(target_input_2e, axis=-1, keepdims=True)],
        axis=-1,
    )
    input_2e, _ = models.equivariance.compute_electron_electron(
        elec_pos, include_ee_norm=True
    )
    np.testing.assert_allclose(
        input_2e,
        target_input_2e,
    )


@pytest.mark.slow
def test_ferminet_one_electron_layer_shape_and_equivariance():
    """Test the equivariance of the one-electron layer in the FermiNet."""
    nchains, nelec_total, nion, d, permutation, spin_split, _ = get_elec_hyperparams()

    (
        input_1e,
        input_2e,
        _,
        perm_input_1e,
        perm_input_2e,
        _,
        key,
    ) = get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    ndense = 10
    kernel_initializer_unmixed = models.weights.get_kernel_initializer("orthogonal")
    kernel_initializer_mixed = models.weights.get_kernel_initializer("lecun_normal")
    kernel_initializer_2e = models.weights.get_kernel_initializer("xavier_uniform")
    bias_initializer = models.weights.get_bias_initializer("normal")
    activation_fn = jnp.tanh

    for cyclic_spin in [False, True]:
        one_elec_layer = models.equivariance.FermiNetOneElectronLayer(
            spin_split,
            ndense,
            kernel_initializer_unmixed,
            kernel_initializer_mixed,
            kernel_initializer_2e,
            bias_initializer,
            activation_fn,
            cyclic_spins=cyclic_spin,
        )

        key, subkey = jax.random.split(key)
        params = one_elec_layer.init(subkey, input_1e, input_2e)

        output = one_elec_layer.apply(params, input_1e, input_2e)
        perm_output = one_elec_layer.apply(params, perm_input_1e, perm_input_2e)

        assert output.shape == (nchains, nelec_total, ndense)
        np.testing.assert_allclose(output[:, permutation, :], perm_output, atol=1e-5)


@pytest.mark.slow
def test_ferminet_two_electron_layer_shape_and_equivariance():
    """Test that the two-electron stream is doubly equivariant."""
    nchains, nelec_total, nion, d, permutation, _, _ = get_elec_hyperparams()

    (
        _,
        input_2e,
        _,
        _,
        perm_input_2e,
        _,
        key,
    ) = get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    ndense = 11
    kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
    bias_initializer = models.weights.get_bias_initializer("normal")
    activation_fn = jnp.tanh

    two_elec_layer = models.equivariance.FermiNetTwoElectronLayer(
        ndense, kernel_initializer, bias_initializer, activation_fn
    )

    key, subkey = jax.random.split(key)
    params = two_elec_layer.init(subkey, input_2e)

    output = two_elec_layer.apply(params, input_2e)
    perm_output = two_elec_layer.apply(params, perm_input_2e)

    assert output.shape == (nchains, nelec_total, nelec_total, ndense)
    np.testing.assert_allclose(
        output[:, permutation][..., permutation, :], perm_output, atol=1e-6
    )


@pytest.mark.slow
def test_split_dense_shape():
    """Test the output shape of the SplitDense layer."""
    nchains = 8
    nelec_per_spin = jnp.array([3, 1, 4, 2])
    nelec_total = jnp.sum(nelec_per_spin)
    d = 5

    split = (3, 4, 8)  # = jnp.cumsum(nspins)[:-1], locations to split at
    ndense = (4, 2, 7, 5)
    kernel_initializer = models.weights.get_kernel_initializer("xavier_normal")
    bias_initializer = models.weights.get_bias_initializer("uniform")

    split_dense_layer = models.equivariance.SplitDense(
        split, ndense, kernel_initializer, bias_initializer
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (nchains, nelec_total, d))

    key, subkey = jax.random.split(key)
    outputs, _ = split_dense_layer.init_with_output(subkey, inputs)

    for i, output in enumerate(outputs):
        chex.assert_shape(output, (nchains, nelec_per_spin[i], ndense[i]))
