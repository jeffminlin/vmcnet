"""Test equivariant model parts."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.models as models
import vmcnet.physics as physics

from .utils import get_elec_hyperparams, get_input_streams_from_hyperparams
from itertools import product


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
    kernel_initializer_transformer = models.weights.get_kernel_initializer("orthogonal")
    kernel_initializer_unmixed = models.weights.get_kernel_initializer("orthogonal")
    kernel_initializer_mixed = models.weights.get_kernel_initializer("lecun_normal")
    kernel_initializer_2e = models.weights.get_kernel_initializer("xavier_uniform")
    bias_initializer = models.weights.get_bias_initializer("normal")
    bias_initializer_transformer = models.weights.get_bias_initializer("normal")
    activation_fn = jnp.tanh

    num_heads_options = [1, 3]
    cyclic_spins_options = [False, True]
    use_transformer_options = [False, True]

    for num_heads, cyclic_spin, use_transformer in product(
        num_heads_options, cyclic_spins_options, use_transformer_options
    ):
        one_elec_layer = models.equivariance.FermiNetOneElectronLayer(
            spin_split,
            ndense,
            kernel_initializer_transformer,
            kernel_initializer_unmixed,
            kernel_initializer_mixed,
            kernel_initializer_2e,
            bias_initializer,
            bias_initializer_transformer,
            activation_fn,
            cyclic_spins=cyclic_spin,
            use_transformer=use_transformer,
            num_heads=num_heads,
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


@pytest.mark.slow
def test_doubly_equivariant_orbital_layer_shape_and_equivariance():
    """Test that the output of the layer has the correct shape and symmetry.

    The output should have shape [nspins: (nchains, nelec, nelec, norbitals)].
    Furthermore, the -2 and -3 axes of each array should both be equivariant.
    """
    (
        nchains,
        nelec_total,
        nion,
        d,
        permutation,
        orbitals_split,
        split_perm,
    ) = get_elec_hyperparams()
    (
        input_1e,
        _,
        input_ei,
        perm_input_1e,
        _,
        perm_input_ei,
        key,
    ) = get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    nelec_per_spin = models.core.get_nelec_per_split(orbitals_split, nelec_total)
    norbitals_per_split = [2 * n for n in nelec_per_spin]
    nspins = len(nelec_per_spin)
    kernel_initializer = models.weights.get_kernel_initializer("xavier_normal")
    bias_initializer = models.weights.get_bias_initializer("uniform")

    equivariant_orbital_layer = models.equivariance.DoublyEquivariantOrbitalLayer(
        orbitals_split,
        norbitals_per_split,
        kernel_initializer,
        kernel_initializer,
        kernel_initializer,
        bias_initializer,
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    output, params = equivariant_orbital_layer.init_with_output(
        subkey, input_1e, input_ei
    )
    perm_output = equivariant_orbital_layer.apply(params, perm_input_1e, perm_input_ei)

    for i in range(nspins):
        nelec = nelec_per_spin[i]
        norbitals = norbitals_per_split[i]
        out_i = output[i]
        chex.assert_shape(out_i, (nchains, nelec, nelec, norbitals))

        perm_out_i = perm_output[i]
        perm_i = jnp.array(split_perm[i])
        # Both the orbital matrix index and the particle index should be permuted
        expected_perm_out_i = out_i[:, perm_i, :, :][:, :, perm_i, :]

        np.testing.assert_allclose(perm_out_i, expected_perm_out_i, 1e-5)


@pytest.mark.slow
def test_doubly_equivariant_orbital_layer_no_batch_dims():
    """Test that the layer can be evaluated on inputs with no batch dimensions.

    An initial implementation of this layer did not satisfy this criterion; hence the
    regression test.
    """
    _, nelec_total, nion, d, permutation, orbitals_split, _ = get_elec_hyperparams()
    # Set nchains to 1 to get effectively batchless inputs
    nchains = 1
    input_1e, _, input_ei, _, _, _, key = get_input_streams_from_hyperparams(
        nchains, nelec_total, nion, d, permutation
    )
    # Delete length 1 batch dim from inputs to get truly batchless inputs.
    input_1e = jnp.squeeze(input_1e, 0)
    input_ei - jnp.squeeze(input_ei, 0)

    nelec_per_spin = models.core.get_nelec_per_split(orbitals_split, nelec_total)
    norbitals_per_split = [2 * n for n in nelec_per_spin]
    kernel_initializer = models.weights.get_kernel_initializer("xavier_normal")
    bias_initializer = models.weights.get_bias_initializer("uniform")

    equivariant_orbital_layer = models.equivariance.DoublyEquivariantOrbitalLayer(
        orbitals_split,
        norbitals_per_split,
        kernel_initializer,
        kernel_initializer,
        kernel_initializer,
        bias_initializer,
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    output, _ = equivariant_orbital_layer.init_with_output(subkey, input_1e, input_ei)
