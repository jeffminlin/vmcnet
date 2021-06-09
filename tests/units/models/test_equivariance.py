"""Test equivariant model parts."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models
import vmcnet.physics as physics


def _get_elec_hyperparams():
    nchains = 25
    nelec_total = 7
    d = 3
    permutation = (1, 0, 2, 5, 6, 3, 4)

    spin_split = (3,)
    return nchains, nelec_total, d, permutation, spin_split


def _get_input_streams_from_hyperparams(nchains, nelec_total, d, permutation):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    elec_pos = jax.random.normal(subkey, (nchains, nelec_total, d))
    permuted_elec_pos = elec_pos[:, permutation, :]

    input_1e, input_2e, _ = models.equivariance.compute_input_streams(elec_pos)
    perm_input_1e, perm_input_2e, _ = models.equivariance.compute_input_streams(
        permuted_elec_pos
    )
    return (
        input_1e,
        input_2e,
        perm_input_1e,
        perm_input_2e,
        key,
    )


def test_electron_electron_add_norm():
    """Test that adding the norm normally gives the same output."""
    elec_pos = jnp.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ]
    )
    target_input_2e = physics.potential._compute_displacements(elec_pos, elec_pos)
    target_input_2e = jnp.concatenate(
        [target_input_2e, jnp.linalg.norm(target_input_2e, axis=-1, keepdims=True)],
        axis=-1,
    )

    np.testing.assert_allclose(
        target_input_2e,
        models.equivariance.compute_electron_electron(elec_pos, include_ee_norm=True),
    )


def test_ferminet_one_electron_layer_shape_and_equivariance():
    """Test the equivariance of the one-electron layer in the FermiNet."""
    nchains, nelec_total, d, permutation, spin_split = _get_elec_hyperparams()

    (
        input_1e,
        input_2e,
        perm_input_1e,
        perm_input_2e,
        key,
    ) = _get_input_streams_from_hyperparams(nchains, nelec_total, d, permutation)

    ndense = 10
    kernel_initializer_unmixed = models.weights.get_kernel_initializer("orthogonal")
    kernel_initializer_mixed = models.weights.get_kernel_initializer("lecun_normal")
    kernel_initializer_2e = models.weights.get_kernel_initializer("xavier_uniform")
    bias_initializer = models.weights.get_bias_initializer("normal")
    activation_fn = jnp.tanh

    cyclic_spins_options = [False, True]

    for cyclic_spin in cyclic_spins_options:
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


def test_ferminet_two_electron_layer_shape_and_equivariance():
    """Test that the two-electron stream is doubly equivariant."""
    nchains, nelec_total, d, permutation, _ = _get_elec_hyperparams()

    (
        _,
        input_2e,
        _,
        perm_input_2e,
        key,
    ) = _get_input_streams_from_hyperparams(nchains, nelec_total, d, permutation)

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


def test_split_dense_shape():
    """Test the output shape of the SplitDense layer."""
    nchains = 8
    nspins = jnp.array([3, 1, 4, 2])
    nspins_total = jnp.sum(nspins)
    d = 5

    spin_split = (3, 4, 8)  # = jnp.cumsum(nspins)[:-1], locations to split at
    ndense = (4, 2, 7, 5)
    kernel_initializer = models.weights.get_kernel_initializer("xavier_normal")
    bias_initializer = models.weights.get_bias_initializer("uniform")

    split_dense_layer = models.equivariance.SplitDense(
        spin_split, ndense, kernel_initializer, bias_initializer
    )

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (nchains, nspins_total, d))

    key, subkey = jax.random.split(key)
    outputs, _ = split_dense_layer.init_with_output(subkey, inputs)

    for i, output in enumerate(outputs):
        assert output.shape[0] == 8
        assert output.shape[1] == nspins[i]
        assert output.shape[2] == ndense[i]
