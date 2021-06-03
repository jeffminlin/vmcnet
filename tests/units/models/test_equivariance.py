"""Test equivariant model parts."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models
import vmcnet.physics as physics


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


def test_ferminet_one_electron_layer_equivariance():
    """Test the equivariance of the one-electron layer in the FermiNet."""
    nchains = 25
    nelec_total = 7
    d = 3
    permutation = (1, 0, 2, 5, 6, 3, 4)

    spin_split = (3,)
    ndense = 10
    kernel_initializer_unmixed = models.weights.get_kernel_initializer("orthogonal")
    kernel_initializer_mixed = models.weights.get_kernel_initializer("lecun_normal")
    kernel_initializer_2e = models.weights.get_kernel_initializer("xavier_uniform")
    bias_initializer = models.weights.get_bias_initializer("normal")
    activation_fn = jnp.tanh

    cyclic_spins_options = [False, True]

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    elec_pos = jax.random.normal(subkey, (nchains, nelec_total, d))
    permuted_elec_pos = elec_pos[:, permutation, :]

    input_1e, input_2e, _ = models.equivariance.compute_input_streams(elec_pos)
    perm_input_1e, perm_input_2e, _ = models.equivariance.compute_input_streams(
        permuted_elec_pos
    )

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

        np.testing.assert_allclose(output[:, permutation, :], perm_output, atol=1e-5)
