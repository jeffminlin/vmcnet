"""Test model antiequivariances."""
import chex
import jax.numpy as jnp
import numpy as np
import vmcnet.models as models
import vmcnet.models.antiequivariance as antieq

from .utils import get_elec_hyperparams, get_input_streams_from_hyperparams


def test_slog_cofactor_output_with_batches():
    """Test slog_cofactor_antieq outputs correct value on simple inputs."""
    input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    negative_input = -input
    doubled_input = input * 2
    full_input = jnp.stack([input, negative_input, doubled_input])

    expected_out = jnp.array([-3, 12, -9])
    expected_signs = jnp.sign(expected_out)
    expected_logs = jnp.log(jnp.abs(expected_out))
    full_expected_signs = jnp.stack([expected_signs, -expected_signs, expected_signs])
    full_expected_logs = jnp.stack(
        [expected_logs, expected_logs, expected_logs + jnp.log(8)]
    )

    y = antieq.slog_cofactor_antieq(full_input)

    np.testing.assert_allclose(y[0], full_expected_signs)
    np.testing.assert_allclose(y[1], full_expected_logs, rtol=1e-6)


def test_slog_cofactor_antiequivarance():
    """Test slog_cofactor_antieq is antiequivariant."""
    input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    permutation = jnp.array([1, 0, 2])
    perm_input = input[permutation, :]

    output_signs, output_logs = antieq.slog_cofactor_antieq(input)
    perm_signs, perm_logs = antieq.slog_cofactor_antieq(perm_input)

    expected_perm_signs = -output_signs[permutation]
    expected_perm_logs = output_logs[permutation]

    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs)


def test_orbital_cofactor_layer_antiequivariance():
    """Test evaluation and antiequivariance of orbital cofactor equivariance layer."""
    # Generate example hyperparams and input streams
    (
        nchains,
        nelec_total,
        nion,
        d,
        permutation,
        spin_split,
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

    # Set up antiequivariant layer
    kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
    bias_initializer = models.weights.get_bias_initializer("normal")
    orbital_cofactor_antieq = antieq.OrbitalCofactorAntiequivarianceLayer(
        spin_split,
        kernel_initializer,
        kernel_initializer,
        kernel_initializer,
        bias_initializer,
    )
    params = orbital_cofactor_antieq.init(key, input_1e, input_ei)

    # Evaluate layer on original and permuted inputs
    output = orbital_cofactor_antieq.apply(params, input_1e, input_ei)
    perm_output = orbital_cofactor_antieq.apply(params, perm_input_1e, perm_input_ei)

    # Verify output shape and verify all signs values are  +-1
    nelec_per_spin = models.core.get_nelec_per_spin(spin_split, nelec_total)
    nspins = len(nelec_per_spin)
    assert len(output) == nspins
    for i in range(nspins):
        assert len(output[i]) == 2
        d_input_1e = input_1e.shape[-1]
        np.testing.assert_allclose(
            jnp.abs(output[i][0]),
            jnp.ones((nchains, nelec_per_spin[i], d_input_1e)),
        )
        chex.assert_shape(output[i][1], (nchains, nelec_per_spin[i], d_input_1e))

    # Verify that permutation has generated appropriate antiequivariant transformation
    flips = [-1, 1]  # First spin permutation is odd; second is even
    for i in range(nspins):
        signs, logs = output[i]
        perm_signs, perm_logs = perm_output[i]
        expected_perm_signs = signs[:, split_perm[i], :] * flips[i]
        expected_perm_logs = logs[:, split_perm[i], :]
        np.testing.assert_allclose(perm_signs, expected_perm_signs)
        np.testing.assert_allclose(perm_logs, expected_perm_logs)
