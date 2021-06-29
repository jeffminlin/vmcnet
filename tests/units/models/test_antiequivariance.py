"""Test model antiequivariances."""
import jax.numpy as jnp
import numpy as np
import vmcnet.models as models
import vmcnet.models.antiequivariance as antieq

from .utils import _get_elec_hyperparams, _get_input_streams_from_hyperparams


def test_slog_cofactor_output_with_batches():
    """Test slog_cofactor_antieq outputs correct value on simple inputs."""
    base_input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    negative_input = -base_input
    doubled_input = base_input * 2
    input = jnp.stack([base_input, negative_input, doubled_input])

    base_output = jnp.array([-3.0, 12.0, -9.0])
    base_signs = jnp.sign(base_output)
    base_logs = jnp.log(jnp.abs(base_output))
    output_signs = jnp.stack([base_signs, -base_signs, base_signs])
    output_logs = jnp.stack([base_logs, base_logs, base_logs + jnp.log(8)])

    y = antieq.slog_cofactor_antieq(input)

    np.testing.assert_allclose(y[0], output_signs)
    np.testing.assert_allclose(y[1], output_logs, rtol=1e-6)


def test_slog_cofactor_antiequivarance():
    """Test slog_cofactor_antieq is antiequivariant."""
    input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    permutation = (1, 0, 2)
    perm_input = input[permutation, :]

    output_signs, output_logs = antieq.slog_cofactor_antieq(input)
    perm_signs, perm_logs = antieq.slog_cofactor_antieq(perm_input)

    # Fancy indexing like output_signs[permutation] doesn't work on single
    # dimensional arrays, thinking the elements of permutation are intended for
    # different array axes. Hence, use jnp.take.
    expected_perm_signs = -jnp.take(output_signs, permutation)
    expected_perm_logs = jnp.take(output_logs, permutation)

    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs)


def test_orbital_cofactor_layer_antiequivariance():
    """Test evaluation and antiequivariance of orbital cofactor equivariance layer."""
    nchains, nelec_total, nion, d, permutation, spin_split = _get_elec_hyperparams()

    (
        input_1e,
        _,
        input_ei,
        perm_input_1e,
        _,
        perm_input_ei,
        key,
    ) = _get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
    bias_initializer = models.weights.get_bias_initializer("normal")
    norbitals = models.core.get_nelec_per_spin(spin_split, nelec_total)

    orbital_layer = models.equivariance.FermiNetOrbitalLayer(
        spin_split,
        norbitals,
        kernel_initializer,
        kernel_initializer,
        kernel_initializer,
        bias_initializer,
    )
    orbital_cofactor_antieq = antieq.OrbitalCofactorAntiequivarianceLayer(orbital_layer)
    params = orbital_cofactor_antieq.init(key, input_1e, input_ei)

    output_signs, output_logs = orbital_cofactor_antieq.apply(
        params, input_1e, input_ei
    )
    assert output_signs.shape == (nchains, nelec_total, d * (1 + nion))
    assert output_logs.shape == (nchains, nelec_total, d * (1 + nion))

    perm_signs, perm_logs = orbital_cofactor_antieq.apply(
        params, perm_input_1e, perm_input_ei
    )

    # First spin permutation is odd; second spin permutation is even
    flips = jnp.reshape(jnp.array([-1, -1, -1, 1, 1, 1, 1]), (1, 7, 1))
    expected_perm_signs = output_signs[:, permutation, :] * flips
    expected_perm_logs = output_logs[:, permutation, :]

    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs)
