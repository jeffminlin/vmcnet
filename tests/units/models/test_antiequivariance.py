"""Test model antiequivariances."""
from typing import Any, Callable, Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.models as models
import vmcnet.models.antiequivariance as antieq
from vmcnet.utils.slog_helpers import slog_sum_over_axis
from vmcnet.utils.typing import Array, ParticleSplit

from .utils import get_elec_hyperparams, get_input_streams_from_hyperparams


def _get_singular_matrix():
    return jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])


def _get_nonsingular_matrix():
    return jnp.array([[1, 4, 7], [2, 5, 8], [3, 5, 4]])


@pytest.mark.slow
def test_cofactor_output_with_batches():
    """Test cofactor_antieq outputs correct value on simple inputs."""
    input = _get_singular_matrix()
    negative_input = -input
    doubled_input = input * 2
    full_input = jnp.stack([input, negative_input, doubled_input])

    expected_out = jnp.array([-3, 12, -9])
    full_expected_out = jnp.stack([expected_out, -expected_out, 8 * expected_out])

    y = antieq.cofactor_antieq(full_input)

    np.testing.assert_allclose(y, full_expected_out, rtol=1e-6)


@pytest.mark.slow
def test_slog_cofactor_output_with_batches():
    """Test slog_cofactor_antieq outputs correct value on simple inputs."""
    input = _get_singular_matrix()
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


@pytest.mark.slow
def test_sum_cofactor_equals_nonzero_det():
    """Test that the sum of the cofactors along the first col give the determinant."""
    nonsing_in = _get_nonsingular_matrix()
    y = antieq.cofactor_antieq(nonsing_in)
    np.testing.assert_allclose(jnp.sum(y), jnp.linalg.det(nonsing_in), rtol=1e-6)


@pytest.mark.slow
def test_sum_slog_cofactor_equals_nonzero_slogdet():
    """Test that the slog_sum of the slog cofactors along the first col give slogdet."""
    nonsing_in = _get_nonsingular_matrix()
    y = antieq.slog_cofactor_antieq(nonsing_in)
    expected_signs, expected_logs = jnp.linalg.slogdet(nonsing_in)
    signs, logs = slog_sum_over_axis(y)

    np.testing.assert_allclose(signs, expected_signs)
    np.testing.assert_allclose(logs, expected_logs, rtol=1e-6)


@pytest.mark.slow
def test_cofactor_antiequivariance():
    """Test cofactor_antieq is antiequivariant."""
    input = _get_singular_matrix()
    permutation = jnp.array([1, 0, 2])
    perm_input = input[permutation, :]

    output = antieq.cofactor_antieq(input)
    perm_output = antieq.cofactor_antieq(perm_input)
    expected_perm_output = -output[permutation]

    np.testing.assert_allclose(perm_output, expected_perm_output)


@pytest.mark.slow
def test_slog_cofactor_antiequivariance():
    """Test slog_cofactor_antieq is antiequivariant."""
    input = _get_singular_matrix()
    permutation = jnp.array([1, 0, 2])
    perm_input = input[permutation, :]

    output_signs, output_logs = antieq.slog_cofactor_antieq(input)
    perm_signs, perm_logs = antieq.slog_cofactor_antieq(perm_input)

    expected_perm_signs = -output_signs[permutation]
    expected_perm_logs = output_logs[permutation]

    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs)


def _assert_slogabs_signs_allclose_to_one(
    nchains: int,
    output_i: Tuple[Array, Array],
    nelec_i: int,
):
    assert len(output_i) == 2
    chex.assert_shape(output_i, (nchains, nelec_i, 1))
    np.testing.assert_allclose(
        jnp.abs(output_i[0]),
        jnp.ones((nchains, nelec_i, 1)),
    )


def _assert_permuted_slog_values_allclose(
    split_perm_i: Tuple[int, ...],
    output_i: Tuple[Array, Array],
    perm_output_i: Tuple[Array, Array],
    flips_i: int,
    rtol: float = 1e-7,
    atol: float = 1e-7,
):
    signs, logs = output_i
    perm_signs, perm_logs = perm_output_i
    expected_perm_signs = signs[:, split_perm_i, :] * flips_i
    expected_perm_logs = logs[:, split_perm_i, :]
    np.testing.assert_allclose(perm_signs, expected_perm_signs)
    np.testing.assert_allclose(perm_logs, expected_perm_logs, rtol=rtol, atol=atol)


def _test_layer_antiequivariance(
    build_layer: Callable[[ParticleSplit], models.core.Module],
    logabs: bool = False,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Test evaluation and antiequivariance of an antiequivariant layer."""
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
    antieq_layer = build_layer(spin_split)
    params = antieq_layer.init(key, input_1e, input_ei)

    # Evaluate layer on original and permuted inputs
    output: Any = antieq_layer.apply(params, input_1e, input_ei)
    perm_output: Any = antieq_layer.apply(params, perm_input_1e, perm_input_ei)

    # Verify output shape and verify all signs values are  +-1
    nelec_per_spin = models.core.get_nelec_per_split(spin_split, nelec_total)
    nspins = len(nelec_per_spin)
    assert len(output) == nspins

    # Verify that permutation has generated appropriate antiequivariant transformation
    flips = [-1, 1]  # First spin permutation is odd; second is even

    for i in range(nspins):
        if logabs:
            _assert_slogabs_signs_allclose_to_one(nchains, output[i], nelec_per_spin[i])
            _assert_permuted_slog_values_allclose(
                split_perm[i], output[i], perm_output[i], flips[i], rtol=rtol, atol=atol
            )
        else:
            chex.assert_shape(output[i], (nchains, nelec_per_spin[i], 1))
            np.testing.assert_allclose(
                output[i],
                perm_output[i][:, split_perm[i], :] * flips[i],
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.slow
def test_orbital_cofactor_layer_antiequivariance():
    """Test orbital cofactor antiequivariance."""

    def build_orbital_cofactor_layer(spin_split):
        kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
        bias_initializer = models.weights.get_bias_initializer("normal")
        return antieq.OrbitalCofactorAntiequivarianceLayer(
            spin_split,
            kernel_initializer,
            kernel_initializer,
            kernel_initializer,
            bias_initializer,
        )

    _test_layer_antiequivariance(build_orbital_cofactor_layer, atol=1e-6)


@pytest.mark.slow
def test_slog_orbital_cofactor_layer_antiequivariance():
    """Test orbital cofactor antiequivariance."""

    def build_orbital_cofactor_layer(spin_split):
        kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
        bias_initializer = models.weights.get_bias_initializer("normal")
        return antieq.SLogOrbitalCofactorAntiequivarianceLayer(
            spin_split,
            kernel_initializer,
            kernel_initializer,
            kernel_initializer,
            bias_initializer,
        )

    _test_layer_antiequivariance(build_orbital_cofactor_layer, logabs=True)


@pytest.mark.slow
def test_per_particle_determinant_antiequivariance():
    """Test per particle determinant antiequivariance."""

    def build_per_particle_determinant_layer(spin_split):
        kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
        bias_initializer = models.weights.get_bias_initializer("normal")
        return antieq.PerParticleDeterminantAntiequivarianceLayer(
            spin_split,
            kernel_initializer,
            kernel_initializer,
            kernel_initializer,
            bias_initializer,
        )

    _test_layer_antiequivariance(
        build_per_particle_determinant_layer, rtol=1e-5, atol=1e-6
    )


@pytest.mark.skip("Fragile test for non-active code; better not to run it.")
@pytest.mark.slow
def test_slog_per_particle_determinant_antiequivariance():
    """Test per particle determinant antiequivariance."""

    def build_per_particle_determinant_layer(spin_split):
        kernel_initializer = models.weights.get_kernel_initializer("orthogonal")
        bias_initializer = models.weights.get_bias_initializer("normal")
        return antieq.SLogPerParticleDeterminantAntiequivarianceLayer(
            spin_split,
            kernel_initializer,
            kernel_initializer,
            kernel_initializer,
            bias_initializer,
        )

    _test_layer_antiequivariance(
        build_per_particle_determinant_layer, logabs=True, rtol=1e-6
    )
