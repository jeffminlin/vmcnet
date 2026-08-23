"""Tests for the streaming Nyström-SPRING prototype."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

import vmcnet.examples.harmonic_oscillator as qho
import vmcnet.models as models
import vmcnet.physics as physics
from vmcnet.mcmc.position_amplitude_core import (
    get_position_from_data,
    get_update_data_fn,
)
from vmcnet.mcmc.simple_position_amplitude import make_simple_position_amplitude_data
from vmcnet.updates.spring_nystrom import (
    apply_nystrom_inverse_to_flat_vector,
    get_effective_rank,
    get_nystrom_metric_parameters,
    get_phasein_coefficient,
    get_nystrom_eigendecomposition_from_sketch,
    initialize_nystrom_state,
    normalize_nystrom_eigenvalues,
    initialize_spring_nystrom,
    reconstruct_nystrom_matrix,
    solve_preconditioned_spring_system,
)


class TinyTrial(models.core.Module):
    """Small model for Nyström-SPRING tests."""

    width: int = 3

    @nn.compact
    def __call__(self, x):
        h = models.core.Dense(self.width)(x)
        h = jnp.tanh(h)
        h = models.core.Dense(1)(h)
        return -0.25 * jnp.sum(jnp.square(x), axis=(-2, -1)) + (
            0.1 * jnp.sum(h, axis=(-2, -1))
        )


def _make_test_inputs(nchains=4):
    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, shape=(nchains, 1, 1))
    model = TinyTrial()
    key, subkey = jax.random.split(key)
    params = model.init(subkey, positions)
    data = make_simple_position_amplitude_data(
        positions, model.apply(params, positions)
    )
    return model, params, data, key


def _make_config():
    return ConfigDict(
        dict(
            schedule_type="constant",
            learning_rate=0.02,
            learning_decay_rate=0.0,
            mu=0.99,
            sketch_damping=1e-3,
            constrain_norm=False,
            norm_constraint=1e-3,
            nystrom_rank=2,
            nystrom_ema_decay=0.0,
            nystrom_seed=17,
            metric_shift_strategy="constant",
            metric_identity_shift=1.0,
            metric_normalization="none",
            nystrom_warmup_steps=1,
            nystrom_phasein_steps=1,
            eigenvalue_floor=1e-8,
            collect_during_warmup=True,
        )
    )


def test_phasein_coefficient_schedule():
    assert get_phasein_coefficient(jnp.asarray(0), 1, 2) == 0.0
    assert get_phasein_coefficient(jnp.asarray(2), 1, 2) == 0.5
    assert get_phasein_coefficient(jnp.asarray(3), 1, 2) == 1.0


def test_nystrom_approximation_improves_with_rank():
    basis_seed = jnp.asarray(
        [
            [1.0, 0.3, -0.2, 0.4],
            [0.2, 1.0, 0.5, -0.3],
            [-0.4, 0.1, 1.0, 0.2],
            [0.3, -0.5, 0.1, 1.0],
        ]
    )
    eigenvectors, _ = jnp.linalg.qr(basis_seed)
    eigenvalues = jnp.asarray([6.0, 2.0, 0.5, 0.1])
    matrix = (eigenvectors * eigenvalues) @ eigenvectors.T

    def get_approximation_error(rank):
        omega = eigenvectors[:, :rank]
        y_matrix = matrix @ omega
        decomposition = get_nystrom_eigendecomposition_from_sketch(
            omega, y_matrix, 1e-8
        )
        approximation = reconstruct_nystrom_matrix(
            decomposition.eigenvectors, decomposition.eigenvalues
        )
        return jnp.linalg.norm(matrix - approximation), approximation

    rank_one_error, _ = get_approximation_error(1)
    rank_three_error, _ = get_approximation_error(3)
    full_rank_error, full_rank_approximation = get_approximation_error(4)

    assert rank_three_error < rank_one_error
    np.testing.assert_allclose(full_rank_error, 0.0, atol=2e-5)
    np.testing.assert_allclose(full_rank_approximation, matrix, atol=2e-5)


def test_nystrom_inverse_application_matches_dense_metric_inverse():
    eigenvectors = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    eigenvalues = jnp.asarray([3.0, 1.0])
    phasein = 0.75
    metric_shift = 2.0
    vector = jnp.asarray([2.0, -1.0, 4.0])

    diagonal_weight = (1.0 - phasein) + phasein * metric_shift
    dense_metric = diagonal_weight * jnp.eye(3) + phasein * (
        eigenvectors * eigenvalues
    ) @ eigenvectors.T

    actual = apply_nystrom_inverse_to_flat_vector(
        vector, eigenvectors, eigenvalues, phasein, metric_shift
    )
    expected = jnp.linalg.solve(dense_metric, vector)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_regularization_coupled_metric_is_identity_plus_fisher_over_damping():
    eigenvalues = jnp.asarray([3.0, 1.0])
    damping = 0.2

    metric_eigenvalues, metric_shift, metric_scale = get_nystrom_metric_parameters(
        eigenvalues,
        metric_identity_shift=7.0,
        metric_shift_strategy="regularization_coupled",
        sketch_damping=damping,
        eigenvalue_floor=1e-8,
    )

    np.testing.assert_allclose(metric_eigenvalues, eigenvalues / damping)
    np.testing.assert_allclose(metric_shift, 1.0)
    np.testing.assert_allclose(metric_scale, damping)


def test_trace_effective_rank_normalization_sets_weighted_average_to_one():
    eigenvalues = jnp.asarray([10.0, 2.0, 1.0, 0.5])

    normalized_eigenvalues, metric_scale = normalize_nystrom_eigenvalues(
        eigenvalues, "trace_effective_rank", 1e-8
    )
    weighted_average = (
        jnp.sum(jnp.square(normalized_eigenvalues))
        / jnp.sum(normalized_eigenvalues)
    )

    np.testing.assert_allclose(
        metric_scale,
        jnp.sum(jnp.square(eigenvalues)) / jnp.sum(eigenvalues),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(weighted_average, 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        get_effective_rank(normalized_eigenvalues),
        get_effective_rank(eigenvalues),
        rtol=1e-6,
        atol=1e-6,
    )


def test_preconditioned_spring_solve_satisfies_linear_system():
    scaled_jacobian = jnp.asarray(
        [
            [1.0, -0.2],
            [0.3, 0.7],
            [-0.5, 0.4],
        ]
    )
    centered_energies = jnp.asarray([0.6, -0.2, -0.4])
    flat_mu_previous = jnp.asarray([0.1, -0.3])
    inverse_metric = jnp.asarray(
        [
            [0.4, -0.1],
            [-0.1, 0.7],
        ]
    )
    sketch_damping = 0.2

    result = solve_preconditioned_spring_system(
        scaled_jacobian,
        centered_energies,
        flat_mu_previous,
        lambda vector: inverse_metric @ vector,
        sketch_damping,
    )

    expected_kernel = scaled_jacobian @ inverse_metric @ scaled_jacobian.T
    expected_residual = centered_energies / jnp.sqrt(3.0)
    expected_residual = expected_residual - scaled_jacobian @ flat_mu_previous
    expected_direction = inverse_metric @ scaled_jacobian.T @ result.centered_zeta

    np.testing.assert_allclose(
        result.preconditioned_kernel, expected_kernel, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        result.regularized_kernel @ result.zeta,
        expected_residual,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(result.residual, expected_residual, atol=1e-6)
    np.testing.assert_allclose(result.flat_direction, expected_direction, atol=1e-6)
    np.testing.assert_allclose(jnp.mean(result.centered_zeta), 0.0, atol=1e-6)


def test_spring_nystrom_update_is_finite_and_keeps_fixed_sketch():
    nchains = 4
    model, params, data, key = _make_test_inputs(nchains)
    local_energy_fn = qho.make_harmonic_oscillator_local_energy(1.0, model.apply)
    energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
        local_energy_fn, nchains, None, True
    )
    config = _make_config()
    sampler_key = key
    expected_omega = initialize_nystrom_state(
        params, jax.random.PRNGKey(config.nystrom_seed), config.nystrom_rank
    ).omega

    update_param_fn, optimizer_state, key = initialize_spring_nystrom(
        model.apply,
        energy_and_statistics_fn,
        params,
        get_position_from_data,
        get_update_data_fn(model.apply),
        key,
        lambda _: config.learning_rate,
        config,
        apply_pmap=False,
    )
    initial_omega = optimizer_state.nystrom_state.omega

    np.testing.assert_array_equal(key, sampler_key)
    np.testing.assert_allclose(initial_omega, expected_omega)

    params, _, optimizer_state, metrics, key = update_param_fn(
        params, data, optimizer_state, key
    )

    assert jnp.isfinite(metrics["energy"])
    assert jnp.isfinite(metrics["variance"])
    assert jnp.isfinite(metrics["nystrom_spring_kernel_trace"])
    assert jnp.isfinite(metrics["nystrom_spring_metric_scale"])
    assert jnp.isfinite(metrics["nystrom_spring_trace"])
    assert metrics["nystrom_spring_metric_scale"] == 1.0
    assert metrics["nystrom_spring_phasein"] == 0.0
    np.testing.assert_allclose(optimizer_state.nystrom_state.omega, initial_omega)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(params))
