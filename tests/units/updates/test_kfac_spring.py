"""Tests for the explicit KFAC-SPRING prototype."""

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
from vmcnet.updates.kfac_spring import (
    get_centered_jacobian,
    initialize_kfac_spring,
)


class TinyTrial(models.core.Module):
    """Small KFAC-tagged model for KFAC-SPRING tests."""

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


def _make_config(mu=0.0):
    return ConfigDict(
        dict(
            schedule_type="constant",
            learning_rate=0.02,
            learning_decay_rate=0.0,
            mu=mu,
            sketch_damping=1e-3,
            constrain_norm=False,
            norm_constraint=1e-3,
            kfac_l2_reg=0.0,
            kfac_norm_constraint=None,
            kfac_curvature_ema=0.0,
            kfac_min_damping=1e-4,
            kfac_register_only_generic=False,
            kfac_estimation_mode="fisher_exact",
            kfac_inverse_damping=1e-3,
            exact_power=True,
            use_cached=False,
        )
    )


def test_get_centered_jacobian_has_zero_column_means():
    model, params, data, _ = _make_test_inputs()

    jacobian, _ = get_centered_jacobian(
        model.apply, params, get_position_from_data(data)
    )

    np.testing.assert_allclose(jnp.mean(jacobian, axis=0), 0.0, atol=1e-6)


def test_kfac_minsr_update_is_finite():
    nchains = 4
    model, params, data, key = _make_test_inputs(nchains)
    local_energy_fn = qho.make_harmonic_oscillator_local_energy(1.0, model.apply)
    energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
        local_energy_fn, nchains, None, True
    )
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        model.apply, local_energy_fn, nchains, None, nan_safe=True
    )
    config = _make_config(mu=0.0)

    update_param_fn, optimizer_state, key = initialize_kfac_spring(
        model.apply,
        energy_and_statistics_fn,
        energy_data_val_and_grad,
        params,
        data,
        get_position_from_data,
        get_update_data_fn(model.apply),
        key,
        lambda _: config.learning_rate,
        config,
        apply_pmap=False,
    )

    params, _, _, metrics, _ = update_param_fn(params, data, optimizer_state, key)

    assert jnp.isfinite(metrics["energy"])
    assert jnp.isfinite(metrics["variance"])
    assert jnp.isfinite(metrics["kfac_spring_kernel_trace"])
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(params))
