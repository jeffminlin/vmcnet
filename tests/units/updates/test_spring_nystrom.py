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
    get_phasein_coefficient,
    initialize_spring_nystrom,
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
            metric_shift_strategy="constant",
            metric_identity_shift=1.0,
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


def test_spring_nystrom_update_is_finite_and_keeps_fixed_sketch():
    nchains = 4
    model, params, data, key = _make_test_inputs(nchains)
    local_energy_fn = qho.make_harmonic_oscillator_local_energy(1.0, model.apply)
    energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
        local_energy_fn, nchains, None, True
    )
    config = _make_config()

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

    params, _, optimizer_state, metrics, key = update_param_fn(
        params, data, optimizer_state, key
    )

    assert jnp.isfinite(metrics["energy"])
    assert jnp.isfinite(metrics["variance"])
    assert jnp.isfinite(metrics["nystrom_spring_kernel_trace"])
    assert jnp.isfinite(metrics["nystrom_spring_trace"])
    assert metrics["nystrom_spring_phasein"] == 0.0
    np.testing.assert_allclose(optimizer_state.nystrom_state.omega, initial_omega)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(params))
