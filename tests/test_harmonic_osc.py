"""Integration tests for the quantum harmonic oscillator."""
import logging

import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc as mcmc
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils


def _make_initial_positions_and_model(model_omega, nchains):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    random_particle_positions = jax.random.normal(subkey, shape=(nchains, 5, 1))

    split_spin_fn = lambda x: jnp.split(x, [2], axis=-2)  # (2, 3) particle split
    orbitals = models.harmonic_osc.HarmonicOscillatorOrbitals(model_omega)
    logdet_fn = models.antisymmetry.logdet_product

    log_psi_model = models.construct.ComposedModel([split_spin_fn, orbitals, logdet_fn])

    key, subkey = jax.random.split(key)
    params = log_psi_model.init(key, random_particle_positions)
    amplitudes = log_psi_model.apply(params, random_particle_positions)
    return log_psi_model, params, random_particle_positions, amplitudes, key


def test_five_particle_ground_state_harmonic_oscillator():
    """Test five non-interacting harmonic oscillators with two spins."""
    omega = 2.0
    (
        log_psi_model,
        params,
        random_particle_positions,
        _,
        _,
    ) = _make_initial_positions_and_model(omega, 4)

    local_energy_fn = physics.energy.make_harmonic_oscillator_local_energy(
        omega, log_psi_model.apply
    )
    local_energies = local_energy_fn(params, random_particle_positions)

    np.testing.assert_allclose(
        local_energies,
        omega * (0.5 + 1.5 + 0.5 + 1.5 + 2.5) * jnp.ones(4),
        rtol=1e-5,
    )


def test_harmonic_oscillator_vmc(caplog):
    """Test that the trainable omega converges to the true spring constant."""
    model_omega = 2.5
    spring_constant = 1.5

    nchains = 100 * jax.device_count()
    nburn = 100
    nepochs = 50
    nsteps_per_param_update = 5
    std_move = 0.25
    learning_rate = 1e-4

    (
        log_psi_model,
        params,
        random_particle_positions,
        amplitudes,
        key,
    ) = _make_initial_positions_and_model(model_omega, nchains)

    local_energy_fn = physics.energy.make_harmonic_oscillator_local_energy(
        spring_constant, log_psi_model.apply
    )

    def optimizer_apply(grad, params, state):
        return jax.tree_map(lambda a, b: a - learning_rate * b, params, grad), state

    proposal_fn = mcmc.metropolis.make_position_and_amplitude_gaussian_proposal(
        log_psi_model.apply, std_move
    )
    acceptance_fn = (
        mcmc.metropolis.make_position_and_amplitude_metropolis_symmetric_acceptance()
    )
    update_data_fn = updates.data.update_position_and_amplitude

    metrop_step_fn = mcmc.metropolis.make_metropolis_step(
        proposal_fn, acceptance_fn, update_data_fn
    )
    update_param_fn = updates.params.create_position_amplitude_data_update_param_fn(
        log_psi_model.apply, local_energy_fn, nchains, optimizer_apply
    )

    (
        random_particle_positions,
        params,
        key,
    ) = utils.distribute.distribute_data_params_and_key(
        random_particle_positions, params, key
    )
    amplitudes = utils.distribute.distribute_data(amplitudes)
    data = updates.data.PositionAmplitudeData(random_particle_positions, amplitudes)

    with caplog.at_level(logging.INFO):
        params, _, _ = train.vmc.vmc_loop(
            params,
            None,
            data,
            nchains,
            nburn,
            nepochs,
            nsteps_per_param_update,
            metrop_step_fn,
            update_param_fn,
            key,
        )

    np.testing.assert_allclose(
        jax.tree_leaves(params)[0], jnp.sqrt(spring_constant), rtol=1e-6
    )
