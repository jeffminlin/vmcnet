"""Integration tests for the quantum harmonic oscillator."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models
import vmcnet.physics as physics


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
    return log_psi_model, params, random_particle_positions, key


def test_five_particle_ground_state_harmonic_oscillator():
    """Test five non-interacting harmonic oscillators with two spins."""
    omega = 2.0
    (
        log_psi_model,
        params,
        random_particle_positions,
        _,
    ) = _make_initial_positions_and_model(omega, 4)

    local_energy_fn = physics.energy.make_harmonic_oscillator_local_energy(
        omega, log_psi_model.apply
    )
    local_energies = local_energy_fn(params, random_particle_positions)

    np.testing.assert_allclose(
        local_energies,
        omega * (0.5 + 1.5 + 0.5 + 1.5 + 2.5) * jnp.ones((4, 1)),
        rtol=1e-5,
    )
