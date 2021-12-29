"""Test kinetic energy functions."""
import vmcnet.physics as physics
import jax.numpy as jnp

from tests.test_utils import make_dummy_log_f, make_dummy_x #, make_dummy_antisymmetric


def test_kinetic_energy_shape():
    """Check that the vmapped shape is correct."""
    _, log_f = make_dummy_log_f()
    x = make_dummy_x()

    kinetic_energy_fn = physics.kinetic.create_continuous_kinetic_energy(log_f)
    kinetic_energies = kinetic_energy_fn(None, x)

    assert kinetic_energies.shape == (x.shape[0],)


def test_hubbard_kinetic_energy_shape():
    """Check that the vmapped shape is correct."""

    f,_ = make_dummy_log_f() #not in log domain
    x = jnp.array([[0,1,3]])
    side_length=4

    kinetic_energy_fn = physics.kinetic.create_hubbard_kinetic_energy(f,side_length)
    kinetic_energies = kinetic_energy_fn(None, x)

    assert kinetic_energies.shape == (x.shape[0],)


