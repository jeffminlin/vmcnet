"""Test kinetic energy functions."""
import jax

import vmcnet.physics as physics

from tests.test_utils import make_dummy_log_f, make_dummy_x


def test_kinetic_energy_shape():
    """Check that the vmapped shape is correct."""
    _, log_f = make_dummy_log_f()
    x = make_dummy_x()

    kinetic_energy_fn = physics.kinetic.create_laplacian_kinetic_energy(log_f)
    kinetic_energy_fn = jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)
    kinetic_energies = kinetic_energy_fn(None, x)

    assert kinetic_energies.shape == (x.shape[0],)
