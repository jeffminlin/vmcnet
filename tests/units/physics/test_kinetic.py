"""Test kinetic energy functions."""

import numpy as np
import jax

import vmcnet.physics as physics

from tests.test_utils import make_dummy_log_f, make_dummy_x


def test_kinetic_energy():
    """Check that the vmapped shape is correct."""
    f, log_f = make_dummy_log_f()
    x = make_dummy_x()

    kinetic_energy_fn = physics.kinetic.create_laplacian_kinetic_energy(log_f)
    kinetic_energy_fn = jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)
    kinetic_energies = kinetic_energy_fn(None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles (2). We then divide by f(x) and multiply by -0.5 to get -0.5 (nabla^2 f) / f
    np.testing.assert_allclose(
        kinetic_energies, -2 / jax.vmap(f, in_axes=(None, 0))(None, x), rtol=1e-6
    )
    assert kinetic_energies.shape == (x.shape[0],)
