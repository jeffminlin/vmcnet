"""Test kinetic energy functions."""

import numpy as np
import jax

import vmcnet.physics as physics

from tests.test_utils import make_dummy_log_f, make_dummy_x


def test_laplacian_psi_over_psi():
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = make_dummy_log_f()
    x = make_dummy_x()

    grad_log_f = jax.grad(log_f, argnums=1)

    local_laplacian = physics.kinetic.laplacian_psi_over_psi(grad_log_f, None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles. We then divide by f(x) to get (nabla^2 f) / f
    np.testing.assert_allclose(local_laplacian, 6 * 2 / f(None, x), rtol=1e-6)


def test_kinetic_energy_shape():
    """Check that the vmapped shape is correct."""
    _, log_f = make_dummy_log_f()
    x = make_dummy_x()

    kinetic_energy_fn = physics.kinetic.create_laplacian_kinetic_energy(log_f)
    kinetic_energy_fn = jax.vmap(kinetic_energy_fn, in_axes=(None, 0), out_axes=0)
    kinetic_energies = kinetic_energy_fn(None, x)

    assert kinetic_energies.shape == (x.shape[0],)
