"""Testing energy calculations."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics


def _make_dummy_log_f():
    f = lambda _, x: jnp.sum(jnp.square(x) + 3 * x)
    log_f = lambda _, x: jnp.log(jnp.abs(f(_, x)))
    return f, log_f


def _make_dummy_x():
    return jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_laplacian_psi_over_psi():
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = _make_dummy_log_f()
    x = _make_dummy_x()

    grad_log_f = jax.grad(log_f, argnums=1)

    local_laplacian = physics.energy.laplacian_psi_over_psi(grad_log_f, None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles. We then divide by f(x) to get (nabla^2 f) / f
    np.testing.assert_allclose(local_laplacian, 6 * 2 / f(None, x), rtol=1e-6)


def test_make_harmonic_oscillator_local_energy_with_zero_omega():
    """Test the creation of the harmonic oscillator local energy with omega = 0."""
    f, log_f = _make_dummy_log_f()
    vmapped_f = jax.vmap(f, in_axes=(None, 0), out_axes=0)
    multichain_x = jnp.reshape(jnp.arange(12, dtype=jnp.float32), (4, 3, 1))

    local_energy_fn = physics.energy.make_harmonic_oscillator_local_energy(0.0, log_f)

    kinetic = local_energy_fn(None, multichain_x)  # kinetic only because omega = 0

    # expect -(1/2) (nabla^2 f) / f, so because d^2f/dx_i^2 = 2 for all i, for 3
    # particles per sample we expect each sample x to have kinetic energy
    # -(1/2) * 3 * 2 / f(x), or -3 / f(x)
    expected = jnp.expand_dims(-3.0 / vmapped_f(None, multichain_x), axis=-1)

    np.testing.assert_allclose(kinetic, expected, rtol=1e-6)


def test_make_harmonic_oscillator_local_energy_with_nonzero_omega():
    """Test the creation of the harmonic oscillator local energy with omega = 1.0."""
    f, log_f = _make_dummy_log_f()
    vmapped_f = jax.vmap(f, in_axes=(None, 0), out_axes=0)
    multichain_x = jnp.reshape(jnp.arange(12, dtype=jnp.float32), (4, 3, 1))

    omega = 2.0

    local_energy_fn = physics.energy.make_harmonic_oscillator_local_energy(omega, log_f)

    local_energy = local_energy_fn(None, multichain_x)

    # expect -(1/2) (nabla^2 f) / f, so because d^2f/dx_i^2 = 2 for all i, for 3
    # particles per sample we expect each sample x to have kinetic energy
    # -(1/2) * 3 * 2 / f(x), or -3 / f(x)
    kinetic = jnp.expand_dims(-3.0 / vmapped_f(None, multichain_x), axis=-1)
    potential = 0.5 * jnp.sum(jnp.square(omega * multichain_x), axis=-2)

    np.testing.assert_allclose(local_energy, kinetic + potential, rtol=1e-6)
