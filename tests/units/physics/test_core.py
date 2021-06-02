"""Testing energy calculations."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics


def _make_dummy_log_f():
    f = lambda unused_params, x: jnp.sum(jnp.square(x) + 3 * x)
    log_f = lambda unused_params, x: jnp.log(jnp.abs(f(unused_params, x)))
    return f, log_f


def _make_dummy_x():
    return jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_laplacian_psi_over_psi():
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = _make_dummy_log_f()
    x = _make_dummy_x()

    grad_log_f = jax.grad(log_f, argnums=1)

    local_laplacian = physics.core.laplacian_psi_over_psi(grad_log_f, None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles. We then divide by f(x) to get (nabla^2 f) / f
    np.testing.assert_allclose(local_laplacian, 6 * 2 / f(None, x), rtol=1e-6)


def test_total_energy_grad():
    """Test the custom gradient of the total energy."""
    # since log|psi(x)| = a * sum(x^2), we have
    # log_psi_grad(x) = (grad_a psi / psi)(x) = sum(x^2)
    log_psi_apply = lambda a, x: a * jnp.sum(jnp.square(x), axis=-1)
    a = 3.5
    x = _make_dummy_x()
    log_psi_grad_x = jnp.array([5.0, 25.0, 61.0])
    nchains = x.shape[0]

    target_local_energies = jnp.array([3.0, 1.0, -1.0])
    target_energy = 1.0
    target_variance = 4.0
    target_grad_energy = 2.0 * jnp.sum(
        (target_local_energies - target_energy) * log_psi_grad_x
    )

    def local_energy_fn(a, x):
        del a
        return target_local_energies

    total_energy_value_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    energy_data, grad_energy = total_energy_value_and_grad(a, x)
    energy = energy_data[0]
    variance = energy_data[1][0]
    local_energies = energy_data[1][1]

    np.testing.assert_allclose(local_energies, target_local_energies)
    np.testing.assert_allclose(energy, target_energy)
    np.testing.assert_allclose(variance, target_variance)
    np.testing.assert_allclose(grad_energy, target_grad_energy)
