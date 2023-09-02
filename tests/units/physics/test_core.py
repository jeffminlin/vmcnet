"""Testing energy calculations."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics

from tests.test_utils import make_dummy_log_f, make_dummy_x


def test_laplacian_psi_over_psi():
    """Test (nabla^2 f)(x) / f(x) for f(x) = sum_i x_i^2 + 3x_i."""
    f, log_f = make_dummy_log_f()
    x = make_dummy_x()

    grad_log_f = jax.grad(log_f, argnums=1)

    local_laplacian = physics.core.laplacian_psi_over_psi(grad_log_f, None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles. We then divide by f(x) to get (nabla^2 f) / f
    np.testing.assert_allclose(local_laplacian, 6 * 2 / f(None, x), rtol=1e-6)


def test_total_energy_grad():
    """Test the custom gradient of the total energy."""

    def log_psi_apply(a, x):
        # since log|psi(x)| = a * sum(x^2), we have
        # log_psi_grad(x) = (grad_a psi / psi)(x) = sum(x^2)
        return a * jnp.sum(jnp.square(x), axis=-1)

    a = 3.5
    x = make_dummy_x()
    log_psi_grad_x = jnp.array([5.0, 25.0, 61.0])
    nchains = x.shape[0]
    key = jax.random.PRNGKey(0)

    def local_energy_fn(a, x, key):
        return jnp.sum(x)

    # Based on the specific values returned by make_dummy_x
    target_local_energies = jnp.array([3.0, 7.0, 11.0])
    target_energy = 7.0
    target_variance = 16.0
    target_grad_energy = (
        2.0
        * jnp.mean((target_local_energies - target_energy) * log_psi_grad_x)
        * (nchains / (nchains - 1))
    )

    total_energy_value_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    energy_data, grad_energy = total_energy_value_and_grad(a, key, x)
    energy = energy_data[0]
    variance = energy_data[1]["variance"]
    local_energies = energy_data[1]["local_energies_noclip"]

    np.testing.assert_allclose(local_energies, target_local_energies)
    np.testing.assert_allclose(energy, target_energy)
    np.testing.assert_allclose(variance, target_variance)
    np.testing.assert_allclose(grad_energy, target_grad_energy, rtol=1e-6)
