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

    local_laplacian = physics.energy.laplacian_psi_over_psi(grad_log_f, None, x)

    # d^2f/dx_i^2 = 2 for all i, so the Laplacian is simply 2 * n, where n is the number
    # of particles. We then divide by f(x) to get (nabla^2 f) / f
    np.testing.assert_allclose(local_laplacian, 6 * 2 / f(None, x), rtol=1e-6)
