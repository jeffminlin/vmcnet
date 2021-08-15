"""Test stochastic reconfiguration routines."""
import functools

import jax.numpy as jnp
import numpy as np

import vmcnet.updates as updates


def _setup_fisher():
    energy_grad = jnp.array([0.5, -0.5, 1.2])
    params = jnp.array([1.0, 2.0, 3.0])
    positions = jnp.array(
        [
            [1.0, -0.1, 0.1],
            [-0.1, 1.0, 0.0],
            [0.1, 0.0, 1.0],
            [0.01, -0.01, 0.0],
            [0.0, 0.0, -0.02],
        ]
    )
    nchains = len(positions)

    jacobian = positions
    centered_jacobian = jacobian - jnp.mean(jacobian, axis=0)
    centered_JT_J = jnp.matmul(jnp.transpose(centered_jacobian), centered_jacobian)
    fisher = centered_JT_J / nchains  # technically 0.25 * Fisher

    def log_psi_apply(params, positions):
        return jnp.matmul(positions, params)

    mean_grad_fn = functools.partial(jnp.mean, axis=0)
    return energy_grad, params, positions, fisher, log_psi_apply, mean_grad_fn


def test_fisher_inverse_matches_exact_solve():
    """Check that the fisher inverse fn in lazy mode produces the solution to Fx = b."""
    (
        energy_grad,
        params,
        positions,
        fisher,
        log_psi_apply,
        mean_grad_fn,
    ) = _setup_fisher()

    fisher_inverse_fn = updates.sr.get_fisher_inverse_fn(
        log_psi_apply, mean_grad_fn, damping=0.0, maxiter=None
    )

    Finverse_grad = fisher_inverse_fn(energy_grad, params, positions)

    np.testing.assert_allclose(
        jnp.matmul(fisher, Finverse_grad), energy_grad, atol=1e-6
    )


def test_lazy_and_debug_fisher_inverse_match():
    """Check that fisher inverse fn in both lazy and debug mode give the same result."""
    (
        energy_grad,
        params,
        positions,
        _,
        log_psi_apply,
        mean_grad_fn,
    ) = _setup_fisher()

    fisher_inverse_fn_lazy = updates.sr.get_fisher_inverse_fn(
        log_psi_apply,
        mean_grad_fn,
        damping=0.0,
        maxiter=None,
        mode=updates.sr.SRMode.LAZY,
    )
    fisher_inverse_fn_debug = updates.sr.get_fisher_inverse_fn(
        log_psi_apply,
        mean_grad_fn,
        damping=0.0,
        maxiter=None,
        mode=updates.sr.SRMode.DEBUG,
    )

    lazy_Finverse_grad = fisher_inverse_fn_lazy(energy_grad, params, positions)
    debug_Finverse_grad = fisher_inverse_fn_debug(energy_grad, params, positions)

    np.testing.assert_allclose(
        lazy_Finverse_grad, debug_Finverse_grad, rtol=1e-6, atol=1e-6
    )
