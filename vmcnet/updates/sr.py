"""Stochastic reconfiguration (SR) routine."""
from enum import Enum, auto

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp

from vmcnet.utils.typing import Array, ModelApply, P


class SRMode(Enum):
    """Modes for computing the preconditioning by the Fisher inverse during SR.

    If LAZY, then uses composed jvp and vjp calls to lazily compute the various
    Jacobian-vector products. This is more computationally and memory-efficient.
    If DEBUG, then directly computes the Jacobian (per-example gradients) and
    uses jnp.matmul to compute the Jacobian-vector products. Defaults to LAZY.
    """

    LAZY = auto()
    DEBUG = auto()


def get_fisher_inverse_fn(
    log_psi_apply: ModelApply[P], damping: chex.Scalar, momentum: chex.Scalar
):
    """Get a Fisher-preconditioned update.

    Given a gradient update grad_E, the function returned here approximates

        (0.25 * F + damping * I)^{-1} * grad_E,

    where F is the Fisher information matrix. The inversion is approximated via the
    conjugate gradient algorithm (possibly truncated to a finite number of iterations).

    This preconditioned gradient update, when used as-is, is also known as the
    stochastic reconfiguration algorithm. See https://arxiv.org/pdf/1909.02487.pdf,
    Appendix C for the connection between natural gradient descent and stochastic
    reconfiguration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|

    Returns:
        Callable: function which computes the gradient preconditioned with the inverse
        of the Fisher information matrix. Has the signature
            (energy_grad, params, positions) -> preconditioned_grad
    """
    # TODO(Jeffmin): explore preconditioners for speeding up convergence and to provide
    # more stability
    # TODO(Jeffmin): investigate damping scheduling and possibly adaptive damping

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def precondition_grad_with_fisher(
        centered_energies: P, params: P, prev_grad, positions: Array
    ) -> P:
        nchains = positions.shape[0]
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad *= momentum

        # (nsample, nparam)
        log_psi_grads = batch_raveled_log_psi_grad(params, positions)
        centered_log_psi_grads = log_psi_grads - jnp.mean(
            log_psi_grads, axis=0, keepdims=True
        )

        e_tilde = centered_energies - centered_log_psi_grads @ prev_grad

        T = centered_log_psi_grads @ centered_log_psi_grads.T
        eigval, eigvec = jnp.linalg.eigh(T)

        # should be nonnegative since it's PSDF matrix
        eigval = jnp.where(eigval < 0, 0, eigval)
        # Damping has scale factor of nchains since we didn't divide T
        eigval += damping * nchains
        eigval_inv = 1 / eigval

        Tinv = eigvec @ jnp.diag(eigval_inv) @ eigvec.T

        SR_G_tilde = centered_log_psi_grads.T @ Tinv @ e_tilde
        SR_G = SR_G_tilde + prev_grad

        return unravel_fn(SR_G)

    return precondition_grad_with_fisher
