"""Stochastic reconfiguration (SR) routine."""
from enum import Enum, auto

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


def get_fisher_inverse_fn(log_psi_apply: ModelApply[P]):
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
        centered_energies: P, params: P, positions: Array
    ) -> P:
        example_grad = jax.grad(log_psi_apply)(params, positions[0, ...])
        _, unravel_fn = jax.flatten_util.ravel_pytree(example_grad)

        # (nsample, nparam)
        log_psi_grads = batch_raveled_log_psi_grad(params, positions)

        SR_G = jnp.linalg.lstsq(log_psi_grads, centered_energies)[0]

        return unravel_fn(SR_G)

    return precondition_grad_with_fisher
