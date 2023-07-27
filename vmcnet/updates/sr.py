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
    log_psi_apply: ModelApply[P],
    damping_type: str = "diag_shift",
    damping: chex.Scalar = 0.001,
    parallel_momentum: chex.Scalar = 0.9,
    complement_decay: chex.Scalar = 0.95,
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
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> P:
        nchains = positions.shape[0]
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)

        # (nsample, nparam)
        log_psi_grads = batch_raveled_log_psi_grad(params, positions)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)

        raw_G = 2 * centered_energies @ log_psi_grads / nchains

        T = Ohat @ Ohat.T
        eigval, eigvec = jnp.linalg.eigh(T)

        # should be nonnegative since it's PSDF matrix
        eigval = jnp.where(eigval < 0, 0, eigval)
        # Damping has scale factor of nchains since we didn't divide T

        if damping_type == "diag_shift":
            eigval += damping * nchains
            eigval_inv = 1 / eigval
        elif damping_type == "pinv":
            eigval_inv = jnp.where(eigval >= damping * nchains, 1 / eigval, 0.0)
        else:
            raise ValueError("Damping type must be either diag_shift or pinv")

        Tinv = eigvec @ jnp.diag(eigval_inv) @ eigvec.T

        # prev_grad *= decay
        P = Ohat.T @ Tinv

        min_sr_solution = P @ centered_energies

        Ohat_prev_grad = Ohat @ prev_grad
        prev_grad_subspace = P @ Ohat_prev_grad
        prev_grad_complement = prev_grad - prev_grad_subspace

        prev_grad_subspace_parallel = (
            min_sr_solution
            * (min_sr_solution @ prev_grad_subspace)
            / (min_sr_solution @ min_sr_solution)
        )

        subspace_direction_with_momentum = (
            prev_grad_subspace_parallel * parallel_momentum
            + min_sr_solution * (1 - parallel_momentum)
        )
        SR_G = (
            subspace_direction_with_momentum + prev_grad_complement * complement_decay
        )

        return unravel_fn(raw_G), unravel_fn(SR_G)

    return precondition_grad_with_fisher
