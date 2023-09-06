"""Stochastic reconfiguration (SR) routine."""
import jax
import jax.flatten_util
import jax.numpy as jnp

from vmcnet.utils.typing import Array, ModelApply, P, Tuple

import chex
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet import utils


def get_proxsr_update_fn(
    log_psi_apply: ModelApply[P],
    damping_type: str = "diag_shift",
    damping: chex.Scalar = 0.001,
    prev_grad_decay: chex.Scalar = 0.99,
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

    def invert_T(T, nchains):
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
        return Tinv

    def precondition_grad_with_fisher(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)

        # (nsample, nparam)
        log_psi_grads = batch_raveled_log_psi_grad(params, positions)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)

        T = Ohat @ Ohat.T
        Tinv = invert_T(T, nchains)

        OhatT_Tinv = Ohat.T @ Tinv

        min_sr_solution = OhatT_Tinv @ centered_energies

        Ohat_prev_grad = Ohat @ prev_grad
        prev_grad_subspace = OhatT_Tinv @ Ohat_prev_grad
        prev_grad_complement = prev_grad - prev_grad_subspace

        # The update consists of a linear combination of 4 components. The first
        # component is min_sr_solution, which is the update used by the original
        # MinSR method. The remaining components come from a simple decomposition of the
        # previous gradient into several mutually orthogonal components.
        #
        # First, prev_grad is decomposed as prev_grad = prev_grad_subspace +
        # prev_grad_complement, where prev_grad_subspace lies within the span of the
        # current gradients and prev_grad_complement lies in the orthogonal space.
        # Second, prev_grad_subspace is decomposed as prev_grad_subspace =
        # prev_grad_parallel + prev_grad_orthogonal, where prev_grad_parallel is a
        # multiple of min_sr_solution and prev_grad_orthogonal is orthogonal to
        # min_sr_solution.
        #
        # Finally, the update is taken as a linear combination of min_sr_solution,
        # prev_grad_complement, prev_grad_parallel, and prev_grad_orthogonal.
        SR_G = min_sr_solution + prev_grad_complement * prev_grad_decay

        # This vector is returned to facilitate a "natural" norm constraint, since the
        # norm of this vector, i.e. Ohat_G.T @ Ohat_G, gives the distance of the update
        # step w.r.t to the Fisher information metric.
        Ohat_G = Ohat @ SR_G / jnp.sqrt(nchains)

        return Ohat_G, unravel_fn(SR_G)

    return precondition_grad_with_fisher


def constrain_norm(
    grad: P,
    Ohat_grad: P,
    learning_rate: chex.Numeric,
    norm_constraint: chex.Numeric = 0.001,
    norm_type: str = "grad",
) -> P:
    """Constrains the preconditioned norm of the update, adapted from KFAC."""
    if norm_type == "natural":
        sq_norm_precond_grads = tree_inner_product(Ohat_grad, Ohat_grad)
    elif norm_type == "euclidean":
        sq_norm_precond_grads = tree_inner_product(grad, grad)
    else:
        raise ValueError("Norm type should be either Ohat_grad or grad")

    sq_norm_scaled_grads = sq_norm_precond_grads * learning_rate**2

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)

    max_coefficient = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(max_coefficient, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
