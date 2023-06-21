"""Stochastic reconfiguration (SR) routine."""
from enum import Enum, auto
from typing import Callable, Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jscp

from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.pytree_helpers import multiply_tree_by_scalar, tree_sum
from vmcnet.utils.typing import Array, ArrayLike, ModelApply, P


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
    mean_grad_fn: Callable[[ArrayLike], ArrayLike],
    damping: float = 1e-3,
    rcond: float = 1e-5,
    maxiter: Optional[int] = None,
    mode: SRMode = SRMode.LAZY,
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
        mean_grad_fn (Callable): function which is used to average the local gradient
            terms over all local devices. Has the signature local_grads -> avg_grad / 2,
            and should only average over the batch axis 0.
        damping (float, optional): multiple of the identity to add to the Fisher before
            inverting. Without this term, the approximation to the Fisher will always
            be less than full rank when nchains < nparams, and so CG will fail to
            converge. This should be tuned together with the learning rate. Defaults to
            0.001.
        maxiter (int, optional): maximum number of CG iterations to do when computing
            the inverse application of the Fisher. Defaults to None, which uses maxiter
            equal to 10 * number of params.
        mode (SRMode, optional): mode of computing the forward Fisher-vector products.
            If LAZY, then uses composed jvp and vjp calls to lazily compute the various
            Jacobian-vector products. This is more computationally and memory-efficient.
            If DEBUG, then directly computes the Jacobian (per-example gradients) and
            uses jnp.matmul to compute the Jacobian-vector products. Defaults to LAZY.

    Returns:
        Callable: function which computes the gradient preconditioned with the inverse
        of the Fisher information matrix. Has the signature
            (energy_grad, params, positions) -> preconditioned_grad
    """
    # TODO(Jeffmin): explore preconditioners for speeding up convergence and to provide
    # more stability
    # TODO(Jeffmin): investigate damping scheduling and possibly adaptive damping
    if mode == SRMode.DEBUG:

        def raveled_log_psi_grad(params: P, positions: Array) -> Array:
            log_grads = jax.grad(log_psi_apply)(params, positions)
            return jax.flatten_util.ravel_pytree(log_grads)[0]

        batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

        def precondition_grad_with_fisher(
            energy_grad: P, params: P, positions: Array
        ) -> P:
            G, unravel_fn = jax.flatten_util.ravel_pytree(energy_grad)

            # nparam, nsample, tall and skinny
            L = batch_raveled_log_psi_grad(params, positions).T
            nchains_local = L.shape[-1]
            L = (L - jnp.mean(L, axis=-1, keepdims=True)) / jnp.sqrt(nchains_local)

            S = L.T @ L
            damped_F = S + damping * jnp.eye(nchains_local, nchains_local)

            b = jnp.linalg.solve(damped_F, L.T @ G)
            G_preconditioned = (1 / damping) * (G - L @ b)
            G_preconditioned = pmean_if_pmap(G_preconditioned)

            return unravel_fn(G_preconditioned)

    elif mode == SRMode.LAZY:

        def precondition_grad_with_fisher(
            energy_grad: P, params: P, positions: Array
        ) -> P:
            def partial_log_psi_apply(params: P) -> ArrayLike:
                return log_psi_apply(params, positions)

            _, vjp_fn = jax.vjp(partial_log_psi_apply, params)

            def fisher_apply(x: Array) -> Array:
                # x is a pytree with same structure as params
                nchains_local = positions.shape[0]
                _, jacobian_vector_prod = jax.jvp(
                    partial_log_psi_apply, (params,), (x,)
                )
                mean_jacobian_vector_prod = mean_grad_fn(jacobian_vector_prod)
                centered_jacobian_vector_prod = (
                    jacobian_vector_prod - mean_jacobian_vector_prod
                )
                local_device_fisher_times_x = multiply_tree_by_scalar(
                    vjp_fn(centered_jacobian_vector_prod)[0], 1.0 / nchains_local
                )
                fisher_times_x = pmean_if_pmap(local_device_fisher_times_x)
                return tree_sum(fisher_times_x, multiply_tree_by_scalar(x, damping))

            sr_grad, _ = jscp.sparse.linalg.cg(
                fisher_apply,
                energy_grad,
                x0=energy_grad,
                maxiter=maxiter,
            )

            return sr_grad

    else:
        raise ValueError(
            "Requested Fisher apply mode not supported; only {} are supported, "
            "but {} was requested.".format(", ".join(SRMode.__members__.keys()), mode)
        )

    return precondition_grad_with_fisher
