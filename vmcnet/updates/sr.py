"""Stochastic reconfiguration (SR) routine."""
from enum import Enum, auto
from typing import Callable, Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jscp
import chex

from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import Array, ArrayLike, ModelApply, P

from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_sum,
)
from vmcnet import utils


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
    damping: float = 0.001,
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
            raveled_energy_grad, unravel_fn = jax.flatten_util.ravel_pytree(energy_grad)

            log_psi_grads = batch_raveled_log_psi_grad(params, positions)
            mean_log_psi_grads = mean_grad_fn(log_psi_grads)
            centered_log_psi_grads = (
                log_psi_grads - mean_log_psi_grads
            )  # shape (nchains, nparams)

            def fisher_apply(x: Array) -> Array:
                # x is shape (nparams,)
                nchains_local = centered_log_psi_grads.shape[0]
                centered_jacobian_vector_prod = jnp.matmul(centered_log_psi_grads, x)
                local_fisher_times_x = (
                    jnp.matmul(
                        jnp.transpose(centered_log_psi_grads),
                        centered_jacobian_vector_prod,
                    )
                    / nchains_local
                )
                fisher_times_x = pmean_if_pmap(local_fisher_times_x)
                return fisher_times_x + damping * x

            sr_grad, _ = jscp.sparse.linalg.cg(
                fisher_apply,
                raveled_energy_grad,
                x0=raveled_energy_grad,
                maxiter=maxiter,
            )

            return unravel_fn(sr_grad)

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


def constrain_norm(
    grads: P,
    preconditioned_grads: P,
    learning_rate: chex.Numeric,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Constrains the preconditioned norm of the update, adapted from KFAC."""
    sq_norm_grads = tree_inner_product(preconditioned_grads, grads)
    sq_norm_scaled_grads = sq_norm_grads * learning_rate**2

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(preconditioned_grads, coefficient)

    return constrained_grads
