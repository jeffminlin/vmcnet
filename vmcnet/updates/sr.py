"""Stochastic reconfiguration routines."""
from typing import Callable, Optional, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jscp

from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import ModelApply, P


def get_sr_energy_bwd(
    log_psi_apply: ModelApply[P],
    mean_grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
    damping: float = 0.001,
    maxiter: Optional[int] = None,
):
    """Get a Fisher-preconditioned backward pass of the total energy.

    Given the standard gradient formula

        grad_E = 2 * E_p[(local_e - E_p[local_e]) * grad_log_psi],

    the function returned here approximates (0.25 * F + damping * I)^{-1}*grad_E, where
    F is the Fisher information matrix. The inversion is approximated via the conjugate
    gradient algorithm (possibly truncated to a finite number of iterations).

    This gradient update, when used as-is, is also known as the stochastic
    reconfiguration algorithm.

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

    Returns:
        Callable: function which computes the backward pass in the custom vjp of the
        total energy. The gradient is preconditioned with the inverse of the Fisher
        information matrix. Has the signature (res, cotangents) -> (gradients, None)
    """

    def raveled_log_psi_grad(params, positions):
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def _get_raveled_energy_grad(raveled_batch_grads, centered_local_energies):
        return 2.0 * mean_grad_fn(
            centered_local_energies[..., None] * raveled_batch_grads
        )

    def energy_bwd(res, cotangents) -> Tuple[P, None]:
        energy, local_energies, params, positions = res
        centered_local_energies = local_energies - energy

        _, unravel_fn = jax.flatten_util.ravel_pytree(params)
        log_grads = batch_raveled_log_psi_grad(params, positions)

        mean_log_grads = mean_grad_fn(log_grads)
        centered_log_grads = log_grads - mean_log_grads

        def fisher_apply(x: jnp.ndarray) -> jnp.ndarray:
            nchains_local = centered_log_grads.shape[0]
            jvp = jnp.matmul(centered_log_grads, x)
            Fx = jnp.matmul(jnp.transpose(centered_log_grads), jvp) / nchains_local
            return pmean_if_pmap(Fx) + damping * x

        def fisher_diag_inverse_apply(x: jnp.ndarray) -> jnp.ndarray:
            diag_F = mean_grad_fn(jnp.square(centered_log_grads))
            return x / (diag_F + damping)

        partial_grad = _get_raveled_energy_grad(log_grads, centered_local_energies)
        full_grad = cotangents[0] * partial_grad

        sr_grad, _ = jscp.sparse.linalg.cg(
            fisher_apply,
            full_grad,
            x0=full_grad,
            maxiter=maxiter,
            M=fisher_diag_inverse_apply,
        )

        return unravel_fn(sr_grad), None

    return energy_bwd
