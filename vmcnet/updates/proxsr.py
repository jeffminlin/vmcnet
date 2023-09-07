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
    """
    Get the ProxSR update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping_type (str): either "diag_shift" or "pinv"
        damping (float): damping parameter
        prev_grad_decay (float): ProxSR-specific parameter

    Returns:
        Callable: ProxSR update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

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

    def proxsr_update_fn(
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

        SR_G = min_sr_solution + prev_grad_complement * prev_grad_decay
        return unravel_fn(SR_G)

    return proxsr_update_fn


def constrain_norm(
    grad: P,
    learning_rate: chex.Numeric,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_precond_grads = tree_inner_product(grad, grad)
    sq_norm_scaled_grads = sq_norm_precond_grads * learning_rate**2

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
