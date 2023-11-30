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
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    momentum: chex.Scalar = 0.0,
):
    """
    Get the ProxSR update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping (float): damping parameter
        mu (float): ProxSR-specific regularization

    Returns:
        Callable: ProxSR update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def proxsr_update_fn(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]

        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad

        log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(
            nchains
        )
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)

        T = Ohat @ Ohat.T
        ones = jnp.ones((nchains, 1))
        T_reg = T + ones @ ones.T + damping * jnp.eye(nchains)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed

        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(
            T_reg, epsion_tilde, assume_a="pos"
        )

        SR_G = dtheta_residual + prev_grad_decayed
        SR_G = (1 - momentum) * SR_G + momentum * prev_grad

        return unravel_fn(SR_G)

    return proxsr_update_fn


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
