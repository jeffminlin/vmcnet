"""Stochastic reconfiguration (SR) routine."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt

from vmcnet.utils.typing import Array, ModelApply, P, Tuple

import chex
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet import utils


def get_spring_update_fn(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    momentum: chex.Scalar = 0.0,  # TODO: remove
):
    """
    Get the SPRING update function.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        damping (float): damping parameter
        mu (float): SPRING-specific regularization

    Returns:
        Callable: SPRING update function. Has the signature
        (centered_energies, params, prev_grad, positions) -> new_grad
    """

    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def spring_update_fn(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        mu_prev = jax.tree_map(lambda x: mu * x, prev_grad)

        T = kernel_fn(positions, positions, "ntk", params)
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        ones = jnp.ones((nchains, 1))
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        O_prev = jax.jvp(
            log_psi_apply,
            (params, positions),
            (mu_prev, jnp.zeros_like(positions)),
        )[1]
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsion_tilde = epsilon_bar - Ohat_prev

        zeta = jax.scipy.linalg.solve(T_reg, epsion_tilde, assume_a="pos")
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta_residual = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]

        return jax.tree_map(lambda x, y: x + y, dtheta_residual, mu_prev)

    return spring_update_fn


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
