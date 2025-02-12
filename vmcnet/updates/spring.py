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
        ones = jnp.ones((nchains, 1))

        # Calculate T = Ohat @ Ohat^T using neural-tangents
        # Some GPUs, particularly A100s and A5000s, can exhibit large numerical errors in these
        # calculations. As a result, we explicitly symmetrize T and, rather than using a Cholesky
        # solver to solve against T, we calculate its eigendecomposition and explicitly fix any negative
        # eigenvalues. We then use the fixed and regularized igendecomposition to solve against T.
        # This appears to be more stable than Cholesky in practice.
        T = kernel_fn(positions, positions, "ntk", params) / nchains
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = T + ones @ ones.T / nchains
        T = (T + T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        O_prev = jax.jvp(
            log_psi_apply,
            (params, positions),
            (mu_prev, jnp.zeros_like(positions)),
        )[1] / jnp.sqrt(nchains)
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsilon_tilde = epsilon_bar - Ohat_prev

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_tilde
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta_residual = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]

        return jax.tree_map(
            lambda dt, mup: dt / jnp.sqrt(nchains) + mup, dtheta_residual, mu_prev
        )

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
