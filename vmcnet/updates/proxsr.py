"""Stochastic reconfiguration (SR) routine."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from vmcnet.utils.typing import Array, ModelApply, P, Tuple

import chex
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet import utils


def get_proxsr_update_fn(
    log_psi_apply: ModelApply[P],
    solve_type: str = "pos",
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

        # residual_solution is Ohat.T @ x where T @ x = epsilon_diff
        prev_grad_decayed = prev_grad * prev_grad_decay
        epsilon_prev = Ohat @ prev_grad_decayed
        epsilon_diff = centered_energies - epsilon_prev

        T = Ohat @ Ohat.T
        if solve_type == "pos":
            x = solveT_pos(T, damping, nchains, epsilon_diff)
        elif solve_type == "sym":
            x = solveT_sym(T, damping, nchains, epsilon_diff)
        elif solve_type == "eigh":
            x = solveT_eigh(T, damping, nchains, epsilon_diff)
        else:
            raise ValueError(
                f"solve_type must be pos, sym, or eigh. Received {solve_type}."
            )

        residual_solution = Ohat.T @ x

        SR_G = residual_solution + prev_grad_decayed
        return unravel_fn(SR_G)

    return proxsr_update_fn


def solveT_pos(T, damping, nchains, epsilon_diff):
    return jax.scipy.linalg.solve(
        T + jnp.eye(nchains) * damping * nchains, epsilon_diff, assume_a="pos"
    )


def solveT_sym(T, damping, nchains, epsilon_diff):
    return jax.scipy.linalg.solve(
        T + jnp.eye(nchains) * damping * nchains, epsilon_diff, assume_a="sym"
    )


def solveT_eigh(T, damping, nchains, epsilon_diff):
    eigval, eigvec = jnp.linalg.eigh(T)

    eigval += damping * nchains
    eigval_inv = 1 / eigval

    Tinv = eigvec @ jnp.diag(eigval_inv) @ eigvec.T
    return Tinv @ epsilon_diff


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
