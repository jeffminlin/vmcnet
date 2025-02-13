"""Stochastic reconfiguration (SR) routine."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule
import vmcnet.physics as physics

from .update_param_fns import UpdateParamFn, construct_default_update_param_fn
from .optax_utils import initialize_optax_optimizer


def initialize_spring(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(regular_grad, params, optimizer_state, data, aux):
        del regular_grad
        grad = spring_step(
            aux["centered_local_energies"],
            params,
            prev_update(optimizer_state),
            get_position_fn(data),
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = construct_default_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def get_spring_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
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

    def spring_step(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        mu_prev = jax.tree_map(lambda x: mu * x, prev_grad)
        ones = jnp.ones((nchains, 1))

        # Calculate T = Ohat @ Ohat^T using neural-tangents
        # Some GPUs, particularly A100s and A5000s, can exhibit large numerical
        # errors in these calculations. As a result, we explicitly symmetrize T
        # and, rather than using a Cholesky solver to solve against T, we
        # calculate its eigendecomposition and explicitly fix any negative
        # eigenvalues. We then use the fixed and regularized igendecomposition
        # to solve against T. This appears to be more stable than Cholesky
        # in practice.
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

    return spring_step


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
