"""Explicit KFAC-preconditioned SPRING/minSR prototype."""

from typing import Callable, NamedTuple, Tuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import kfac_jax
import optax
from kfac_jax import Optimizer as kfac_Optimizer
from ml_collections import ConfigDict

import vmcnet.physics as physics
import vmcnet.utils as utils
import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks
from vmcnet.utils.pytree_helpers import tree_reduce_l1
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    P,
    PRNGKey,
    UpdateDataFn,
)

from .optax_utils import initialize_optax_optimizer
from .spring import constrain_norm
from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)


class KFACSpringState(NamedTuple):
    """Optimizer state for the explicit KFAC-SPRING prototype."""

    kfac_state: kfac_jax.optimizer.OptimizerState
    descent_state: optax.OptState


def get_centered_jacobian(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
) -> Tuple[Array, Callable]:
    """Return the centered log-amplitude Jacobian and parameter unravel function."""

    def ravel_grad_log_psi(single_position):
        grad = jax.grad(log_psi_apply, argnums=0)(params, single_position)
        return jax.flatten_util.ravel_pytree(grad)[0]

    jacobian = jax.vmap(ravel_grad_log_psi)(positions)
    jacobian = jacobian - jnp.mean(jacobian, axis=0, keepdims=True)
    _, unravel_fn = jax.flatten_util.ravel_pytree(params)
    return jacobian, unravel_fn


def apply_kfac_inverse_to_flat_vector(
    kfac_optimizer: kfac_jax.Optimizer,
    kfac_state: kfac_jax.optimizer.OptimizerState,
    unravel_fn,
    flat_vector: Array,
    identity_weight: chex.Numeric,
    exact_power: bool,
    use_cached: bool,
) -> Array:
    """Apply the damped KFAC inverse to one flat parameter-space vector."""
    vector = unravel_fn(flat_vector)
    preconditioned_vector = kfac_optimizer.estimator.multiply_inverse(
        state=kfac_state.estimator_state,
        parameter_structured_vector=vector,
        identity_weight=identity_weight,
        exact_power=exact_power,
        use_cached=use_cached,
        pmap_axis_name=None,
    )
    return jax.flatten_util.ravel_pytree(preconditioned_vector)[0]


def get_kfac_spring_step(
    log_psi_apply: ModelApply[P],
    kfac_optimizer: kfac_jax.Optimizer,
    sketch_damping: chex.Scalar,
    kfac_inverse_damping: chex.Scalar,
    mu: chex.Scalar = 0.99,
    exact_power: bool = True,
    use_cached: bool = False,
):
    """Get an explicit KFAC-preconditioned SPRING/minSR step function."""

    def kfac_spring_step(
        centered_energies: Array,
        params: P,
        previous_direction: P,
        positions: Array,
        kfac_state: kfac_jax.optimizer.OptimizerState,
    ) -> Tuple[P, Array]:
        nchains = positions.shape[0]
        jacobian, unravel_fn = get_centered_jacobian(log_psi_apply, params, positions)
        scaled_jacobian = jacobian / jnp.sqrt(nchains)

        apply_inverse = lambda flat_vector: apply_kfac_inverse_to_flat_vector(
            kfac_optimizer,
            kfac_state,
            unravel_fn,
            flat_vector,
            kfac_inverse_damping,
            exact_power,
            use_cached,
        )

        preconditioned_jacobian_t = jax.vmap(apply_inverse, in_axes=1, out_axes=1)(
            scaled_jacobian.T
        )
        preconditioned_kernel = scaled_jacobian @ preconditioned_jacobian_t
        preconditioned_kernel = (preconditioned_kernel + preconditioned_kernel.T) / 2

        kernel_eigenvalues, kernel_eigenvectors = jnp.linalg.eigh(
            preconditioned_kernel
        )
        kernel_eigenvalues = jnp.maximum(kernel_eigenvalues, 0) + sketch_damping

        mu_previous = jax.tree_map(lambda x: mu * x, previous_direction)
        flat_mu_previous, _ = jax.flatten_util.ravel_pytree(mu_previous)
        residual = centered_energies / jnp.sqrt(nchains)
        residual = residual - scaled_jacobian @ flat_mu_previous

        zeta = kernel_eigenvectors @ (
            (kernel_eigenvectors.T @ residual) / kernel_eigenvalues
        )
        zeta = zeta - jnp.mean(zeta)

        flat_direction = preconditioned_jacobian_t @ zeta
        direction = unravel_fn(flat_direction)
        direction = jax.tree_map(
            lambda dtheta, previous: dtheta + previous, direction, mu_previous
        )
        return direction, preconditioned_kernel

    return kfac_spring_step


def construct_kfac_spring_update_param_fn(
    energy_and_statistics_fn,
    kfac_optimizer: kfac_jax.Optimizer,
    kfac_step_damping: chex.Scalar,
    optimizer_apply,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = False,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, KFACSpringState]:
    """Create the explicit KFAC-SPRING update function."""
    if apply_pmap:
        raise NotImplementedError("KFAC-SPRING prototype only supports single-device.")

    def update_param_fn(params, data, optimizer_state, key):
        positions = get_position_fn(data)

        key, subkey = jax.random.split(key)
        _, kfac_state, _ = kfac_optimizer.step(
            params=params,
            state=optimizer_state.kfac_state,
            rng=subkey,
            batch=positions,
            learning_rate=jnp.asarray(0.0),
            momentum=jnp.asarray(0.0),
            damping=kfac_step_damping,
            global_step_int=0,
        )
        optimizer_state = KFACSpringState(kfac_state, optimizer_state.descent_state)

        energy, local_energies, stats = energy_and_statistics_fn(params, positions)
        centered_local_energies = local_energies - energy

        params, optimizer_state, kernel_trace = optimizer_apply(
            centered_local_energies,
            params,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics["kfac_spring_kernel_trace"] = kernel_trace
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    return make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)


def initialize_kfac_spring(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    key: PRNGKey,
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = False,
) -> Tuple[UpdateParamFn[P, D, KFACSpringState], KFACSpringState, PRNGKey]:
    """Initialize the explicit KFAC-preconditioned SPRING/minSR prototype."""
    if apply_pmap:
        raise NotImplementedError("KFAC-SPRING prototype only supports single-device.")

    def kfac_value_and_grad_fn(params, rng, positions):
        del rng
        energy, stats, grad_energy = energy_data_val_and_grad(params, positions)
        return (energy, stats), grad_energy

    kfac_optimizer = kfac_Optimizer(
        kfac_value_and_grad_fn,
        l2_reg=optimizer_config.kfac_l2_reg,
        norm_constraint=optimizer_config.kfac_norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        curvature_ema=optimizer_config.kfac_curvature_ema,
        inverse_update_period=1,
        min_damping=optimizer_config.kfac_min_damping,
        num_burnin_steps=0,
        register_only_generic=optimizer_config.kfac_register_only_generic,
        estimation_mode=optimizer_config.kfac_estimation_mode,
        multi_device=False,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        use_exact_inverses=optimizer_config.exact_power,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
    )

    key, subkey = jax.random.split(key)
    kfac_state = kfac_optimizer.init(params, subkey, get_position_fn(data))

    kfac_spring_step = get_kfac_spring_step(
        log_psi_apply,
        kfac_optimizer,
        optimizer_config.sketch_damping,
        optimizer_config.kfac_inverse_damping,
        optimizer_config.mu,
        optimizer_config.exact_power,
        optimizer_config.use_cached,
    )
    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def previous_direction(optimizer_state):
        return optimizer_state.descent_state[0].trace

    def optimizer_apply(centered_local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)
        direction, preconditioned_kernel = kfac_spring_step(
            centered_local_energies,
            params,
            previous_direction(optimizer_state),
            positions,
            optimizer_state.kfac_state,
        )
        updates, descent_state = descent_optimizer.update(
            direction, optimizer_state.descent_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm(updates, optimizer_config.norm_constraint)

        params = optax.apply_updates(params, updates)
        optimizer_state = KFACSpringState(optimizer_state.kfac_state, descent_state)
        return params, optimizer_state, jnp.trace(preconditioned_kernel)

    update_param_fn = construct_kfac_spring_update_param_fn(
        energy_and_statistics_fn,
        kfac_optimizer,
        optimizer_config.kfac_inverse_damping,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    descent_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
    optimizer_state = KFACSpringState(kfac_state, descent_state)

    return update_param_fn, optimizer_state, key
