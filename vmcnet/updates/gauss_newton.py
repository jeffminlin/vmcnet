"""Gauss Newton implementation."""

import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, LocalEnergyApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import (
    UpdateDataFn,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
)

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer


def construct_gauss_newton_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)

        energy, _, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state = optimizer_apply(
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
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_gauss_newton(
    local_energy_fn: LocalEnergyApply[P],
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    gauss_newton_step = get_gauss_newton_step(
        local_energy_fn,
        log_psi_apply,
        optimizer_config.tau,
        optimizer_config.damping,
        optimizer_config.clip_threshold,
        optimizer_config.sketchy,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def optimizer_apply(params, optimizer_state, data):
        positions = get_position_fn(data)
        grad = gauss_newton_step(params, positions)

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

    update_param_fn = construct_gauss_newton_update_param_fn(
        energy_and_statistics_fn,
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


def get_gauss_newton_step(
    local_energy_fn: LocalEnergyApply[P],
    log_psi_apply: ModelApply[P],
    tau: chex.Scalar = 0.0,
    damping: chex.Scalar = 0.001,
    clip_threshold: chex.Scalar = 5.0,
    sketchy=True,
):
    """Get the Gauss Newton update function."""
    # TODO: optimize this approach to avoid calculating the local energies repeatedly.
    batch_local_energy_fn = jax.vmap(local_energy_fn, in_axes=(None, 0), out_axes=0)

    def ravel_grad_log_psi(params, positions):
        grad = jax.grad(log_psi_apply, argnums=0)(params, positions)
        return jax.flatten_util.ravel_pytree(grad)[0]

    def ravel_grad_E(params, positions):
        grad = jax.grad(local_energy_fn, argnums=0)(params, positions)
        return jax.flatten_util.ravel_pytree(grad)[0]

    def get_O_A(params, positions, local_energies):
        nchains = local_energies.shape[0]

        O = jax.vmap(ravel_grad_log_psi, in_axes=(None, 0))(params, positions)
        O = (O - jnp.mean(O, axis=0, keepdims=True))

        A = jax.vmap(ravel_grad_E, in_axes=(None, 0))(params, positions)
        A = A + jnp.expand_dims(local_energies, -1) * O

        return O / jnp.sqrt(nchains), A / jnp.sqrt(nchains)

    def gauss_newton_step(
        params: P,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        _, unravel_fn = jax.flatten_util.ravel_pytree(params)

        local_energies = batch_local_energy_fn(params, positions)

        E = jnp.mean(local_energies)
        r = local_energies - E
        mean_abs_res = jnp.mean(jnp.abs(r))
        r = jnp.clip(
            r, -clip_threshold * mean_abs_res, clip_threshold * mean_abs_res
        )
        r /= jnp.sqrt(nchains)

        O, A = get_O_A(params, positions, local_energies)
        A = A - tau * O

        if sketchy:
            TO = O @ O.T
            TA = A @ A.T

            solve_part = jnp.linalg.solve(TA @ TO + damping * jnp.eye(nchains), r)
            flat_update = A.T @ (TO @ solve_part)
        else:
            solve_part = jnp.linalg.solve(A @ O.T + damping * jnp.eye(nchains), r)
            flat_update = O.T @ solve_part

        return unravel_fn(flat_update)

    return gauss_newton_step


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
