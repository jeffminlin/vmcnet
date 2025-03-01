"""Gauss Newton implementation."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
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

        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state = optimizer_apply(
            local_energies,
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
        optimizer_config.E,
        optimizer_config.damping,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def optimizer_apply(local_energies, params, optimizer_state, data):
        grad = gauss_newton_step(
            local_energies,
            params,
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
    E: chex.Scalar,
    damping: chex.Scalar = 0.001,
):
    """Get the Gauss Newton update function."""

    def single_linearized_residual(params, position, local_energy):
        return local_energy_fn(params, position) + (local_energy - E) * log_psi_apply(
            params, position
        )

    grad_single_residual = jax.grad(single_linearized_residual, argnums=0)

    def ravel_grad_single_residual(params, position, local_energy):
        grad = grad_single_residual(params, position, local_energy)
        return jax.flatten_util.ravel_pytree(grad)[0]

    get_jacobian = jax.vmap(ravel_grad_single_residual, in_axes=(None, 0, 0))
    batch_local_energy_fn = jax.vmap(local_energy_fn, in_axes=(None, 0), out_axes=0)

    def gauss_newton_step(
        local_energies: Array,
        params: P,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]

        local_energies_noclip = batch_local_energy_fn(params, positions)
        J = get_jacobian(params, positions, local_energies_noclip) / jnp.sqrt(nchains)

        # Remove any directions that change the norm of the wavefunction
        grad_norm = jax.vjp(log_psi_apply, params, positions)[1](jnp.ones(nchains))[0]
        flat_grad_norm, unravel_fn = jax.flatten_util.ravel_pytree(grad_norm)
        flat_unit_grad_norm = flat_grad_norm / jnp.linalg.norm(flat_grad_norm)
        J = J - jnp.expand_dims(J @ flat_unit_grad_norm, -1) * jnp.expand_dims(
            flat_unit_grad_norm, 0
        )

        T = J @ J.T
        T = (T + T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        residuals = (local_energies - E) / jnp.sqrt(nchains)

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ residuals
        flat_update = J.T @ zeta

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
