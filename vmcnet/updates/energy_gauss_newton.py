"""EGN implementation, WIP."""

from typing import Callable, Dict
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, LocalEnergyApply, ModelApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer


def construct_energy_gauss_newton_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
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
            energy,
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


def initialize_energy_gauss_newton(
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
    """Get an update param function and initial state for EGN."""
    energy_gauss_newton_step = get_energy_gauss_newton_step(
        log_psi_apply,
        local_energy_fn,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        grad = energy_gauss_newton_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            positions,
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

    update_param_fn = construct_energy_gauss_newton_update_param_fn(
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


def get_energy_gauss_newton_step(
    log_psi_apply: ModelApply[P],
    local_energy_fn: LocalEnergyApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
):

    """Get the EGN update function."""
    def ravel_grad_log_psi(params, positions):
        grad = jax.grad(log_psi_apply, argnums=0)(params, positions)
        return jax.flatten_util.ravel_pytree(grad)[0]

    def ravel_grad_E(params, positions):
        value, grad = jax.value_and_grad(local_energy_fn, argnums=0)(params, positions)
        return value, jax.flatten_util.ravel_pytree(grad)[0]

    def energy_gauss_newton_step(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        _, unravel_fn = jax.flatten_util.ravel_pytree(params)

        O = jax.vmap(ravel_grad_log_psi, in_axes=(None, 0))(params, positions)
        O = (O - jnp.mean(O, axis=0, keepdims=True))

        EL, EL_grad = jax.vmap(ravel_grad_E, in_axes=(None, 0))(params, positions)
        EL = (EL - jnp.mean(EL)) / jnp.sqrt(nchains)
        A = EL_grad + EL[:, None] * O
        A = (A - jnp.mean(A, axis=0, keepdims=True)) / jnp.sqrt(nchains)
        O /= jnp.sqrt(nchains)

        flat_update = O.T @ jnp.linalg.lstsq(A @ O.T, EL, rcond=damping)[0]

        return unravel_fn(flat_update)

    return energy_gauss_newton_step


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
