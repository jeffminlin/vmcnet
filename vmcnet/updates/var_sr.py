"""SR implementation for variance minimization algorithm."""

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


def construct_var_sr_update_param_fn(
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


def initialize_var_sr(
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
    var_sr_step = get_var_sr_step(
        local_energy_fn,
        log_psi_apply,
        optimizer_config.E,
        optimizer_config.damping,
        optimizer_config.clip_threshold,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def optimizer_apply(params, optimizer_state, data):
        grad = var_sr_step(
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

    update_param_fn = construct_var_sr_update_param_fn(
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


def get_var_sr_step(
    local_energy_fn: LocalEnergyApply[P],
    log_psi_apply: ModelApply[P],
    E: chex.Scalar,
    damping: chex.Scalar = 0.001,
    clip_threshold: chex.Scalar = 5.0,
):
    """Get the Gauss Newton update function."""

    SR_kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())
    batch_local_energy_fn = jax.vmap(local_energy_fn, in_axes=(None, 0), out_axes=0)
    
    def ravel_grad_E(params, positions):
        grad = jax.grad(local_energy_fn, argnums=0)(params, positions)
        return jax.flatten_util.ravel_pytree(grad)[0]

    def get_J(params, positions):
        return jax.vmap(ravel_grad_E, in_axes=(None, 0))(params, positions)

    def var_sr_step(
        params: P,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        _, unravel_fn = jax.flatten_util.ravel_pytree(params)

        local_energies = batch_local_energy_fn(params, positions)
        residuals = local_energies - E
        mean_abs_res = jnp.mean(jnp.abs(residuals))
        residuals = jnp.clip(
            residuals, -clip_threshold * mean_abs_res, clip_threshold * mean_abs_res
        )

        # Interestingly (at least on CPU) building J is much faster than doing a vjp
        g_E = unravel_fn(get_J(params, positions).T @ residuals)
        v_Psi = residuals**2 - jnp.mean(residuals**2) # Center the residuals insteads of the amplitudes
        g_Psi = jax.vjp(log_psi_apply, params, positions)[1](v_Psi)[0]
        g = jax.tree_map(lambda x, y: (x + y) / nchains, g_E, g_Psi)

        SR_T = SR_kernel_fn(positions, positions, "ntk", params) / nchains
        SR_T = SR_T - jnp.mean(SR_T, axis=0, keepdims=True)
        SR_T = SR_T - jnp.mean(SR_T, axis=1, keepdims=True)
        SR_T = (SR_T + SR_T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(SR_T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        O_g = jax.jvp(
            log_psi_apply,
            (params, positions),
            (g, jnp.zeros_like(positions)),
        )[1] / jnp.sqrt(nchains)
        O_bar_g = O_g - jnp.mean(O_g, axis=0, keepdims=True)

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ O_bar_g
        zeta_hat = zeta - jnp.mean(zeta) # Center zeta instead of O

        O_contribution = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]
        return jax.tree_map(
            lambda g_raw, g_O: (g_raw - g_O / jnp.sqrt(nchains)) / damping,
            g,
            O_contribution,
        )

    return var_sr_step


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
