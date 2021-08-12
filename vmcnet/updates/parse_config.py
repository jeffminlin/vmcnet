"""Get update functions from ConfigDicts."""
from typing import Callable, Tuple

import jax.numpy as jnp
import kfac_ferminet_alpha
import kfac_ferminet_alpha.optimizer as kfac_opt
import optax
from ml_collections import ConfigDict

import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.typing import D, GetPositionFromData, ModelApply, P

from .params import (
    UpdateParamFn,
    constrain_norm,
    create_grad_energy_update_param_fn,
    create_kfac_update_param_fn,
)
from .sr import SRMode, get_fisher_inverse_fn


def _get_kfac_update_fn(
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: jnp.ndarray,
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, kfac_opt.State], kfac_opt.State, jnp.ndarray]:
    optimizer = kfac_ferminet_alpha.Optimizer(
        energy_data_val_and_grad,
        l2_reg=optimizer_config.l2_reg,
        norm_constraint=optimizer_config.norm_constraint,
        value_func_has_aux=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=optimizer_config.curvature_ema,
        inverse_update_period=optimizer_config.inverse_update_period,
        min_damping=optimizer_config.min_damping,
        num_burnin_steps=0,
        register_only_generic=optimizer_config.register_only_generic,
        estimation_mode=optimizer_config.estimation_mode,
        multi_device=apply_pmap,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
    )
    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)

    optimizer_state = optimizer.init(params, subkey, get_position_fn(data))

    update_param_fn = create_kfac_update_param_fn(
        optimizer,
        optimizer_config.damping,
        pacore.get_position_from_data,
        record_param_l1_norm=record_param_l1_norm,
    )

    return update_param_fn, optimizer_state, key


def _init_optax_optimizer(
    optimizer: optax.GradientTransformation, params: P, apply_pmap: bool = True
) -> optax.OptState:
    optimizer_init = optimizer.init
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def _get_optax_update_fn(
    optimizer: optax.GradientTransformation,
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    def optimizer_apply(grad, params, optimizer_state, data):
        del data
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )

    optimizer_state = _init_optax_optimizer(optimizer, params, apply_pmap=apply_pmap)

    return update_param_fn, optimizer_state


def _get_adam_update_fn(
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    optimizer = optax.adam(learning_rate=learning_rate_schedule, **optimizer_config)

    return _get_optax_update_fn(
        optimizer,
        params,
        get_position_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def _get_sgd_update_fn(
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_config.momentum if optimizer_config.momentum != 0 else None,
        nesterov=optimizer_config.nesterov,
    )

    return _get_optax_update_fn(
        optimizer,
        params,
        get_position_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def _get_sr_update_fn(
    log_psi_apply: ModelApply[P],
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    descent_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
    nan_safe: bool = True,
):
    maxiter = optimizer_config.maxiter if optimizer_config.maxiter >= 0 else None
    mean_grad_fn = utils.distribute.get_mean_fn(nan_safe=nan_safe)
    precondition_grad_fn = get_fisher_inverse_fn(
        log_psi_apply,
        mean_grad_fn,
        damping=optimizer_config.damping,
        maxiter=maxiter,
        mode=SRMode[optimizer_config.mode.upper()],
    )

    if optimizer_config.descent_type == "adam":
        descent_opt_constructor = optax.adam
    elif optimizer_config.descent_type == "sgd":
        descent_opt_constructor = optax.sgd
    else:
        raise ValueError(
            "Requested descent type not supported; {} was requested".format(
                optimizer_config.descent_type
            )
        )

    descent_optimizer = descent_opt_constructor(
        learning_rate=learning_rate_schedule, **descent_config
    )

    def get_optimizer_step_count(optimizer_state):
        return optimizer_state[1].count

    def optimizer_apply(grad, params, optimizer_state, data):

        preconditioned_grad = precondition_grad_fn(grad, params, get_position_fn(data))
        step_count = get_optimizer_step_count(optimizer_state)
        learning_rate = learning_rate_schedule(step_count)
        constrained_grad = constrain_norm(
            grad, preconditioned_grad, learning_rate, optimizer_config.norm_constraint
        )

        updates, optimizer_state = descent_optimizer.update(
            constrained_grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = _init_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state
