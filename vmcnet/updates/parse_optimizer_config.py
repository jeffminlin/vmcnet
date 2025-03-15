"""Get update functions from ConfigDicts."""

from typing import Optional, Tuple
from ml_collections import ConfigDict

import vmcnet.physics as physics
from vmcnet.utils.typing import (
    ClippingFn,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    LocalEnergyApply,
    ModelApply,
    OptimizerState,
    P,
    PRNGKey,
    UpdateDataFn,
)

from .update_param_fns import UpdateParamFn
from .optax_utils import (
    initialize_adam,
    initialize_sgd,
)
from .spring import initialize_spring
from .kfac import initialize_kfac
from .gauss_newton import initialize_gauss_newton
from .var_sr import initialize_var_sr


def _get_learning_rate_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate

    elif optimizer_config.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / (
                1.0 + optimizer_config.learning_decay_rate * t
            )

    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type
            )
        )

    return learning_rate_schedule


def initialize_optimizer(
    log_psi_apply: ModelApply[P],
    local_energy_fn: LocalEnergyApply[P],
    clipping_fn: Optional[ClippingFn],
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    key: PRNGKey,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update function and initialize optimizer state from the vmc configuration."""
    learning_rate_schedule = _get_learning_rate_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )

    if vmc_config.optimizer_type == "kfac":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply,
            local_energy_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        return initialize_kfac(
            params,
            data,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            key,
            learning_rate_schedule,
            vmc_config.optimizer.kfac,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
    elif vmc_config.optimizer_type == "sgd":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply,
            local_energy_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_sgd(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sgd,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "adam":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply,
            local_energy_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_adam(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.adam,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key

    elif vmc_config.optimizer_type == "spring":
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            local_energy_fn, vmc_config.nchains, clipping_fn, vmc_config.nan_safe
        )

        (
            update_param_fn,
            optimizer_state,
        ) = initialize_spring(
            log_psi_apply,
            energy_and_statistics_fn,
            params,
            get_position_fn,
            update_data_fn,
            learning_rate_schedule,
            vmc_config.optimizer.spring,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "gauss_newton":
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            local_energy_fn, vmc_config.nchains, clipping_fn, vmc_config.nan_safe
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_gauss_newton(
            local_energy_fn,
            log_psi_apply,
            energy_and_statistics_fn,
            params,
            get_position_fn,
            update_data_fn,
            learning_rate_schedule,
            vmc_config.optimizer.gauss_newton,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "var_sr":
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            local_energy_fn, vmc_config.nchains, clipping_fn, vmc_config.nan_safe
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_var_sr(
            local_energy_fn,
            log_psi_apply,
            energy_and_statistics_fn,
            params,
            get_position_fn,
            update_data_fn,
            learning_rate_schedule,
            vmc_config.optimizer.gauss_newton,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )
