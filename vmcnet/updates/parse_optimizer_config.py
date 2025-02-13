"""Get update functions from ConfigDicts."""

from typing import Tuple
from ml_collections import ConfigDict

import vmcnet.physics as physics
from vmcnet.utils.typing import (
    D,
    GetPositionFromData,
    LearningRateSchedule,
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
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: PRNGKey,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update function and initialize optimizer state from the vmc configuration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        vmc_config (ConfigDict): configuration for VMC
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        update_data_fn (Callable): function which updates data for new params
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (PRNGKey): PRNGKey with which to initialize optimizer state
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Raises:
        ValueError: A non-supported optimizer type is requested. Currently, KFAC, Adam,
            SGD, and SR (with either Adam or SGD) is supported.

    Returns:
        (UpdateParamFn, OptimizerState, PRNGKey):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
    learning_rate_schedule = _get_learning_rate_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )

    if vmc_config.optimizer_type == "kfac":
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
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_spring(
            log_psi_apply,
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.spring,
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
