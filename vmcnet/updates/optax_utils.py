"""Routines which handle model parameter updating."""

from typing import Callable, Dict, Tuple
import optax
from ml_collections import ConfigDict
from .update_param_fns import construct_default_update_param_fn

import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    D,
    GetPositionFromData,
    LearningRateSchedule,
    P,
    PRNGKey,
    S,
    UpdateDataFn,
)

UpdateParamFn = Callable[[P, D, S, PRNGKey], Tuple[P, D, S, Dict, PRNGKey]]


def _get_adam_optax_optimizer(
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
) -> optax.GradientTransformation:
    return optax.adam(
        learning_rate=learning_rate_schedule,
        b1=optimizer_config.b1,
        b2=optimizer_config.b2,
        eps=optimizer_config.eps,
        eps_root=optimizer_config.eps_root,
    )


def _get_sgd_optax_optimizer(
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
) -> optax.GradientTransformation:
    return optax.sgd(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_config.momentum if optimizer_config.momentum != 0 else None,
        nesterov=optimizer_config.nesterov,
    )


def initialize_optax_optimizer(
    optimizer: optax.GradientTransformation, params: P, apply_pmap: bool = True
) -> optax.OptState:
    """Initialize an optax optimizer, handling pmapping if necessary."""
    optimizer_init = optimizer.init
    if apply_pmap:
        optimizer_init = utils.distribute.pmap(optimizer_init)
    optimizer_state = optimizer_init(params)
    return optimizer_state


def _initialize_optax_based_optimizer(
    optimizer: optax.GradientTransformation,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Initialize a VMC optimizer which is based on an optax optimizer."""

    def optimizer_apply(grad, params, optimizer_state, data, aux):
        del data, aux
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
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
        optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def initialize_adam(
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for Adam.

    Args:
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
        optimizer_config (ConfigDict): configuration for Adam
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
    optimizer = _get_adam_optax_optimizer(learning_rate_schedule, optimizer_config)

    return _initialize_optax_based_optimizer(
        optimizer,
        params,
        get_position_fn,
        update_data_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def initialize_sgd(
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SGD.

    Args:
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
        optimizer_config (ConfigDict): configuration for SGD
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
    optimizer = _get_sgd_optax_optimizer(learning_rate_schedule, optimizer_config)

    return _initialize_optax_based_optimizer(
        optimizer,
        params,
        get_position_fn,
        update_data_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )
