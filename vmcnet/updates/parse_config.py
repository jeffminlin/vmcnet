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
from vmcnet.utils.typing import D, GetPositionFromData, ModelApply, OptimizerState, P

from .params import (
    UpdateParamFn,
    constrain_norm,
    create_grad_energy_update_param_fn,
    create_kfac_update_param_fn,
)
from .sr import SRMode, get_fisher_inverse_fn


def _get_learning_rate_schedule(
    optimizer_config: ConfigDict,
) -> Callable[[int], jnp.float32]:
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


def get_update_fn_and_init_optimizer(
    log_psi_apply: ModelApply[P],
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: jnp.ndarray,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, jnp.ndarray]:
    """Get an update function and initialize optimizer state from the vmc configuration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        vmc_config (ConfigDict): configuration for VMC
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (jnp.ndarray): PRNGKey with which to initialize optimizer state
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Raises:
        ValueError: A non-supported optimizer type is requested. Currently, KFAC, Adam,
        SGD, and SR (with either Adam or SGD) is supported.

    Returns:
        (UpdateParamFn, OptimizerState, jnp.ndarray):
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
        return get_kfac_update_fn_and_state(
            params,
            data,
            get_position_fn,
            energy_data_val_and_grad,
            key,
            learning_rate_schedule,
            vmc_config.optimizer.kfac,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
    elif vmc_config.optimizer_type == "sgd":
        (update_param_fn, optimizer_state,) = get_sgd_update_fn_and_state(
            params,
            get_position_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sgd,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "adam":
        (update_param_fn, optimizer_state,) = get_adam_update_fn_and_state(
            params,
            get_position_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.adam,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    elif vmc_config.optimizer_type == "sr":
        (update_param_fn, optimizer_state,) = get_sr_update_fn_and_state(
            log_psi_apply,
            params,
            get_position_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sr,
            vmc_config.optimizer[vmc_config.optimizer.sr.descent_type],
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
            nan_safe=vmc_config.nan_safe,
        )
        return update_param_fn, optimizer_state, key
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )


def get_kfac_update_fn_and_state(
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
    """Get an update param function, initial state, and key for KFAC.

    Args:
        params (pytree): params with which to initialize optimizer state
        data (pytree): data with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        key (jnp.ndarray): PRNGKey with which to initialize optimizer state
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for KFAC
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, kfac_opt.State, jnp.ndarray):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
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


def _get_optax_update_fn_and_state(
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


def get_adam_update_fn_and_state(
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for Adam.

    Args:
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
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
    optimizer = optax.adam(
        learning_rate=learning_rate_schedule,
        b1=optimizer_config.b1,
        b2=optimizer_config.b2,
        eps=optimizer_config.eps,
        eps_root=optimizer_config.eps_root,
    )

    return _get_optax_update_fn_and_state(
        optimizer,
        params,
        get_position_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def get_sgd_update_fn_and_state(
    params: P,
    get_position_fn: GetPositionFromData[D],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SGD.

    Args:
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
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
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule,
        momentum=optimizer_config.momentum if optimizer_config.momentum != 0 else None,
        nesterov=optimizer_config.nesterov,
    )

    return _get_optax_update_fn_and_state(
        optimizer,
        params,
        get_position_fn,
        energy_data_val_and_grad,
        record_param_l1_norm,
        apply_pmap,
    )


def get_sr_update_fn_and_state(
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
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for stochastic reconfiguration.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        params (pytree): params with which to initialize optimizer state
        get_position_fn (Callable): function which gets the position array from the data
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for stochastic reconfiguration
        descent_config (ConfigDict): configuration for the gradient descent-like method
            used to apply the preconditioned updates
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.
        nan_safe (bool, optional): whether the mean function used when centering the
            Jacobian of log|psi(x)| during the Fisher matvec is nan-safe. Defaults to
            True.

    Raises:
        ValueError: A non-supported descent type is requested. Currently only Adam and
        SGD are supported.

    Returns:
        (UpdateParamFn, optax.OptState):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key), and
        initial optimizer state
    """
    maxiter = optimizer_config.maxiter if optimizer_config.maxiter >= 0 else None
    mean_grad_fn = utils.distribute.get_mean_over_first_axis_fn(nan_safe=nan_safe)
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
