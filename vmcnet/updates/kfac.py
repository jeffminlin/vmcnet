"""Get update functions from ConfigDicts."""

from typing import Tuple

import jax
import kfac_jax
from kfac_jax import Optimizer as kfac_Optimizer
from ml_collections import ConfigDict
import chex
import jax.numpy as jnp

import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.pytree_helpers import (
    tree_reduce_l1,
)

import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks

from vmcnet.utils.typing import (
    Array,
    Callable,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    OptimizerState,
    P,
    PRNGKey,
    PyTree,
    UpdateDataFn,
)

from .update_param_fns import UpdateParamFn, update_metrics_with_noclip


def _get_traced_compute_param_norm(
    apply_pmap: bool = True,
) -> Callable[[PyTree], Array]:
    if not apply_pmap:
        return jax.jit(tree_reduce_l1)

    return utils.distribute.pmap(tree_reduce_l1)


def construct_kfac_update_fn(
    optimizer: kfac_jax.Optimizer,
    damping: chex.Numeric,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, OptimizerState]:
    """Create momentum-less KFAC update step function.

    Args:
        optimizer (kfac_jax.Optimizer): instance of the Optimizer class from
            kfac_jax
        damping (chex.Numeric): damping coefficient
        get_position_fn (GetPositionFromData): function which gets the walker positions
            from the data. Has signature data -> Array
        update_data_fn (Callable): function which updates data for new params

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
    """
    momentum = jnp.asarray(0.0)
    damping = jnp.asarray(damping)
    if optimizer.multi_device:
        momentum = utils.distribute.replicate_all_local_devices(momentum)
        damping = utils.distribute.replicate_all_local_devices(damping)
        update_data_fn = utils.distribute.pmap(update_data_fn)

    traced_compute_param_norm = _get_traced_compute_param_norm(optimizer.multi_device)

    def update_param_fn(params, data, optimizer_state, key):
        key, subkey = utils.distribute.split_or_psplit_key(key, optimizer.multi_device)
        params, optimizer_state, stats = optimizer.step(
            params=params,
            state=optimizer_state,
            rng=subkey,
            data_iterator=iter([get_position_fn(data)]),
            momentum=momentum,
            damping=damping,
        )
        data = update_data_fn(data, params)

        energy = stats["loss"]
        variance = stats["aux"]["variance"]
        energy_noclip = stats["aux"]["energy_noclip"]
        variance_noclip = stats["aux"]["variance_noclip"]
        picked_stats = (energy, variance, energy_noclip, variance_noclip)

        if record_param_l1_norm:
            param_l1_norm = traced_compute_param_norm(params)
            picked_stats = picked_stats + (param_l1_norm,)

        stats_to_save = picked_stats
        if optimizer.multi_device:
            stats_to_save = [utils.distribute.get_first(stat) for stat in picked_stats]

        metrics = {"energy": stats_to_save[0], "variance": stats_to_save[1]}
        metrics = update_metrics_with_noclip(
            stats_to_save[2], stats_to_save[3], metrics
        )

        if record_param_l1_norm:
            metrics.update({"param_l1_norm": stats_to_save[4]})

        return params, data, optimizer_state, metrics, key

    return update_param_fn


def initialize_kfac(
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    key: PRNGKey,
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update param function, initial state, and key for KFAC.

    Args:
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
        learning_rate_schedule (Callable): function which returns a learning rate from
            epoch number. Has signature epoch -> learning_rate
        optimizer_config (ConfigDict): configuration for KFAC
        record_param_l1_norm (bool, optional): whether to record the L1 norm of the
            parameters in the metrics. Defaults to False.
        apply_pmap (bool, optional): whether to pmap the optimizer steps. Defaults to
            True.

    Returns:
        (UpdateParamFn, kfac_opt.State, PRNGKey):
        update param function with signature
            (params, data, optimizer_state, key)
            -> (new params, new state, metrics, new key),
        initial optimizer state, and
        PRNGKey
    """
    optimizer = kfac_Optimizer(
        energy_data_val_and_grad,
        l2_reg=optimizer_config.l2_reg,
        norm_constraint=optimizer_config.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,  # type:ignore
        curvature_ema=optimizer_config.curvature_ema,
        inverse_update_period=optimizer_config.inverse_update_period,
        min_damping=optimizer_config.min_damping,
        num_burnin_steps=0,
        register_only_generic=optimizer_config.register_only_generic,
        estimation_mode=optimizer_config.estimation_mode,
        multi_device=apply_pmap,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        # Mypy can't find GRAPH_PATTERNS because we've ignored types in the curvature
        # tags file since it's not typed properly.
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,  # type: ignore
        ),
    )
    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)

    optimizer_state = optimizer.init(params, subkey, get_position_fn(data))

    update_param_fn = construct_kfac_update_fn(
        optimizer,
        optimizer_config.damping,
        pacore.get_position_from_data,
        update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
    )

    return update_param_fn, optimizer_state, key
