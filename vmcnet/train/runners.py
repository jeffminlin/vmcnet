"""Entry points for running standard jobs."""
import datetime
import functools
import logging
import os
from typing import Callable, Optional, Tuple
import flax

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha
import kfac_ferminet_alpha.optimizer as kfac_opt
import optax
from ml_collections import ConfigDict

import vmcnet.mcmc as mcmc
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.updates as updates
import vmcnet.utils as utils
import vmcnet.train as train
from vmcnet.utils.typing import P, D, S, ModelApply, OptimizerState


def _get_logdir_and_save_config(reload_config: ConfigDict, config: ConfigDict) -> str:
    logging.info("Reload configuration: \n%s", reload_config)
    logging.info("Running with configuration: \n%s", config)
    if config.logdir:
        if config.save_to_current_datetime_subfolder:
            config.logdir = os.path.join(
                config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        logdir = config.logdir
        reload_config_filename = utils.io.add_suffix_for_uniqueness(
            "reload_config", logdir, trailing_suffix=".json"
        )
        config_filename = utils.io.add_suffix_for_uniqueness(
            "config", logdir, trailing_suffix=".json"
        )
        with utils.io.open_or_create(
            logdir, reload_config_filename + ".json", "w"
        ) as f:
            f.write(reload_config.to_json(indent=4))
        with utils.io.open_or_create(logdir, config_filename + ".json", "w") as f:
            f.write(config.to_json(indent=4))
    else:
        logdir = None
    return logdir


def _get_dtype(config: ConfigDict) -> jnp.dtype:
    if config.dtype == "float32":
        return jnp.float32
    elif config.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
        return jnp.float64

    raise ValueError(
        "dtype other than float32, float64 not supported; {} was requested".format(
            config.dtype
        )
    )


def _get_electron_ion_config_as_arrays(
    config: ConfigDict, dtype=jnp.float32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    ion_pos = jnp.array(config.problem.ion_pos, dtype=dtype)
    ion_charges = jnp.array(config.problem.ion_charges, dtype=dtype)
    nelec = jnp.array(config.problem.nelec)
    nelec_total = jnp.sum(nelec)
    return ion_pos, ion_charges, nelec, nelec_total


def _get_and_init_model(
    model_config: ConfigDict,
    ion_pos: jnp.ndarray,
    nelec: jnp.ndarray,
    init_pos: jnp.ndarray,
    key: jnp.ndarray,
    dtype=jnp.float32,
) -> Tuple[flax.linen.Module, flax.core.FrozenDict, jnp.ndarray]:
    log_psi = models.construct.get_model_from_config(
        model_config, nelec, ion_pos, dtype=dtype
    )
    key, subkey = jax.random.split(key)
    params = log_psi.init(subkey, init_pos[0:1])
    params = utils.distribute.replicate_all_local_devices(params)
    return log_psi, params, key


# TODO: figure out how to merge this and other distributing logic with the current
# vmcnet/utils/distribute.py as well as vmcnet/mcmc
# TODO: make this flexible w.r.t. the type of data, not just use dwpa
# TODO: Here and elsewhere, fix the type hinting for model.apply and the local energy,
# which are more accurately described as Callables with signature
# (params, potentially-multiple-args-not-necessarily-arrays...) -> array
#
# The easiest, but somewhat inaccurate solution might be to just do
# Callable[[P, Union[jnp.ndarray, SLArray]], jnp.ndarray]
#
# The ideal would probably be something like Callable[[P, ...], jnp.ndarray], but this
# is not allowed (probably for good reason)
#
# The correct solution is probably something like this involving Protocols (PEP 544):
#
#     class ModelApply(Protocol[P]):
#         def __call__(params: P, *args) -> jnp.ndarray:
#             ...
#
# which creates a Generic class called ModelApply with only the first argument typed
def _make_initial_distributed_data(
    distributed_log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    init_pos: jnp.ndarray,
    params: P,
):
    sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
    sharded_amplitudes = distributed_log_psi_apply(params, sharded_init_pos)
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=run_config.std_move, move_acceptance_sum=0.0, moves_since_update=0
        )
    )
    data = pacore.make_position_amplitude_data(
        sharded_init_pos, sharded_amplitudes, move_metadata
    )

    return data


# TODO: add threshold_adjust_std_move options to configs
# TODO: add more options than just dwpa
# TODO: remove dependence on exact field names
def _get_mcmc_fns(
    run_config: ConfigDict, log_psi_apply: ModelApply[P]
) -> Tuple[
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi_apply,
        run_config.nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )
    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        run_config.nsteps_per_param_update, metrop_step_fn
    )

    return burning_step, walker_fn


# TODO: figure out where this should go, perhaps in a physics/molecule.py file?
def _assemble_mol_local_energy_fn(
    ion_pos: jnp.ndarray,
    ion_charges: jnp.ndarray,
    log_psi_apply: ModelApply[P],
) -> ModelApply[P]:
    # Define parameter updates
    kinetic_fn = physics.kinetic.create_continuous_kinetic_energy(log_psi_apply)
    ei_potential_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_pos, ion_charges
    )
    ee_potential_fn = physics.potential.create_electron_electron_coulomb_potential()
    ii_potential_fn = physics.potential.create_ion_ion_coulomb_potential(
        ion_pos, ion_charges
    )

    local_energy_fn = physics.core.combine_local_energy_terms(
        [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
    )

    return local_energy_fn


# TODO: figure out where this should go -- the act of clipping energies is kind of just
# a training trick rather than a physics thing, so maybe this stays here
def total_variation_clipping_fn(local_energies, threshold=5.0):
    """Clip local energy to within a multiple of the total variation from the median."""
    median_local_e = jnp.nanmedian(local_energies)
    total_variation = jnp.nanmean(jnp.abs(local_energies - median_local_e))
    clipped_local_e = jnp.clip(
        local_energies,
        median_local_e - threshold * total_variation,
        median_local_e + threshold * total_variation,
    )
    return clipped_local_e


# TODO: possibly include other types of clipping functions? e.g. using std deviation
# instead of total variation
def _get_clipping_fn(
    vmc_config: ConfigDict,
) -> Optional[Callable[[jnp.ndarray], jnp.ndarray]]:
    clipping_fn = None
    if vmc_config.clip_threshold > 0.0:
        clipping_fn = functools.partial(
            total_variation_clipping_fn, threshold=vmc_config.clip_threshold
        )
    return clipping_fn


# TODO: Here and in physics/core.py, clean up the type hinting
def _get_energy_fns(
    vmc_config: ConfigDict,
    ion_pos: jnp.ndarray,
    ion_charges: jnp.ndarray,
    log_psi_apply: ModelApply[P],
) -> Tuple[ModelApply[P], physics.core.ValueGradEnergyFn[P]]:
    local_energy_fn = _assemble_mol_local_energy_fn(ion_pos, ion_charges, log_psi_apply)
    clipping_fn = _get_clipping_fn(vmc_config)
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply,
        local_energy_fn,
        vmc_config.nchains,
        clipping_fn,
        nan_safe=vmc_config.nan_safe,
    )

    return local_energy_fn, energy_data_val_and_grad


def _get_learning_rate_schedule(vmc_config: ConfigDict) -> Callable[[int], jnp.float32]:
    if vmc_config.schedule_type == "constant":

        def learning_rate_schedule(t):
            return vmc_config.learning_rate

    elif vmc_config.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return vmc_config.learning_rate / (1.0 + vmc_config.learning_decay_rate * t)

    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                vmc_config.schedule_type
            )
        )

    return learning_rate_schedule


def _get_kfac_update_fn(
    params: P,
    data: D,
    get_position_fn: Callable[[D], jnp.ndarray],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    sharded_key: jnp.ndarray,
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
) -> Tuple[
    updates.params.UpdateParamFn[P, D, kfac_opt.State], kfac_opt.State, jnp.ndarray
]:
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
        multi_device=True,
        pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
    )
    sharded_key, subkeys = utils.distribute.p_split(sharded_key)
    optimizer_state = optimizer.init(params, subkeys, get_position_fn(data))

    update_param_fn = updates.params.create_kfac_update_param_fn(
        optimizer, optimizer_config.damping, pacore.get_position_from_data
    )

    return update_param_fn, optimizer_state, sharded_key


def _get_adam_update_fn(
    params: P,
    get_position_fn: Callable[[D], jnp.ndarray],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    learning_rate_schedule: Callable[[int], jnp.float32],
    optimizer_config: ConfigDict,
) -> Tuple[updates.params.UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    optimizer = optax.adam(learning_rate=learning_rate_schedule, **optimizer_config)
    optimizer_state = utils.distribute.pmap(optimizer.init)(params)

    def optimizer_apply(grad, params, optimizer_state):
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        optimizer_apply,
        get_position_fn=get_position_fn,
    )

    return update_param_fn, optimizer_state


def _get_update_fn_and_init_optimizer(
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: Callable[[D], jnp.ndarray],
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    sharded_key: jnp.ndarray,
) -> Tuple[
    updates.params.UpdateParamFn[P, D, OptimizerState],
    OptimizerState,
    jnp.ndarray,
]:

    learning_rate_schedule = _get_learning_rate_schedule(vmc_config)

    if vmc_config.optimizer_type == "kfac":
        return _get_kfac_update_fn(
            params,
            data,
            get_position_fn,
            energy_data_val_and_grad,
            sharded_key,
            learning_rate_schedule,
            vmc_config.optimizer.kfac,
        )
    elif vmc_config.optimizer_type == "adam":
        update_param_fn, optimizer_state = _get_adam_update_fn(
            params,
            get_position_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.adam,
        )
        return update_param_fn, optimizer_state, sharded_key
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )


# TODO: don't forget to update type hint to be more general when
# _make_initial_distributed_data is more general
# TODO: along with many of the other... intense type hints in this file, this should
# probably be cleaned up -- it's probably not really serving the readability goal atm
def _setup_distributed_vmc(
    config: ConfigDict,
    init_pos: jnp.ndarray,
    ion_pos: jnp.ndarray,
    ion_charges: jnp.ndarray,
    nelec: jnp.ndarray,
    key: jnp.ndarray,
    dtype=jnp.float32,
) -> Tuple[
    flax.linen.Module,
    ModelApply[flax.core.FrozenDict],
    mcmc.metropolis.BurningStep[flax.core.FrozenDict, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[flax.core.FrozenDict, dwpa.DWPAData],
    ModelApply[flax.core.FrozenDict],
    updates.params.UpdateParamFn[flax.core.FrozenDict, dwpa.DWPAData, OptimizerState],
    flax.core.FrozenDict,
    dwpa.DWPAData,
    OptimizerState,
    jnp.ndarray,
]:

    # Make the model
    log_psi, params, key = _get_and_init_model(
        config.model, ion_pos, nelec, init_pos, key, dtype=dtype
    )
    distributed_log_psi_apply = utils.distribute.pmap(log_psi.apply)

    # Make initial data
    data = _make_initial_distributed_data(
        distributed_log_psi_apply, config.vmc, init_pos, params
    )

    # Setup metropolis step
    burning_step, walker_fn = _get_mcmc_fns(config.vmc, log_psi.apply)

    local_energy_fn, energy_data_val_and_grad = _get_energy_fns(
        config.vmc, ion_pos, ion_charges, log_psi.apply
    )

    # Setup parameter updates
    sharded_key = utils.distribute.make_different_rng_key_on_all_devices(key)
    update_param_fn, optimizer_state, sharded_key = _get_update_fn_and_init_optimizer(
        config.vmc,
        params,
        data,
        pacore.get_position_from_data,
        energy_data_val_and_grad,
        sharded_key,
    )

    return (
        log_psi,
        distributed_log_psi_apply,
        burning_step,
        walker_fn,
        local_energy_fn,
        update_param_fn,
        params,
        data,
        optimizer_state,
        sharded_key,
    )


# TODO: update output type hints when _get_mcmc_fns is made more general
def _setup_distributed_eval(
    config: ConfigDict,
    log_psi_apply: ModelApply[P],
    local_energy_fn: ModelApply[P],
    get_position_fn: Callable[[dwpa.DWPAData], jnp.ndarray],
) -> Tuple[
    updates.params.UpdateParamFn[P, dwpa.DWPAData, OptimizerState],
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    eval_update_param_fn = updates.params.create_eval_update_param_fn(
        local_energy_fn,
        config.eval.nchains,
        get_position_fn,
        nan_safe=config.eval.nan_safe,
    )
    eval_burning_step, eval_walker_fn = _get_mcmc_fns(config.eval, log_psi_apply)
    return eval_update_param_fn, eval_burning_step, eval_walker_fn


def _burn_and_run_vmc(
    run_config: ConfigDict,
    logdir: str,
    params: P,
    optimizer_state: S,
    data: D,
    burning_step: mcmc.metropolis.BurningStep[P, D],
    walker_fn: mcmc.metropolis.WalkerFn[P, D],
    update_param_fn: updates.params.UpdateParamFn[P, D, S],
    sharded_key: jnp.ndarray,
    should_checkpoint: bool = True,
) -> Tuple[P, S, D, jnp.ndarray]:
    if should_checkpoint:
        checkpoint_every = run_config.checkpoint_every
        best_checkpoint_every = run_config.best_checkpoint_every
        checkpoint_dir = run_config.checkpoint_dir
        checkpoint_variance_scale = run_config.checkpoint_variance_scale
        nhistory_max = run_config.nhistory_max
    else:
        checkpoint_every = None
        best_checkpoint_every = None
        checkpoint_dir = ""
        checkpoint_variance_scale = 0
        nhistory_max = 0

    data, sharded_key = mcmc.metropolis.burn_data(
        burning_step, run_config.nburn, params, data, sharded_key
    )
    params, optimizer_state, data, sharded_key = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        run_config.nchains,
        run_config.nepochs,
        walker_fn,
        update_param_fn,
        sharded_key,
        logdir=logdir,
        checkpoint_every=checkpoint_every,
        best_checkpoint_every=best_checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_variance_scale=checkpoint_variance_scale,
        nhistory_max=nhistory_max,
    )

    return params, optimizer_state, data, sharded_key


# TODO: add integration test which runs this runner with a close-to-default config
# (probably use smaller nchains and smaller nepochs) to make sure it doesn't raise
# top-level errors
def run_molecule() -> None:
    """Run VMC on a molecule."""
    reload_config, config = train.parse_config_flags.parse_flags()

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(reload_config, config)

    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec, nelec_total = _get_electron_ion_config_as_arrays(
        config, dtype=dtype_to_use
    )

    key = jax.random.PRNGKey(config.initial_seed)
    key, init_pos = physics.core.initialize_molecular_pos(
        key, config.vmc.nchains, ion_pos, ion_charges, nelec_total, dtype=dtype_to_use
    )

    (
        log_psi,
        distributed_log_psi_apply,
        burning_step,
        walker_fn,
        local_energy_fn,
        update_param_fn,
        params,
        data,
        optimizer_state,
        sharded_key,
    ) = _setup_distributed_vmc(
        config,
        init_pos,
        ion_pos,
        ion_charges,
        nelec,
        key,
        dtype=dtype_to_use,
    )

    reload_from_checkpoint = (
        reload_config.log_dir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_checkpoint_file
    )

    if reload_from_checkpoint:
        checkpoint_file_path = os.path.join(
            reload_config.log_dir, reload_config.checkpoint_relative_file_path
        )
        directory, filename = os.path.split(checkpoint_file_path)
        _, data, params, optimizer_state, key = utils.io.reload_vmc_state(
            directory, filename
        )
        (
            data,
            params,
            optimizer_state,
            sharded_key,
        ) = utils.distribute.distribute_vmc_state_from_checkpoint(
            data, params, optimizer_state, key
        )

    params, optimizer_state, data, sharded_key = _burn_and_run_vmc(
        config.vmc,
        logdir,
        params,
        optimizer_state,
        data,
        burning_step,
        walker_fn,
        update_param_fn,
        sharded_key,
        should_checkpoint=True,
    )

    logging.info("Completed VMC! Evaluating")

    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post
    # evaluation
    eval_logdir = os.path.join(logdir, "eval")

    eval_update_param_fn, eval_burning_step, eval_walker_fn = _setup_distributed_eval(
        config, log_psi.apply, local_energy_fn, pacore.get_position_from_data
    )
    optimizer_state = None

    if not config.eval.use_data_from_training:
        data = _make_initial_distributed_data(
            distributed_log_psi_apply, config.eval, init_pos, params
        )

    _burn_and_run_vmc(
        config.eval,
        eval_logdir,
        params,
        optimizer_state,
        data,
        eval_burning_step,
        eval_walker_fn,
        eval_update_param_fn,
        sharded_key,
        should_checkpoint=False,
    )
