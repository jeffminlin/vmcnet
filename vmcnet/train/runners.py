"""Entry points for running standard jobs."""
import argparse
import datetime
import functools
import logging
import os
import subprocess
from typing import Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from ml_collections import ConfigDict

import vmcnet.mcmc as mcmc
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    P,
    ClippingFn,
    PRNGKey,
    D,
    S,
    GetPositionFromData,
    GetAmplitudeFromData,
    ModelApply,
    OptimizerState,
)

FLAGS = flags.FLAGS


def _get_logdir_and_save_config(reload_config: ConfigDict, config: ConfigDict) -> str:
    logging.info("Reload configuration: \n%s", reload_config)
    logging.info("Running with configuration: \n%s", config)
    if config.logdir:
        if config.save_to_current_datetime_subfolder:
            config.logdir = os.path.join(
                config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        logdir = config.logdir
        utils.io.save_config_dict_to_json(config, logdir, "config")
        utils.io.save_config_dict_to_json(reload_config, logdir, "reload_config")
    else:
        logdir = None
    return logdir


def _save_git_hash(logdir):
    if logdir is None:
        return

    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    git_file = os.path.join(logdir, "git_hash.txt")
    writer = open(git_file, "wt")
    writer.write(git_hash)
    writer.close()


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
) -> Tuple[Array, Array, Array]:
    ion_pos = jnp.array(config.problem.ion_pos, dtype=dtype)
    ion_charges = jnp.array(config.problem.ion_charges, dtype=dtype)
    nelec = jnp.array(config.problem.nelec)
    return ion_pos, ion_charges, nelec


def _get_and_init_model(
    model_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    init_pos: Array,
    key: PRNGKey,
    dtype=jnp.float32,
    apply_pmap: bool = True,
) -> Tuple[ModelApply[flax.core.FrozenDict], flax.core.FrozenDict, PRNGKey]:
    slog_psi = models.construct.get_model_from_config(
        model_config, nelec, ion_pos, ion_charges, dtype=dtype
    )
    key, subkey = jax.random.split(key)
    params = slog_psi.init(subkey, init_pos[0:1])
    if apply_pmap:
        params = utils.distribute.replicate_all_local_devices(params)
    log_psi_apply = models.construct.slog_psi_to_log_psi_apply(slog_psi.apply)
    return log_psi_apply, params, key


# TODO: figure out how to merge this and other distributing logic with the current
# vmcnet/utils/distribute.py as well as vmcnet/mcmc
# TODO: make this flexible w.r.t. the type of data, not just use dwpa
# TODO: Here and elsewhere, fix the type hinting for model.apply and the local energy,
# which are more accurately described as Callables with signature
# (params, potentially-multiple-args-not-necessarily-arrays...) -> array
#
# The easiest, but somewhat inaccurate solution might be to just do
# Callable[[P, Union[Array, SLArray]], Array]
#
# The ideal would probably be something like Callable[[P, ...], Array], but this
# is not allowed (probably for good reason)
#
# The correct solution is probably something like this involving Protocols (PEP 544):
#
#     class ModelApply(Protocol[P]):
#         def __call__(params: P, *args) -> Array:
#             ...
#
# which creates a Generic class called ModelApply with only the first argument typed
def _make_initial_distributed_data(
    distributed_log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    init_pos: Array,
    params: P,
) -> dwpa.DWPAData:
    # Need to use distributed_log_psi_apply here, in the case where there is not enough
    # memory to form the initial amplitudes on a single device
    sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
    sharded_amplitudes = distributed_log_psi_apply(params, sharded_init_pos)
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=run_config.std_move, move_acceptance_sum=0.0, moves_since_update=0
        )
    )
    return pacore.make_position_amplitude_data(
        sharded_init_pos, sharded_amplitudes, move_metadata
    )


def _make_initial_single_device_data(
    log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    init_pos: Array,
    params: P,
) -> dwpa.DWPAData:
    amplitudes = log_psi_apply(params, init_pos)
    return dwpa.make_dynamic_width_position_amplitude_data(
        init_pos,
        amplitudes,
        std_move=run_config.std_move,
        move_acceptance_sum=0.0,
        moves_since_update=0,
    )


def _make_initial_data(
    log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    init_pos: Array,
    params: P,
    apply_pmap: bool = True,
) -> dwpa.DWPAData:
    if apply_pmap:
        return _make_initial_distributed_data(
            utils.distribute.pmap(log_psi_apply), run_config, init_pos, params
        )
    else:
        return _make_initial_single_device_data(
            log_psi_apply, run_config, init_pos, params
        )


# TODO: add threshold_adjust_std_move options to configs
# TODO: add more options than just dwpa
# TODO: remove dependence on exact field names
def _get_mcmc_fns(
    run_config: ConfigDict, log_psi_apply: ModelApply[P], apply_pmap: bool = True
) -> Tuple[
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi_apply,
        run_config.nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )
    burning_step = mcmc.metropolis.make_jitted_burning_step(
        metrop_step_fn, apply_pmap=apply_pmap
    )
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        run_config.nsteps_per_param_update, metrop_step_fn, apply_pmap=apply_pmap
    )

    return burning_step, walker_fn


# TODO: figure out where this should go, perhaps in a physics/molecule.py file?
def _assemble_mol_local_energy_fn(
    ion_pos: Array,
    ion_charges: Array,
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

    local_energy_fn: ModelApply[P] = physics.core.combine_local_energy_terms(
        [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
    )

    return local_energy_fn


# TODO: figure out where this should go -- the act of clipping energies is kind of just
# a training trick rather than a physics thing, so maybe this stays here
def total_variation_clipping_fn(
    local_energies: Array, energy_noclip: jnp.float32, threshold=5.0, clip_center="mean"
) -> Array:
    """Clip local es to within a multiple of the total variation from a center."""
    if clip_center == "mean":
        center = energy_noclip
    elif clip_center == "median":
        center = jnp.nanmedian(local_energies)
    else:
        raise ValueError(
            "Only mean and median are supported clipping centers, but {} was "
            "requested".format(clip_center)
        )
    total_variation = jnp.nanmean(jnp.abs(local_energies - center))
    clipped_local_e = jnp.clip(
        local_energies,
        center - threshold * total_variation,
        center + threshold * total_variation,
    )
    return clipped_local_e


# TODO: possibly include other types of clipping functions? e.g. using std deviation
# instead of total variation
def _get_clipping_fn(
    vmc_config: ConfigDict,
) -> Optional[ClippingFn]:
    clipping_fn = None
    if vmc_config.clip_threshold > 0.0:
        clipping_fn = functools.partial(
            total_variation_clipping_fn,
            threshold=vmc_config.clip_threshold,
            clip_center=vmc_config.clip_center,
        )
    return clipping_fn


def _get_energy_fns(
    vmc_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
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


# TODO: don't forget to update type hint to be more general when
# _make_initial_distributed_data is more general
def _setup_vmc(
    config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    key: PRNGKey,
    dtype=jnp.float32,
    apply_pmap: bool = True,
) -> Tuple[
    ModelApply[flax.core.FrozenDict],
    mcmc.metropolis.BurningStep[flax.core.FrozenDict, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[flax.core.FrozenDict, dwpa.DWPAData],
    ModelApply[flax.core.FrozenDict],
    updates.params.UpdateParamFn[flax.core.FrozenDict, dwpa.DWPAData, OptimizerState],
    GetAmplitudeFromData[dwpa.DWPAData],
    flax.core.FrozenDict,
    dwpa.DWPAData,
    OptimizerState,
    PRNGKey,
]:
    nelec_total = jnp.sum(nelec)
    key, init_pos = physics.core.initialize_molecular_pos(
        key, config.vmc.nchains, ion_pos, ion_charges, nelec_total, dtype=dtype
    )

    # Make the model
    log_psi_apply, params, key = _get_and_init_model(
        config.model,
        ion_pos,
        ion_charges,
        nelec,
        init_pos,
        key,
        dtype=dtype,
        apply_pmap=apply_pmap,
    )

    # Make initial data
    data = _make_initial_data(
        log_psi_apply, config.vmc, init_pos, params, apply_pmap=apply_pmap
    )
    get_amplitude_fn = pacore.get_amplitude_from_data

    # Setup metropolis step
    burning_step, walker_fn = _get_mcmc_fns(
        config.vmc, log_psi_apply, apply_pmap=apply_pmap
    )

    local_energy_fn, energy_data_val_and_grad = _get_energy_fns(
        config.vmc, ion_pos, ion_charges, log_psi_apply
    )

    # Setup parameter updates
    if apply_pmap:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)
    (
        update_param_fn,
        optimizer_state,
        key,
    ) = updates.parse_config.get_update_fn_and_init_optimizer(
        log_psi_apply,
        config.vmc,
        params,
        data,
        pacore.get_position_from_data,
        energy_data_val_and_grad,
        key,
        apply_pmap=apply_pmap,
    )

    return (
        log_psi_apply,
        burning_step,
        walker_fn,
        local_energy_fn,
        update_param_fn,
        get_amplitude_fn,
        params,
        data,
        optimizer_state,
        key,
    )


# TODO: update output type hints when _get_mcmc_fns is made more general
def _setup_eval(
    config: ConfigDict,
    log_psi_apply: ModelApply[P],
    local_energy_fn: ModelApply[P],
    get_position_fn: GetPositionFromData[dwpa.DWPAData],
    apply_pmap: bool = True,
) -> Tuple[
    updates.params.UpdateParamFn[P, dwpa.DWPAData, OptimizerState],
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    eval_update_param_fn = updates.params.create_eval_update_param_fn(
        local_energy_fn,
        config.eval.nchains,
        get_position_fn,
        record_local_energies=config.eval.record_local_energies,
        nan_safe=config.eval.nan_safe,
        apply_pmap=apply_pmap,
    )
    eval_burning_step, eval_walker_fn = _get_mcmc_fns(
        config.eval, log_psi_apply, apply_pmap=apply_pmap
    )
    return eval_update_param_fn, eval_burning_step, eval_walker_fn


def _make_new_data_for_eval(
    config: ConfigDict,
    log_psi_apply: ModelApply[P],
    params: P,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    key: PRNGKey,
    dtype=jnp.float32,
) -> Tuple[PRNGKey, dwpa.DWPAData]:
    nelec_total = jnp.sum(nelec)
    # grab the first key if distributed
    key = utils.distribute.get_first_if_distributed(key)
    key, init_pos = physics.core.initialize_molecular_pos(
        key,
        config.eval.nchains,
        ion_pos,
        ion_charges,
        nelec_total,
        dtype=dtype,
    )
    # redistribute if needed
    if config.distribute:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)
    data = _make_initial_data(
        log_psi_apply, config.eval, init_pos, params, apply_pmap=config.distribute
    )

    return key, data


def _burn_and_run_vmc(
    run_config: ConfigDict,
    logdir: str,
    params: P,
    optimizer_state: S,
    data: D,
    burning_step: mcmc.metropolis.BurningStep[P, D],
    walker_fn: mcmc.metropolis.WalkerFn[P, D],
    update_param_fn: updates.params.UpdateParamFn[P, D, S],
    get_amplitude_fn: GetAmplitudeFromData[D],
    key: PRNGKey,
    should_checkpoint: bool = True,
) -> Tuple[P, S, D, PRNGKey]:
    if should_checkpoint:
        checkpoint_every = run_config.checkpoint_every
        best_checkpoint_every = run_config.best_checkpoint_every
        checkpoint_dir = run_config.checkpoint_dir
        checkpoint_variance_scale = run_config.checkpoint_variance_scale
        nhistory_max = run_config.nhistory_max
        checkpoint_if_nans = run_config.checkpoint_if_nans
        only_checkpoint_first_nans = run_config.only_checkpoint_first_nans
    else:
        checkpoint_every = None
        best_checkpoint_every = None
        checkpoint_dir = ""
        checkpoint_variance_scale = 0
        nhistory_max = 0
        checkpoint_if_nans = False
        only_checkpoint_first_nans = True

    data, key = mcmc.metropolis.burn_data(
        burning_step, run_config.nburn, params, data, key
    )
    params, optimizer_state, data, key = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        run_config.nchains,
        run_config.nepochs,
        walker_fn,
        update_param_fn,
        key,
        logdir=logdir,
        checkpoint_every=checkpoint_every,
        best_checkpoint_every=best_checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_variance_scale=checkpoint_variance_scale,
        checkpoint_if_nans=checkpoint_if_nans,
        only_checkpoint_first_nans=only_checkpoint_first_nans,
        record_amplitudes=run_config.record_amplitudes,
        get_amplitude_fn=get_amplitude_fn,
        nhistory_max=nhistory_max,
    )

    return params, optimizer_state, data, key


def _compute_and_save_energy_statistics(
    local_energies_file_path: str, output_dir: str, output_filename: str
) -> None:
    local_energies = np.loadtxt(local_energies_file_path)
    eval_statistics = mcmc.statistics.get_stats_summary(local_energies)
    eval_statistics = jax.tree_map(lambda x: float(x), eval_statistics)
    utils.io.save_dict_to_json(
        eval_statistics,
        output_dir,
        output_filename,
    )


def run_molecule() -> None:
    """Run VMC on a molecule."""
    reload_config, config = train.parse_config_flags.parse_flags(FLAGS)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(reload_config, config)
    _save_git_hash(logdir)

    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec = _get_electron_ion_config_as_arrays(
        config, dtype=dtype_to_use
    )

    key = jax.random.PRNGKey(config.initial_seed)

    (
        log_psi_apply,
        burning_step,
        walker_fn,
        local_energy_fn,
        update_param_fn,
        get_amplitude_fn,
        params,
        data,
        optimizer_state,
        key,
    ) = _setup_vmc(
        config,
        ion_pos,
        ion_charges,
        nelec,
        key,
        dtype=dtype_to_use,
        apply_pmap=config.distribute,
    )

    reload_from_checkpoint = (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_checkpoint_file
    )

    if reload_from_checkpoint:
        checkpoint_file_path = os.path.join(
            reload_config.logdir, reload_config.checkpoint_relative_file_path
        )
        directory, filename = os.path.split(checkpoint_file_path)
        _, data, params, optimizer_state, key = utils.io.reload_vmc_state(
            directory, filename
        )
        (
            data,
            params,
            optimizer_state,
            key,
        ) = utils.distribute.distribute_vmc_state_from_checkpoint(
            data, params, optimizer_state, key
        )

    params, optimizer_state, data, key = _burn_and_run_vmc(
        config.vmc,
        logdir,
        params,
        optimizer_state,
        data,
        burning_step,
        walker_fn,
        update_param_fn,
        get_amplitude_fn,
        key,
        should_checkpoint=True,
    )

    logging.info("Completed VMC! Evaluating")

    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post
    # evaluation
    eval_logdir = os.path.join(logdir, "eval")

    eval_update_param_fn, eval_burning_step, eval_walker_fn = _setup_eval(
        config,
        log_psi_apply,
        local_energy_fn,
        pacore.get_position_from_data,
        apply_pmap=config.distribute,
    )
    optimizer_state = None

    eval_and_vmc_nchains_match = config.vmc.nchains == config.eval.nchains
    if not config.eval.use_data_from_training or not eval_and_vmc_nchains_match:
        key, data = _make_new_data_for_eval(
            config,
            log_psi_apply,
            params,
            ion_pos,
            ion_charges,
            nelec,
            key,
            dtype=dtype_to_use,
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
        get_amplitude_fn,
        key,
        should_checkpoint=False,
    )

    # need to check for local_energy.txt because when config.eval.nepochs=0 the file is
    # not created regardless of config.eval.record_local_energies
    local_es_were_recorded = os.path.exists(
        os.path.join(eval_logdir, "local_energies.txt")
    )
    if config.eval.record_local_energies and local_es_were_recorded:
        local_energies_filepath = os.path.join(eval_logdir, "local_energies.txt")
        _compute_and_save_energy_statistics(
            local_energies_filepath, eval_logdir, "statistics"
        )


def vmc_statistics() -> None:
    """Calculate statistics from a VMC evaluation run and write them to disc."""
    parser = argparse.ArgumentParser(
        description="Calculate statistics from a VMC evaluation run and write them "
        "to disc."
    )
    parser.add_argument(
        "local_energies_file_path",
        type=str,
        help="File path to load local energies from",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        help="File path to which to write the output statistics. The '.json' suffix "
        "will be appended to the supplied path.",
    )
    args = parser.parse_args()

    output_dir, output_filename = os.path.split(os.path.abspath(args.output_file_path))
    _compute_and_save_energy_statistics(
        args.local_energies_file_path, output_dir, output_filename
    )
