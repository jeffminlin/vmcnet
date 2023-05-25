"""Entry points for running standard jobs."""
import datetime
import functools
import logging
import os
import subprocess
import time
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from absl import flags
from ml_collections import ConfigDict

import vmcnet.utils.io as io
import vmcnet.mcmc as mcmc
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    P,
    ClippingFn,
    LocalEnergyApply,
    ModelApply,
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


def _get_dtype(config: ConfigDict):
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


def _get_mcmc_fns(
    run_config: ConfigDict, log_psi_apply: ModelApply[P], apply_pmap: bool = False
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
    local_energy_type: str,
    local_energy_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    ei_softening: chex.Scalar,
    ee_softening: chex.Scalar,
    log_psi_apply: ModelApply[P],
    surrogate: ModelApply[P],
) -> LocalEnergyApply[P]:
    if local_energy_type == "standard":
        kinetic_fn = physics.kinetic.create_laplacian_kinetic_energy(log_psi_apply)
        ei_potential_fn = physics.potential.create_electron_ion_coulomb_potential(
            ion_pos, ion_charges, softening_term=ei_softening
        )
        ee_potential_fn = physics.potential.create_electron_electron_coulomb_potential(
            softening_term=ee_softening
        )
        ii_potential_fn = physics.potential.create_ion_ion_coulomb_potential(
            ion_pos, ion_charges
        )
        local_energy_fn: LocalEnergyApply[P] = physics.core.combine_local_energy_terms(
            [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
        )
        return local_energy_fn
    if local_energy_type == "ibp":
        ibp_parts = local_energy_config.ibp.ibp_parts
        local_energy_fn = physics.ibp.create_ibp_local_energy(
            log_psi_apply,
            ion_pos,
            ion_charges,
            "kinetic" in ibp_parts,
            "ei" in ibp_parts,
            ei_softening,
            "ee" in ibp_parts,
            ee_softening,
        )
        return local_energy_fn
    elif local_energy_type == "random_particle":
        nparticles = local_energy_config.random_particle.nparticles
        sample_parts = local_energy_config.random_particle.sample_parts
        local_energy_fn = (
            physics.random_particle.create_molecular_random_particle_local_energy(
                log_psi_apply,
                ion_pos,
                ion_charges,
                nparticles,
                "kinetic" in sample_parts,
                "ei" in sample_parts,
                ei_softening,
                "ee" in sample_parts,
                ee_softening,
            )
        )
        return local_energy_fn
    elif local_energy_type == "surrogate":
        nparticles = local_energy_config.random_particle.nparticles
        sample_parts = local_energy_config.random_particle.sample_parts
        local_energy_fn = (
            physics.random_particle.create_molecular_surrogate_local_energy(
                log_psi_apply,
                surrogate,
                ion_pos,
                ion_charges,
                nparticles,
                "kinetic" in sample_parts,
                "ei" in sample_parts,
                ei_softening,
                "ee" in sample_parts,
                ee_softening,
            )
        )
        return local_energy_fn
    else:
        raise ValueError(
            f"Requested local energy type {local_energy_type} is not supported"
        )


# TODO: figure out where this should go -- the act of clipping energies is kind of just
# a training trick rather than a physics thing, so maybe this stays here
def total_variation_clipping_fn(
    local_energies: Array,
    energy_noclip: chex.Numeric,
    threshold=5.0,
    clip_center="mean",
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


def _get_random_particle_energy_val_and_grad_fn(
    vmc_config: ConfigDict,
    problem_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    log_psi_apply: ModelApply[P],
) -> physics.core.ValueGradEnergyFn[P]:
    ei_softening = problem_config.ei_softening
    ee_softening = problem_config.ee_softening

    local_energy_fn = _assemble_mol_local_energy_fn(
        "random_particle",
        vmc_config.local_energy,
        ion_pos,
        ion_charges,
        ei_softening,
        ee_softening,
        log_psi_apply,
        None,
    )

    clipping_fn = _get_clipping_fn(vmc_config)

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply,
        local_energy_fn,
        vmc_config.nchains,
        clipping_fn,
        nan_safe=vmc_config.nan_safe,
        local_energy_type="random_particle",
    )

    return energy_data_val_and_grad


def _get_standard_val_and_grad_fn(
    vmc_config: ConfigDict,
    problem_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    log_psi_apply: ModelApply[P],
) -> physics.core.ValueGradEnergyFn[P]:
    ei_softening = problem_config.ei_softening
    ee_softening = problem_config.ee_softening

    local_energy_fn = _assemble_mol_local_energy_fn(
        "standard",
        vmc_config.local_energy,
        ion_pos,
        ion_charges,
        ei_softening,
        ee_softening,
        log_psi_apply,
        None,
    )

    clipping_fn = _get_clipping_fn(vmc_config)

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply,
        local_energy_fn,
        vmc_config.nchains,
        clipping_fn,
        nan_safe=vmc_config.nan_safe,
        local_energy_type="standard",
    )

    return energy_data_val_and_grad


def run_molecule() -> None:
    """Run VMC on a molecule."""
    reload_config, config = train.parse_config_flags.parse_flags(FLAGS)
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(reload_config, config)
    _save_git_hash(logdir)

    dtype = _get_dtype(config)

    ion_pos, ion_charges, nelec = _get_electron_ion_config_as_arrays(
        config, dtype=dtype
    )

    key = jax.random.PRNGKey(config.initial_seed)

    nelec_total = int(jnp.sum(nelec))
    key, init_pos = physics.core.initialize_molecular_pos(
        key, config.vmc.nchains, ion_pos, ion_charges, nelec_total, dtype=dtype
    )

    # Make the model
    slog_psi = models.construct.get_model_from_config(
        config.model, nelec, ion_pos, ion_charges, dtype=dtype
    )
    key, subkey = jax.random.split(key)
    wf_params = slog_psi.init(subkey, init_pos[0:1])

    log_psi_apply = models.construct.slog_psi_to_log_psi_apply(slog_psi.apply)

    spin_split = models.construct.get_spin_split(nelec)

    compute_input_streams = models.construct.get_compute_input_streams_from_config(
        config.surrogate.input_streams, ion_pos
    )

    backflow = models.construct.get_backflow_from_config(
        config.surrogate.backflow,
        spin_split,
        dtype=dtype,
    )

    surrogate_module = models.construct.FermiNetSurrogate(
        spin_split, compute_input_streams, backflow
    )
    key, subkey = jax.random.split(key)
    sg_params = surrogate_module.init(subkey, init_pos[0:1])
    surrogate = surrogate_module.apply

    # Make initial data
    data = _make_initial_single_device_data(
        log_psi_apply, config.vmc, init_pos, wf_params
    )

    # Setup metropolis step
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi_apply,
        config.vmc.nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )
    burning_step = mcmc.metropolis.make_jitted_burning_step(
        metrop_step_fn, apply_pmap=False
    )
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        config.vmc.nsteps_per_param_update,
        metrop_step_fn,
        apply_pmap=False,
    )

    random_particle_energy_data_val_and_grad = jax.jit(
        _get_random_particle_energy_val_and_grad_fn(
            config.vmc, config.problem, ion_pos, ion_charges, log_psi_apply
        )
    )

    pretrain_nepochs = config.vmc.npretrain_wf
    LR = config.vmc.optimizer.sgd.learning_rate
    sg_LR = config.surrogate.learning_rate

    logging.info(f"WF LR {LR}, SG LR {sg_LR}")

    data, key = mcmc.metropolis.burn_data(
        burning_step, config.vmc.nburn, wf_params, data, key
    )
    update_data_fn = jax.jit(pacore.get_update_data_fn(log_psi_apply))

    wf_opt = optax.sgd(LR)
    wf_opt_state = wf_opt.init(wf_params)

    def wf_pretrain_iteration(
        data,
        key,
        wf_opt_state,
        wf_params,
    ):
        accept_ratio, data, key = walker_fn(wf_params, data, key)
        position = data["walker_data"]["position"]
        key, subkey = jax.random.split(key)
        (_, aux_data), grad_e = random_particle_energy_data_val_and_grad(
            wf_params, subkey, position
        )
        updates, wf_opt_state = wf_opt.update(grad_e, wf_opt_state)
        wf_params = optax.apply_updates(wf_params, updates)
        data = update_data_fn(data, wf_params)
        return accept_ratio, aux_data, data, key, wf_opt_state, wf_params

    wf_pretrain_iteration = jax.jit(wf_pretrain_iteration)

    logging.info("Pretraining WF")
    for i in range(pretrain_nepochs):
        (
            accept_ratio,
            aux_data,
            data,
            key,
            wf_opt_state,
            wf_params,
        ) = wf_pretrain_iteration(
            data,
            key,
            wf_opt_state,
            wf_params,
        )

        energy_noclip = aux_data[2]
        variance_noclip = aux_data[3]
        logging.info(
            f" Epoch {i:5d}   Energy {energy_noclip:3e}   Variance {variance_noclip:3e}   Accept ratio: {accept_ratio:3f}"
        )

    logging.info("Done pretraining WF! Now pretraining surrogate")

    wf_energies_and_perms_fn = physics.random_particle.create_wf_energies_and_perms_fn(
        log_psi_apply, ion_pos, ion_charges
    )
    wf_energies_and_perms_fn = jax.jit(wf_energies_and_perms_fn)

    sg_val_and_grad_fn = physics.random_particle.create_sg_val_and_grad_fn(surrogate)
    sg_val_and_grad_fn = jax.jit(sg_val_and_grad_fn)

    clipping_fn = _get_clipping_fn(config.vmc)

    wf_energy_val_and_grad_fn = physics.core.create_surrogate_value_and_grad_energy_fn(
        log_psi_apply,
        surrogate,
        config.vmc.nchains,
        clipping_fn,
        nan_safe=config.vmc.nan_safe,
    )
    wf_energy_val_and_grad_fn = jax.jit(wf_energy_val_and_grad_fn)

    sg_opt = optax.adam(sg_LR)
    sg_opt_state = sg_opt.init(sg_params)

    def surrogate_pretrain_iteration(
        data,
        key,
        sg_opt_state,
        sg_params,
    ):
        accept_ratio, data, key = walker_fn(wf_params, data, key)
        position = data["walker_data"]["position"]
        key = jax.random.split(key, config.vmc.nchains + 1)
        subkey = key[1:]
        key = key[0]
        (
            local_energies_noclip,
            single_particle_energies,
            perms,
        ) = wf_energies_and_perms_fn(wf_params, position, subkey)
        msqe, grad_sg = sg_val_and_grad_fn(
            sg_params, position, single_particle_energies, perms
        )
        updates, sg_opt_state = sg_opt.update(grad_sg, sg_opt_state)
        sg_params = optax.apply_updates(sg_params, updates)
        return accept_ratio, data, key, msqe, sg_opt_state, sg_params

    surrogate_pretrain_iteration = jax.jit(surrogate_pretrain_iteration)

    pretrain_nepochs = config.vmc.npretrain_sg
    for i in range(pretrain_nepochs):
        (
            accept_ratio,
            data,
            key,
            msqe,
            sg_opt_state,
            sg_params,
        ) = surrogate_pretrain_iteration(
            data,
            key,
            sg_opt_state,
            sg_params,
        )

        logging.info(f"Epoch {i:5d}   MSQE {msqe:3e}   Accept ratio: {accept_ratio:3f}")

    logging.info("Done pretraining surrogate! Running main VMC optimization")
    time.sleep(3)

    def vmc_iteration(
        data,
        key,
        sg_opt_state,
        sg_params,
        wf_opt_state,
        wf_params,
    ):
        accept_ratio, data, key = walker_fn(wf_params, data, key)
        position = data["walker_data"]["position"]
        key = jax.random.split(key, config.vmc.nchains + 1)
        subkey = key[1:]
        key = key[0]
        (
            local_energies_noclip,
            single_particle_energies,
            perms,
        ) = wf_energies_and_perms_fn(wf_params, position, subkey)
        msqe = 0.0
        for j in range(config.surrogate.nsteps_per_wf_update):
            msqe, grad_sg = sg_val_and_grad_fn(
                sg_params, position, single_particle_energies, perms
            )
            updates, sg_opt_state = sg_opt.update(grad_sg, sg_opt_state)
            sg_params = optax.apply_updates(sg_params, updates)
        (energy, aux_data), grad_wf = wf_energy_val_and_grad_fn(
            wf_params, sg_params, local_energies_noclip, position, perms
        )
        updates, wf_opt_state = wf_opt.update(grad_wf, wf_opt_state)
        wf_params = optax.apply_updates(wf_params, updates)
        data = update_data_fn(data, wf_params)
        return (
            accept_ratio,
            aux_data,
            data,
            key,
            sg_opt_state,
            msqe,
            sg_params,
            wf_opt_state,
            wf_params,
        )

    vmc_iteration = jax.jit(vmc_iteration)

    for i in range(config.vmc.nepochs):
        (
            accept_ratio,
            aux_data,
            data,
            key,
            sg_opt_state,
            msqe,
            sg_params,
            wf_opt_state,
            wf_params,
        ) = vmc_iteration(
            data,
            key,
            sg_opt_state,
            sg_params,
            wf_opt_state,
            wf_params,
        )

        energy_noclip = aux_data[2]
        variance_noclip = aux_data[3]
        logging.info(
            f"Epoch {i:5d}   Energy {energy_noclip:3e}   Variance {variance_noclip:3e}   MSQE {msqe:3e}   Accept ratio: {accept_ratio:3f}"
        )

    io.save_vmc_state(
        config.logdir,
        "checkpoint.npz",
        (-1, data, {"wf": wf_params, "sg": sg_params}, wf_opt_state, key),
    )

    logging.info("Completed VMC! Running evaluation.")

    standard_local_energy_fn = _assemble_mol_local_energy_fn(
        "standard",
        config.vmc.local_energy,
        ion_pos,
        ion_charges,
        0.0,
        0.0,
        log_psi_apply,
        None,
    )
    standard_local_energy_fn = jax.jit(
        jax.vmap(standard_local_energy_fn, in_axes=(None, 0, None), out_axes=0)
    )

    data, key = mcmc.metropolis.burn_data(
        burning_step, config.vmc.nburn, wf_params, data, key
    )

    energy_sum = 0.0
    variance_sum = 0.0

    def eval_iteration(
        data,
        energy_sum,
        key,
        variance_sum,
    ):
        accept_ratio, data, key = walker_fn(wf_params, data, key)
        position = data["walker_data"]["position"]
        local_energies = standard_local_energy_fn(wf_params, position, None)
        energy = jnp.mean(local_energies)
        variance = jnp.mean(local_energies**2 - jnp.mean(local_energies) ** 2)
        energy_sum += energy
        variance_sum += variance
        return accept_ratio, energy, energy_sum, variance, variance_sum, key, data

    eval_iteration = jax.jit(eval_iteration)

    for i in range(config.eval.nepochs):
        (
            accept_ratio,
            energy,
            energy_sum,
            variance,
            variance_sum,
            key,
            data,
        ) = eval_iteration(
            data,
            energy_sum,
            key,
            variance_sum,
        )
        logging.info(
            f"Epoch {i:5d}   Energy {energy:3e}   Variance {variance:3e}   Accept ratio: {accept_ratio:3f}"
        )

    energy_mean = energy_sum / config.eval.nepochs
    variance_mean = variance_sum / config.eval.nepochs
    utils.io.save_dict_to_json(
        {
            "energy": float(energy_mean),
            "variance": float(variance_mean),
            "stderr": float(
                jnp.sqrt(variance_mean / (config.eval.nepochs * config.vmc.nchains))
            ),
        },
        config.logdir,
        "eval_statistics",
    )
