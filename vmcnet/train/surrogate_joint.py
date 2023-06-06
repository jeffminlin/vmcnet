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

# import matplotlib.pyplot as plt
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


def _assemble_mol_local_energy_fn(
    local_energy_type: str,
    local_energy_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    ei_softening: chex.Scalar,
    ee_softening: chex.Scalar,
    log_psi_apply: ModelApply[P],
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
    else:
        raise ValueError(
            f"Requested local energy type {local_energy_type} is not supported"
        )


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
            config.vmc,
            config.problem,
            ion_pos,
            ion_charges,
            log_psi_apply,
        )
    )

    pretrain_nepochs = config.vmc.npretrain_wf
    LR = config.vmc.optimizer.adam.learning_rate
    sg_LR = config.surrogate.learning_rate

    data, key = mcmc.metropolis.burn_data(
        burning_step, config.vmc.nburn, wf_params, data, key
    )
    update_data_fn = jax.jit(pacore.get_update_data_fn(log_psi_apply))

    wf_opt = optax.adam(LR)
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

    fpath = os.path.join(logdir, "pretrain_wf.txt")
    with open(fpath, "w") as f:
        logging.info("Pretraining WF with single particle method.")
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
            f.write(f"{energy_noclip} {variance_noclip}\n")

    logging.info("Done pretraining WF! Now pretraining surrogate with MSQE loss.")

    wf_energies_and_perms_fn = physics.random_particle.create_wf_energies_and_perms_fn(
        log_psi_apply,
        ion_pos,
        ion_charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
    )
    wf_energies_and_perms_fn = jax.jit(wf_energies_and_perms_fn)

    sg_val_and_grad_fn = physics.random_particle.create_sg_val_and_grad_fn(surrogate)
    sg_val_and_grad_fn = jax.jit(sg_val_and_grad_fn)

    sg_opt = optax.adam(sg_LR)
    sg_opt_state = sg_opt.init(sg_params)

    def sg_train_iteration(
        data,
        key,
        sg_opt_state,
        sg_params,
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

        msqe, grad_sg = sg_val_and_grad_fn(
            sg_params, position, single_particle_energies, perms
        )
        updates, sg_opt_state = sg_opt.update(grad_sg, sg_opt_state)
        sg_params = optax.apply_updates(sg_params, updates)
        return accept_ratio, data, key, msqe, sg_opt_state, sg_params

    sg_train_iteration = jax.jit(sg_train_iteration)

    fpath = os.path.join(logdir, "pretrain_sg.txt")

    pretrain_nepochs = config.vmc.npretrain_sg
    with open(fpath, "w") as f:
        for i in range(pretrain_nepochs):
            (
                accept_ratio,
                data,
                key,
                msqe,
                sg_opt_state,
                sg_params,
            ) = sg_train_iteration(
                data,
                key,
                sg_opt_state,
                sg_params,
                wf_params,
            )

            logging.info(
                f"Epoch {i:5d}   MSQE {msqe:3e}   Accept ratio: {accept_ratio:3f}"
            )
            f.write(f"{msqe}\n")

    logging.info(
        "Done pretraining surrogate! Running joint VMC optimization of WF and SG."
    )
    time.sleep(3)
    clipping_fn = _get_clipping_fn(config.vmc)

    wf_energy_val_and_grad_fn = physics.core.create_surrogate_value_and_grad_energy_fn(
        log_psi_apply,
        surrogate,
        config.vmc.nchains,
        clipping_fn,
        nan_safe=config.vmc.nan_safe,
    )
    wf_energy_val_and_grad_fn = jax.jit(wf_energy_val_and_grad_fn)

    def wf_iteration_with_surrogate(
        data,
        key,
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
            wf_opt_state,
            wf_params,
        )

    wf_iteration_with_surrogate = jax.jit(wf_iteration_with_surrogate)

    fpath = os.path.join(logdir, "train.txt")
    with open(fpath, "w") as f:
        for i in range(config.vmc.nepochs):
            (
                accept_ratio,
                data,
                key,
                msqe,
                sg_opt_state,
                sg_params,
            ) = sg_train_iteration(
                data,
                key,
                sg_opt_state,
                sg_params,
                wf_params,
            )

            (
                accept_ratio,
                aux_data,
                data,
                key,
                wf_opt_state,
                wf_params,
            ) = wf_iteration_with_surrogate(
                data,
                key,
                sg_params,
                wf_opt_state,
                wf_params,
            )

            energy_noclip = aux_data[2]
            variance_noclip = aux_data[3]
            logging.info(
                f"Epoch {i:5d}   Energy {energy_noclip:3e}   Variance {variance_noclip:3e}   MSQE {msqe:3e}   Accept ratio: {accept_ratio:3f}"
            )
            f.write(f"{energy_noclip} {variance_noclip} {msqe}\n")

    io.save_vmc_state(
        config.logdir,
        "train_checkpoint.npz",
        (-1, data, {"wf": wf_params, "sg": sg_params}, wf_opt_state, key),
    )

    logging.info("Completed VMC! Running evalution of WF energy and variance.")
    standard_local_energy_fn = _assemble_mol_local_energy_fn(
        "standard",
        config.vmc.local_energy,
        ion_pos,
        ion_charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi_apply,
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

    logging.info("Completed evaluation run! Saving statistics and terminating.")

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
