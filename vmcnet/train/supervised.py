"""Entry points for running standard jobs."""
import datetime
import functools
import logging
import os
import subprocess
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

# import matplotlib.pyplot as plt
import optax
from absl import flags
from ml_collections import ConfigDict

import vmcnet.mcmc as mcmc
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    P,
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

    checkpoint_file_path = os.path.join(
        reload_config.logdir, reload_config.checkpoint_relative_file_path
    )
    directory, filename = os.path.split(checkpoint_file_path)
    _, _, wf_params, _, _ = utils.io.reload_vmc_state(directory, filename)

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

    sg_opt = optax.adam(config.surrogate.learning_rate)
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

    fpath = os.path.join(logdir, "train.txt")

    nepochs = config.vmc.nsupervised
    with open(fpath, "w") as f:
        for i in range(nepochs):
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

    logging.info("Done training surrogate! Terminating.")
