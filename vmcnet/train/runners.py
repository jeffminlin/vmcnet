"""Entry points for running standard jobs."""
import logging
import os

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha
import optax
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import vmcnet.mcmc as mcmc
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.updates as updates
import vmcnet.utils as utils
import vmcnet.train as train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", os.path.join(os.path.dirname(__file__), "..", "utils", "config.py")
)


def molecule():
    """Run VMC and evaluate a molecule."""
    app.run(_molecule)


def _molecule(argv):
    config = FLAGS.config

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)

    logging.info("Hyperparameter configuration: \n%s", config)
    if config.logdir:
        logdir = config.logdir
        hparam_filename = utils.io.add_suffix_for_uniqueness(
            "hyperparams", logdir, pre_suffix=".json"
        )
        with utils.io.open_or_create(logdir, hparam_filename + ".json", "w") as f:
            f.write(config.to_json(indent=4))
    else:
        logdir = None

    dtype_to_use = jnp.float32
    if config.dtype == "float64":
        dtype_to_use = jnp.float64
        jax.config.update("jax_enable_x64", True)

    ion_pos = jnp.array(config.problem.ion_pos, dtype=dtype_to_use)
    ion_charges = jnp.array(config.problem.ion_charges, dtype=dtype_to_use)
    nelec = jnp.array(config.problem.nelec)
    nelec_total = jnp.sum(nelec)

    key = jax.random.PRNGKey(config.initial_seed)
    key, init_pos = physics.core.initialize_pos(
        key, config.vmc.nchains, ion_pos, ion_charges, nelec_total, dtype=dtype_to_use
    )

    log_psi = models.construct.get_model(
        config.model, nelec, ion_pos, dtype=dtype_to_use
    )

    key, subkey = jax.random.split(key)

    params = log_psi.init(subkey, init_pos[0:1])
    params = utils.distribute.replicate_all_local_devices(params)

    sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
    sharded_amplitudes = utils.distribute.pmap(log_psi.apply)(params, sharded_init_pos)
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=config.vmc.std_move, move_acceptance_sum=0.0, moves_since_update=0
        )
    )
    data = pacore.make_position_amplitude_data(
        sharded_init_pos, sharded_amplitudes, move_metadata
    )

    # Setup metropolis step
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi.apply,
        config.vmc.nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        config.vmc.nsteps_per_param_update, metrop_step_fn
    )

    # Define parameter updates
    kinetic_fn = physics.kinetic.create_continuous_kinetic_energy(log_psi.apply)
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

    # def clipped_local_energy_fn(params, x):
    #     local_energy = local_energy_fn(params, x)
    #     median_local_e = jnp.nanmedian(local_energy)
    #     local_energy = jnp.where(
    #         jnp.isfinite(local_energy),
    #         local_energy,
    #         median_local_e,
    #     )
    #     total_variation = utils.distribute.mean_all_local_devices(
    #         jnp.abs(local_energy - median_local_e)
    #     )
    #     clipped_local_e = jax.lax.cond(
    #         ~jnp.isnan(total_variation),
    #         lambda x: jnp.clip(
    #             x,
    #             median_local_e - 5.0 * total_variation,
    #             median_local_e + 5.0 * total_variation,
    #         ),
    #         lambda x: x,
    #         local_energy,
    #     )
    #     return clipped_local_e

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi.apply, local_energy_fn, config.vmc.nchains
    )

    if config.vmc.schedule_type == "constant":

        def learning_rate_schedule(t):
            return config.vmc.learning_rate

    elif config.vmc.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return config.vmc.learning_rate / (1.0 + config.vmc.learning_decay_rate * t)

    sharded_key = utils.distribute.make_different_rng_key_on_all_devices(key)

    if config.vmc.optimizer_type == "kfac":
        optimizer = kfac_ferminet_alpha.Optimizer(
            energy_data_val_and_grad,
            l2_reg=0.0,
            norm_constraint=0.001,
            value_func_has_aux=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=0.95,
            inverse_update_period=1,
            min_damping=1e-4,
            num_burnin_steps=0,
            register_only_generic=False,
            estimation_mode="fisher_exact",
            multi_device=True,
            pmap_axis_name=utils.distribute.PMAP_AXIS_NAME,
        )
        sharded_key, subkeys = utils.distribute.p_split(sharded_key)
        optimizer_state = optimizer.init(
            params, subkeys, pacore.get_position_from_data(data)
        )

        update_param_fn = updates.params.create_kfac_update_param_fn(
            optimizer, 0.001, pacore.get_position_from_data
        )
    elif config.vmc.optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=learning_rate_schedule)
        optimizer_state = optimizer.init(params)

        def optimizer_apply(grad, params, optimizer_state):
            updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            return params, optimizer_state

        update_param_fn = updates.params.create_grad_energy_update_param_fn(
            log_psi.apply,
            local_energy_fn,
            config.vmc.nchains,
            optimizer_apply,
            get_position_fn=pacore.get_position_from_data,
        )

    # Train
    params, optimizer_state, data, sharded_key = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        config.vmc.nchains,
        config.vmc.nepochs,
        walker_fn,
        update_param_fn,
        sharded_key,
        logdir=config.logdir if config.logdir else None,
        checkpoint_every=config.vmc.checkpoint_every,
        best_checkpoint_every=config.vmc.best_checkpoint_every,
        checkpoint_dir=config.vmc.checkpoint_dir,
        checkpoint_variance_scale=config.vmc.checkpoint_variance_scale,
        nhistory_max=config.vmc.nhistory_max,
    )

    logging.info("Completed VMC! Evaluating")

    eval_update_param_fn = updates.params.create_eval_update_param_fn(
        local_energy_fn, config.eval.nchains, pacore.get_position_from_data
    )
    optimizer_state = None

    if not config.eval.use_data_from_training:
        sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
        sharded_amplitudes = utils.distribute.pmap(log_psi.apply)(
            params, sharded_init_pos
        )
        move_metadata = utils.distribute.replicate_all_local_devices(
            dwpa.MoveMetadata(
                std_move=config.vmc.std_move,
                move_acceptance_sum=0.0,
                moves_since_update=0,
            )
        )
        data = pacore.make_position_amplitude_data(
            sharded_init_pos, sharded_amplitudes, move_metadata
        )

    params, optimizer_state, data, sharded_key = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        config.eval.nchains,
        config.eval.nburn,
        config.eval.nepochs,
        config.eval.nsteps_per_energy_eval,
        metrop_step_fn,
        eval_update_param_fn,
        sharded_key,
        logdir=os.path.join(config.logdir, "eval") if config.logdir else None,
        checkpoint_every=None,
        best_checkpoint_every=None,
        nhistory_max=0,
    )
