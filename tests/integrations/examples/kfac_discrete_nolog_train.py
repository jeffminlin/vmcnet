#"""KFAC integration test runner for examples."""
import logging

import kfac_ferminet_alpha

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.physics as physics
import vmcnet.updates as updates
import vmcnet.utils as utils
import jax.numpy as jnp
import jax
from vmcnet.mcmc.position_amplitude_core import (
    distribute_position_amplitude_data,
    get_position_from_data,
)
from vmcnet.mcmc.simple_position_amplitude import (
    make_simple_pos_amp_gaussian_step,
)


def kfac_discrete_vmc_loop_with_logging(
        caplog,
        data,
        params,
        key,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        learning_rate,
        psi_model,
        local_energy_fn,
        side_length,
        should_distribute_data=True,
        logdir=None,
        checkpoint_every=None,
        checkpoint_dir=None,
):
    """Run a VMC test with the KFAC optimizer and given model."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(
        psi_model.apply, std_move, logabs=False, discrete=True, cyclic=side_length
    )

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    log_psi_apply=lambda _params, _x : jnp.log(jnp.abs(psi_model.apply(_params, _x)))

    # Define parameter updates
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    def learning_rate_schedule(t):
        return learning_rate

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

    update_param_fn = updates.params.create_kfac_update_param_fn(
        optimizer, 0.001, get_position_from_data
    )

    #Distribute everything via jax.pmap
    if should_distribute_data:
        (data, params, _, key,) = utils.distribute.distribute_vmc_state(
            data, params, None, key, distribute_position_amplitude_data
        )

    key, subkey = utils.distribute.p_split(key)
    #key, subkey = jax.random.split(key)

    optimizer_state = optimizer.init(params, subkey, get_position_from_data(data))

    # Train!
    with caplog.at_level(logging.INFO):
        data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)
        params, optimizer_state, data, key = train.vmc.vmc_loop(
            params,
            optimizer_state,
            data,
            nchains,
            nepochs,
            walker_fn,
            update_param_fn,
            key,
            logdir,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )
        return data, params, optimizer_state, key
