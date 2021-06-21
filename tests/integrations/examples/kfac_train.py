"""KFAC integration test runner for examples."""
import logging

import jax.numpy as jnp
import kfac_ferminet_alpha
from kfac_ferminet_alpha import utils as kfac_utils

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.mcmc.position_amplitude_core import (
    distribute_position_amplitude_data,
    get_position_from_data,
)
from vmcnet.mcmc.simple_position_amplitude import (
    make_simple_pos_amp_gaussian_step,
)


def kfac_vmc_loop_with_logging(
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
    log_psi_model,
    local_energy_fn,
    should_distribute_data=True,
    logdir=None,
    checkpoint_every=None,
    checkpoint_dir=None,
):
    """Run a VMC test with the KFAC optimizer and given model."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(log_psi_model.apply, std_move)

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    # Define parameter updates
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_model.apply, local_energy_fn, nchains
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
    momentum = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    damping = kfac_utils.replicate_all_local_devices(jnp.asarray(0.001))

    def update_param_fn(data, params, optimizer_state, key):
        key, subkey = utils.distribute.p_split(key)
        params, optimizer_state, stats = optimizer.step(
            params=params,
            state=optimizer_state,
            rng=subkey,
            data_iterator=iter([get_position_from_data(data)]),
            momentum=momentum,
            damping=damping,
        )
        energy = utils.distribute.get_first(stats["loss"])
        variance = utils.distribute.get_first(stats["aux"][0])
        metrics = {"energy": energy, "variance": variance}
        return params, optimizer_state, metrics, key

    # Distribute everything via jax.pmap
    if should_distribute_data:
        (data, params, _, key,) = utils.distribute.distribute_vmc_state(
            data, params, None, key, distribute_position_amplitude_data
        )

    key, subkey = utils.distribute.p_split(key)
    optimizer_state = optimizer.init(params, subkey, get_position_from_data(data))

    # Train!
    with caplog.at_level(logging.INFO):
        data, key = mcmc.metropolis.burn_data(burning_step, nburn, data, params, key)
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
            checkpoint_every,
            checkpoint_dir,
        )
        return data, params, optimizer_state, key
