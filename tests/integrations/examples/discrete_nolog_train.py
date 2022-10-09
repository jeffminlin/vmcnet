"""test runner for Hubbard example."""
import logging

import jax

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.physics as physics
import vmcnet.updates as updates
import vmcnet.utils as utils
import jax.numpy as jnp
from vmcnet.mcmc.position_amplitude_core import (
    distribute_position_amplitude_data,
    get_position_from_data,
)
from vmcnet.mcmc.simple_position_amplitude import (
    make_simple_pos_amp_gaussian_step,
)


def discrete_vmc_loop_with_logging(
        caplog,
        data,
        params,
        key,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        std_move,
        optimizer_state,
        psi_model,
        local_energy_fn,
        side_length,
        should_distribute_data=True,
        logdir=None,
        checkpoint_every=None,
        checkpoint_dir=None,
):
    """Run a VMC test."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(
        psi_model.apply, std_move, logabs=False, discrete=True, cyclic=side_length
    )

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    def sgd_apply(grad, params, learning_rate, data):
        del data
        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            learning_rate,
        )

    log_psi_apply=lambda _params, _x : jnp.log(jnp.abs(psi_model.apply(_params, _x)))

    # Define parameter updates
    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply, local_energy_fn, nchains
    )

    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        sgd_apply,
        get_position_from_data,
    )

    #Distribute everything via jax.pmap
    if should_distribute_data:
        (data, params, optimizer_state, key,) = utils.distribute.distribute_vmc_state(
            data, params, optimizer_state, key, distribute_position_amplitude_data
        )

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
