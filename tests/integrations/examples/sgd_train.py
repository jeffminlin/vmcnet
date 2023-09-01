"""Shared SGD integration test for examples."""
import logging

import jax

import vmcnet.mcmc as mcmc
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
from vmcnet.mcmc.position_amplitude_core import (
    distribute_position_amplitude_data,
    get_position_from_data,
    get_update_data_fn,
)
from vmcnet.mcmc.simple_position_amplitude import (
    make_simple_pos_amp_gaussian_step,
)


def sgd_vmc_loop_with_logging(
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
    log_psi_model,
    local_energy_fn,
    should_distribute_data=True,
    logdir=None,
    checkpoint_every=None,
    checkpoint_dir=None,
    local_energy_type: str = "standard",
):
    """Run a VMC test with a very simple SGD optimizer and given model."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(log_psi_model.apply, std_move)

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    # Define parameter updates
    def sgd_apply(grad, params, learning_rate, data, aux):
        del data
        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            learning_rate,
        )

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_model.apply,
        local_energy_fn,
        nchains,
        local_energy_type=local_energy_type,
    )
    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        sgd_apply,
        get_position_from_data,
        get_update_data_fn(log_psi_model.apply),
    )

    # Distribute everything via jax.pmap
    if should_distribute_data:
        (
            data,
            params,
            optimizer_state,
            key,
        ) = utils.distribute.distribute_vmc_state(
            data, params, optimizer_state, key, distribute_position_amplitude_data
        )

    # Train!
    with caplog.at_level(logging.INFO):
        data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)
        params, optimizer_state, data, key, _ = train.vmc.vmc_loop(
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
            is_pmapped=should_distribute_data,
        )
        return data, params, optimizer_state, key
