"""Shared SGD integration test for examples."""
import logging

import jax
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
from vmcnet.mcmc.position_amplitude_core import (
    distribute_position_amplitude_data,
    get_position_from_data,
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
    learning_rate,
    log_psi_model,
    local_energy_fn,
):
    """Run a VMC test with a very simple SGD optimizer and given model."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(log_psi_model.apply, std_move)

    # Define parameter updates
    def sgd_apply(grad, params, learning_rate):
        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            learning_rate,
        )

    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        log_psi_model.apply,
        local_energy_fn,
        nchains,
        sgd_apply,
        get_position_from_data,
    )

    # Distribute everything via jax.pmap
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_vmc_state(
        data, params, learning_rate, key, distribute_position_amplitude_data
    )

    # Train!
    with caplog.at_level(logging.INFO):
        params, _, _ = train.vmc.vmc_loop(
            params,
            optimizer_state,
            data,
            nchains,
            nburn,
            nepochs,
            nsteps_per_param_update,
            metrop_step_fn,
            update_param_fn,
            key,
        )
    return params
