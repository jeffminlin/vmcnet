"""Shared SGD integration test for examples."""
import logging

import jax

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils


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
    metrop_step_fn = mcmc.metropolis.make_position_amplitude_gaussian_metropolis_step(
        std_move, log_psi_model.apply
    )

    # Define parameter updates
    def sgd_apply(grad, learning_rate, params):
        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            learning_rate,
        )

    update_param_fn = updates.params.create_position_amplitude_data_update_param_fn(
        log_psi_model.apply, local_energy_fn, nchains, sgd_apply
    )

    # Distribute everything via jax.pmap
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_data_params_optstate_and_key(
        data, params, learning_rate, key
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
