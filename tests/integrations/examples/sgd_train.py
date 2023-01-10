"""Shared SGD integration test for examples."""
import jax
import jax.numpy as jnp

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
    data,
    params,
    key,
    nchains,
    nburn,
    nepochs,
    nsteps_per_param_update,
    std_move,
    init_learning_rate,
    log_psi_model,
    local_energy_fn,
    should_distribute_data=True,
    logdir=None,
    checkpoint_every=None,
    checkpoint_dir=None,
    learning_rate_strat="constant",
):
    """Run a VMC test with a very simple SGD optimizer and given model."""
    # Setup metropolis step
    metrop_step_fn = make_simple_pos_amp_gaussian_step(log_psi_model.apply, std_move)

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    # Define parameter updates
    def sgd_apply(grad, params, opt_state, data):
        del data
        epoch = opt_state["epoch"]
        learning_rate = opt_state["learning_rate"]
        if learning_rate_strat == "constant":
            opt_state = {"epoch": epoch + 1, "learning_rate": init_learning_rate}
        else:
            opt_state = {
                "epoch": epoch + 1,
                "learning_rate": init_learning_rate / (1 + epoch / 10),
            }

        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            opt_state,
        )

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_model.apply, local_energy_fn, nchains
    )
    update_param_fn = updates.params.create_grad_energy_update_param_fn(
        energy_data_val_and_grad,
        sgd_apply,
        get_position_from_data,
        get_update_data_fn(log_psi_model.apply),
    )

    opt_state = {"epoch": 0, "learning_rate": init_learning_rate}
    # Distribute everything via jax.pmap
    if should_distribute_data:
        (data, params, opt_state, key,) = utils.distribute.distribute_vmc_state(
            data, params, opt_state, key, distribute_position_amplitude_data
        )

    data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)

    print(opt_state)
    params, opt_state, data, key, _ = train.vmc.vmc_loop(
        params,
        opt_state,
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
    print(opt_state)
    return data, params, opt_state, key
