"""Testing main VMC routine."""
import logging

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha.utils as kfac_utils
import numpy as np

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.utils as utils

from ..mcmc.test_metropolis import (
    _make_dummy_data_params_and_key,
    _make_dummy_metropolis_fn,
)


def _make_different_pmappable_data(data):
    """Adding (0, 1, ..., ndevices - 1) to the data and concatenating."""
    ndevices = jax.local_device_count()
    return jnp.concatenate([data + i for i in range(ndevices)])


def test_walk_data():
    """Test a few Metropolis steps.

    Test that taking a few Metropolis steps is equivalent to skipping to the end and
    taking one big step. Specifically, with a single proposal fn which adds a constant
    array at each step, test that taking a few steps is equivalent to adding that
    multiple of the proposal array directly (where the moves are accepted).
    """
    nsteps = 6
    data, params, key = _make_dummy_data_params_and_key()
    metrop_step_fn = _make_dummy_metropolis_fn()
    accept_prob, new_data, _ = train.vmc.walk_data(
        nsteps, data, params, key, metrop_step_fn
    )

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, jnp.array([nsteps, 0, 3 * nsteps, 0]))


def test_vmc_loop_logging(caplog):
    """Test vmc_loop logging. Uses pytest's caplog fixture to capture logs."""
    nburn = 4
    nepochs = 13  # eventual number of parameter updates
    nsteps_per_param_update = 10

    fixed_metrics = {
        "energy": 1.0,
        "energy_noclip": 2.5,
        "variance": 3.0,
        "variance_noclip": np.pi,
    }

    def update_param_fn(data, params, optimizer_state):
        del data
        return params, optimizer_state, fixed_metrics

    for pmapped in [True, False]:
        caplog.clear()
        data, params, key = _make_dummy_data_params_and_key()
        metrop_step_fn = _make_dummy_metropolis_fn()
        nchains = data.shape[0]

        if pmapped:
            data = _make_different_pmappable_data(data)
            (
                data,
                params,
                optimizer_state,
                key,
            ) = utils.distribute.distribute_data_params_optstate_and_key(
                data, params, None, key
            )

        with caplog.at_level(logging.INFO):
            train.vmc.vmc_loop(
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
                pmapped=pmapped,
            )

        # 1 line for burning, nepochs lines for training
        assert len(caplog.records) == 1 + nepochs


def test_vmc_loop_number_of_updates():
    """Test number of updates.

    Make sure that data is updated nsteps_per_param_update * nepochs + nburn times and
    params is updated nepoch times.
    """
    data, params, key = _make_dummy_data_params_and_key()
    metrop_step_fn = _make_dummy_metropolis_fn()
    nchains = data.shape[0]

    data = _make_different_pmappable_data(data)
    # storing the number of parameter updates in optimizer_state, replicated on each
    # device; with a real optimizer this is probably something more exciting and
    # possibly data-dependent (e.g. KFAC/Adam's running metrics)
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_data_params_optstate_and_key(data, params, 0, key)

    nburn = 5
    nepochs = 17  # eventual number of parameter updates
    nsteps_per_param_update = 2

    def update_param_fn(data, params, optimizer_state):
        del data
        optimizer_state += 1
        return params, optimizer_state, None

    _, new_optimizer_state, new_data = train.vmc.vmc_loop(
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

    new_optimizer_state = kfac_utils.get_first(new_optimizer_state)

    num_updates = nsteps_per_param_update * nepochs + nburn

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    for device_index in range(jax.local_device_count()):
        np.testing.assert_allclose(
            new_data[device_index],
            jnp.array([num_updates, 0, 3 * num_updates, 0]) + device_index,
        )


def test_vmc_loop_newtons_x_squared():
    """Test Newton's method to find the min of f(x) = (x - a)^2 + (x - a)^4.

    For this function, it can be shown that for x' = x - f'(x) / f''(x),

        (x' - a) / (x - a) = 8(x - a)^2 / (2 + 12(x - a)^2),

    which is globally (super)linear convergence with rate at least 2/3, and locally
    cubic convergence.
    """
    seed = 0
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    a = jax.random.normal(subkey, shape=(1,))

    x = a + 2.5
    dummy_data = jnp.zeros((jax.local_device_count(),))
    nchains = dummy_data.shape[0]

    (
        dummy_data,
        x,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_data_params_optstate_and_key(
        dummy_data, x, None, key
    )

    nburn = 0
    nepochs = 10
    nsteps_per_param_update = 1

    # define some dummy functions which don't do anything
    def proposal_fn(x, data, key):
        del x
        return data, key

    def acceptance_fn(x, data, proposed_data):
        del x, data, proposed_data
        return jnp.ones((1,))

    def update_data_fn(data, proposed_data, move_mask):
        del proposed_data, move_mask
        return data

    metrop_step_fn = mcmc.metropolis.make_metropolis_step(
        proposal_fn, acceptance_fn, update_data_fn
    )

    # do Newton's method, x <- x - f'(x) / f''(x)
    def update_param_fn(data, x, optimizer_state):
        del data
        dmodel = 2.0 * (x - a) + 4.0 * jnp.power(x - a, 3)
        ddmodel = 2.0 + 12.0 * jnp.power(x - a, 2)

        new_x = x - dmodel / ddmodel
        return new_x, optimizer_state, None

    min_x, _, _ = train.vmc.vmc_loop(
        x,
        optimizer_state,
        dummy_data,
        nchains,
        nburn,
        nepochs,
        nsteps_per_param_update,
        metrop_step_fn,
        update_param_fn,
        key,
    )
    min_x = kfac_utils.get_first(min_x)

    np.testing.assert_allclose(min_x, a)
