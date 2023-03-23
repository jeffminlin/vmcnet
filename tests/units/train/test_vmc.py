"""Testing main VMC routine."""
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.utils as utils

from tests.test_utils import make_dummy_data_params_and_key, make_dummy_metropolis_fn


def _make_different_pmappable_data(data):
    """Adding (0, 1, ..., ndevices - 1) to the data and concatenating."""
    ndevices = jax.local_device_count()
    return jnp.concatenate([data + i for i in range(ndevices)])


@pytest.mark.slow
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

    def update_param_fn(params, data, optimizer_state, key):
        return params, data, optimizer_state, fixed_metrics, key

    for pmapped in [True, False]:
        caplog.clear()
        data, params, key = make_dummy_data_params_and_key()
        metrop_step_fn = make_dummy_metropolis_fn()
        nchains = data.shape[0]

        if pmapped:
            data = _make_different_pmappable_data(data)
            (
                data,
                params,
                optimizer_state,
                key,
            ) = utils.distribute.distribute_vmc_state(data, params, None, key)

        burning_step = mcmc.metropolis.make_jitted_burning_step(
            metrop_step_fn, apply_pmap=pmapped
        )
        walker_fn = mcmc.metropolis.make_jitted_walker_fn(
            nsteps_per_param_update, metrop_step_fn, apply_pmap=pmapped
        )

        with caplog.at_level(logging.INFO):
            data, key = mcmc.metropolis.burn_data(
                burning_step, nburn, params, data, key
            )
            train.vmc.vmc_loop(
                params,
                optimizer_state,
                data,
                nchains,
                nepochs,
                walker_fn,
                update_param_fn,
                key,
                is_pmapped=pmapped,
            )

        # 1 line for burning, nepochs lines for training
        assert len(caplog.records) == 1 + nepochs


@pytest.mark.slow
def test_vmc_loop_number_of_updates():
    """Test number of updates.

    Make sure that data is updated nsteps_per_param_update * nepochs + nburn times and
    params is updated nepoch times.
    """
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()
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
    ) = utils.distribute.distribute_vmc_state(data, params, 0, key)

    nburn = 5
    nepochs = 17  # eventual number of parameter updates
    nsteps_per_param_update = 2

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    def update_param_fn(params, data, optimizer_state, key):
        optimizer_state += 1
        return params, data, optimizer_state, None, key

    data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)
    _, new_optimizer_state, new_data, _, _ = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nchains,
        nepochs,
        walker_fn,
        update_param_fn,
        key,
    )

    new_optimizer_state = utils.distribute.get_first(new_optimizer_state)

    num_updates = nsteps_per_param_update * nepochs + nburn

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    for device_index in range(jax.local_device_count()):
        np.testing.assert_allclose(
            new_data[device_index],
            jnp.array([num_updates, 0, 3 * num_updates, 0]) + device_index,
        )
