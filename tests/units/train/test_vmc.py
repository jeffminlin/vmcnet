"""Testing main VMC routine."""
import logging

import jax.numpy as jnp
import jax
import numpy as np

import kfac_ferminet_alpha.utils as kfac_utils

import vmcnet.train as train
import vmcnet.utils as utils


def _make_dummy_data_params_and_key():
    """Make some random data, params, and a key."""
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    key = keys[0]

    data = jax.random.normal(keys[1], shape=(10, 5, 2))

    param_shapes = ((3, 1), (5, 7))
    params = [
        jax.random.normal(keys[i + 2], shape=shape)
        for i, shape in enumerate(param_shapes)
    ]

    return data, params, key


def _make_dummy_metropolis_fns(data, key):
    """Make a random proposal with the shape of data and accept every other row."""
    key, subkey = jax.random.split(key)
    fixed_proposal = jax.random.normal(subkey, shape=data.shape)

    def proposal_fn(params, data, key):
        """Add a fixed proposal to the data."""
        del params
        return data + fixed_proposal, key

    def acceptance_fn(params, data, proposed_data):
        """Accept every other row of the proposal."""
        del params, proposed_data
        acceptance_prob = jnp.zeros(data.shape[0])
        acceptance_prob = jax.ops.index_update(acceptance_prob, jax.ops.index[::2], 1.0)
        return acceptance_prob

    def update_data_fn(data, proposed_data, move_mask):
        pos_mask = jnp.reshape(move_mask, (-1,) + (len(data.shape) - 1) * (1,))
        return jnp.where(pos_mask, proposed_data, data)

    return fixed_proposal, proposal_fn, acceptance_fn, update_data_fn, key


def _get_expected_alternate_row_update(data, fixed_proposal, num_updates):
    """Get the result after updating every other row of data num_updates times."""
    expected_new_data = data
    # update every other row of data by adding num_updates * fixed_proposal
    expected_new_data = jax.ops.index_update(
        expected_new_data,
        jax.ops.index[::2, ...],
        (data + num_updates * fixed_proposal)[::2],
    )

    return expected_new_data


def test_metropolis_step():
    """Test the acceptance probability and data update for a single Metropolis step."""
    data, params, key = _make_dummy_data_params_and_key()
    (
        fixed_proposal,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        key,
    ) = _make_dummy_metropolis_fns(data, key)
    accept_prob, new_data, _ = train.vmc.take_metropolis_step(
        data, params, proposal_fn, acceptance_fn, update_data_fn, key
    )
    expected_new_data = _get_expected_alternate_row_update(data, fixed_proposal, 1)

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, expected_new_data)


def test_walk_data():
    """Test a few Metropolis steps.
    
    Test that taking a few Metropolis steps is equivalent to skipping to the end and
    taking one big step. Specifically, with a single proposal fn which adds a constant
    array at each step, test that taking a few steps is equivalent to adding that
    multiple of the proposal array directly (where the moves are accepted).
    """
    nsteps = 6
    data, params, key = _make_dummy_data_params_and_key()
    (
        fixed_proposal,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        key,
    ) = _make_dummy_metropolis_fns(data, key)
    accept_prob, new_data, _ = train.vmc.walk_data(
        nsteps, data, params, proposal_fn, acceptance_fn, update_data_fn, key
    )
    expected_new_data = _get_expected_alternate_row_update(data, fixed_proposal, nsteps)

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, expected_new_data, rtol=1e-5)


def test_update_position_and_amplitude():
    """Test that the mask modification is working for the position update."""
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)

    nbatch = 10
    nparticles = 7
    ndim = 3

    pos = jax.random.normal(keys[0], shape=(nbatch, nparticles, ndim))
    proposed_pos = jax.random.normal(keys[1], shape=(nbatch, nparticles, ndim))
    amplitude = jax.random.normal(keys[2], shape=(nbatch,))
    proposed_amplitude = jax.random.normal(keys[3], shape=(nbatch,))

    data = {"position": pos, "amplitude": amplitude}
    proposed_data = {"position": proposed_pos, "amplitude": proposed_amplitude}
    move_mask = jnp.array(
        [True, False, True, True, False, True, False, False, False, False]
    )
    inverted_mask = jnp.invert(move_mask)
    updated_data = train.vmc.update_position_and_amplitude(
        data, proposed_data, move_mask
    )

    for datum in data:
        # updated data should be equal to the proposed data where move_mask is True
        np.testing.assert_allclose(
            updated_data[datum][move_mask, ...], proposed_data[datum][move_mask, ...]
        )
        # updated data should be equal to the original data where move_mask is False
        np.testing.assert_allclose(
            updated_data[datum][inverted_mask, ...], data[datum][inverted_mask, ...]
        )


def test_vmc_loop_logging(caplog):
    """Test vmc_loop logging. Uses pytest's caplog fixture to capture logs."""
    nburn = 4
    nepochs = 13  # eventual number of parameter updates
    nskip = 10

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
        _, proposal_fn, acceptance_fn, update_data_fn, key = _make_dummy_metropolis_fns(
            data, key
        )

        if pmapped:
            data, params, key = utils.distribute.distribute_data_params_and_key(
                data, params, key
            )

        with caplog.at_level(logging.INFO):
            train.vmc.vmc_loop(
                params,
                None,
                data,
                nburn,
                nepochs,
                nskip,
                proposal_fn,
                acceptance_fn,
                update_data_fn,
                update_param_fn,
                key,
                pmapped=pmapped,
            )

        # 1 line for burning, nepochs lines for training
        assert len(caplog.records) == 1 + nepochs


def test_vmc_loop_number_of_updates():
    """Test updating data nskip * nepochs + nburn times and params nepoch times."""
    data, params, key = _make_dummy_data_params_and_key()
    (
        fixed_proposal,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        key,
    ) = _make_dummy_metropolis_fns(data, key)

    old_data = data

    data, params, key = utils.distribute.distribute_data_params_and_key(
        data, params, key
    )

    nburn = 5
    nepochs = 17  # eventual number of parameter updates
    nskip = 2

    # storing the number of parameter updates in optimizer_state, replicated on each
    # device; with a real optimizer this is probably something more exciting and
    # possibly data-dependent (e.g. KFAC/Adam's running metrics)
    optimizer_state = kfac_utils.replicate_all_local_devices(0)

    def update_param_fn(data, params, optimizer_state):
        del data
        optimizer_state += 1
        return params, optimizer_state, None

    _, new_optimizer_state, new_data = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nburn,
        nepochs,
        nskip,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        update_param_fn,
        key,
    )

    new_optimizer_state = kfac_utils.get_first(new_optimizer_state)
    new_data = jnp.reshape(new_data, old_data.shape)

    expected_data = _get_expected_alternate_row_update(
        old_data, fixed_proposal, nskip * nepochs + nburn
    )

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    # check that the expected number of "data updates" have been done
    np.testing.assert_allclose(new_data, expected_data, rtol=1e-5)


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
    dummy_data = jnp.zeros((jax.device_count(),))

    dummy_data, x, key = utils.distribute.distribute_data_params_and_key(
        dummy_data, x, key
    )

    nburn = 0
    nepochs = 10
    nskip = 1

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

    # do Newton's method, x <- x - f'(x) / f''(x)
    def update_param_fn(data, x, optimizer_state):
        del data
        dmodel = 2.0 * (x - a) + 4.0 * jnp.power(x - a, 3)
        ddmodel = 2.0 + 12.0 * jnp.power(x - a, 2)

        new_x = x - dmodel / ddmodel
        return new_x, optimizer_state, None

    min_x, _, _ = train.vmc.vmc_loop(
        x,
        None,
        dummy_data,
        nburn,
        nepochs,
        nskip,
        proposal_fn,
        acceptance_fn,
        update_data_fn,
        update_param_fn,
        key,
    )
    min_x = kfac_utils.get_first(min_x)

    np.testing.assert_allclose(min_x, a)
