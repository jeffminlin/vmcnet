"""Testing main VMC routine."""
import logging

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha.utils as kfac_utils
import numpy as np

import vmcnet.train as train
import vmcnet.utils as utils


def _make_dummy_data_params_and_key():
    """Make some random data, params, and a key."""
    seed = 0
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    key = keys[0]

    data = jax.random.normal(keys[1], shape=(10 * jax.device_count(), 5, 2))

    param_shapes = ((3, 1), (5, 7))
    params = [
        jax.random.normal(keys[i + 2], shape=shape)
        for i, shape in enumerate(param_shapes)
    ]

    return data, params, key


def _make_dummy_metropolis_fns(data, key, data_will_be_pmapped=False):
    """Make a random proposal with the shape of data and accept every other row."""
    key, subkey = jax.random.split(key)
    proposal_shape = data.shape
    if data_will_be_pmapped:
        proposal_shape = (data.shape[0] // jax.device_count(),) + data.shape[1:]
    fixed_proposal = jax.random.normal(subkey, shape=proposal_shape)

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


def _get_expected_alternate_row_update(
    data, fixed_proposal, num_updates, pmapped=False
):
    """Get the result after updating every other row of data num_updates times."""

    def single_device_update(data, fixed_proposal):
        expected_new_data = data
        # update every other row of data by adding num_updates * fixed_proposal
        expected_new_data = jax.ops.index_update(
            expected_new_data,
            jax.ops.index[::2, ...],
            (data + num_updates * fixed_proposal)[::2],
        )

        return expected_new_data

    if pmapped:
        fixed_proposal = kfac_utils.replicate_all_local_devices(fixed_proposal)
        return jax.pmap(single_device_update)(data, fixed_proposal)

    return single_device_update(data, fixed_proposal)


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
            data, key, data_will_be_pmapped=pmapped
        )
        nchains = data.shape[0]

        if pmapped:
            data, params, key = utils.distribute.distribute_data_params_and_key(
                data, params, key
            )

        with caplog.at_level(logging.INFO):
            train.vmc.vmc_loop(
                params,
                None,
                data,
                nchains,
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
    ) = _make_dummy_metropolis_fns(data, key, data_will_be_pmapped=True)
    nchains = data.shape[0]

    # keep a copy here because vmc_loop deletes the original buffer
    data_copy = jnp.array(data)
    data_copy = utils.distribute.distribute_data(data_copy)

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
        nchains,
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

    num_updates = nskip * nepochs + nburn
    expected_data = _get_expected_alternate_row_update(
        data_copy, fixed_proposal, num_updates, pmapped=True
    )

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    # check that the expected number of "data updates" have been done. Since repeatedly
    # increasing the size of data accumulates relative numerical error on the order of
    # n * eps, we set an absolute tolerance of n * eps * max(result)
    print(new_data - expected_data)
    np.testing.assert_allclose(
        new_data,
        expected_data,
        atol=num_updates * np.finfo(jnp.float32).eps * jnp.max(new_data),
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
    dummy_data = jnp.zeros((jax.device_count(),))
    nchains = dummy_data.shape[0]

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
        nchains,
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
