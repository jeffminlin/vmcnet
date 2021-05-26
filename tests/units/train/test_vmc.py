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
    data = jnp.array([0, 0, 0, 0])
    params = [jnp.array([1, 2, 3]), jnp.array([[4, 5], [6, 7]])]

    return data, params, key


def _make_different_pmappable_data(data):
    """Adding (0, 1, ..., ndevices - 1) to the data and concatenating."""
    ndevices = jax.device_count()
    return jnp.concatenate([data + i for i in range(ndevices)])


def _make_dummy_metropolis_fn():
    """Make a random proposal with the shape of data and accept every other row."""

    def proposal_fn(params, data, key):
        """Add a fixed proposal to the data."""
        del params
        return data + jnp.array([1, 2, 3, 4]), key

    def acceptance_fn(params, data, proposed_data):
        """Accept every other row of the proposal."""
        del params, proposed_data
        return jnp.array([True, False, True, False], dtype=bool)

    def update_data_fn(data, proposed_data, move_mask):
        pos_mask = jnp.reshape(move_mask, (-1,) + (len(data.shape) - 1) * (1,))
        return jnp.where(pos_mask, proposed_data, data)

    metrop_step_fn = train.vmc.make_metropolis_step(
        proposal_fn, acceptance_fn, update_data_fn
    )

    return metrop_step_fn


def test_metropolis_step():
    """Test the acceptance probability and data update for a single Metropolis step."""
    data, params, key = _make_dummy_data_params_and_key()
    metrop_step_fn = _make_dummy_metropolis_fn()

    accept_prob, new_data, _ = metrop_step_fn(data, params, key)

    np.testing.assert_allclose(accept_prob, 0.5)
    np.testing.assert_allclose(new_data, jnp.array([1, 0, 3, 0]))


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
        metrop_step_fn = _make_dummy_metropolis_fn()
        nchains = data.shape[0]

        if pmapped:
            data = _make_different_pmappable_data(data)
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
                metrop_step_fn,
                update_param_fn,
                key,
                pmapped=pmapped,
            )

        # 1 line for burning, nepochs lines for training
        assert len(caplog.records) == 1 + nepochs


def test_vmc_loop_number_of_updates():
    """Test updating data nskip * nepochs + nburn times and params nepoch times."""
    data, params, key = _make_dummy_data_params_and_key()
    metrop_step_fn = _make_dummy_metropolis_fn()
    nchains = data.shape[0]

    data = _make_different_pmappable_data(data)
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
        metrop_step_fn,
        update_param_fn,
        key,
    )

    new_optimizer_state = kfac_utils.get_first(new_optimizer_state)

    num_updates = nskip * nepochs + nburn

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    for device_id in range(jax.device_count()):
        np.testing.assert_allclose(
            new_data[device_id],
            jnp.array([num_updates, 0, 3 * num_updates, 0]) + device_id,
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

    metrop_step_fn = train.vmc.make_metropolis_step(
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
        None,
        dummy_data,
        nchains,
        nburn,
        nepochs,
        nskip,
        metrop_step_fn,
        update_param_fn,
        key,
    )
    min_x = kfac_utils.get_first(min_x)

    np.testing.assert_allclose(min_x, a)
