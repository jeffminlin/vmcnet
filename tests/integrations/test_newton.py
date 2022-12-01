"""Integration test to make sure VMC can find minima of convex functions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.utils as utils


@pytest.mark.slow
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
    ) = utils.distribute.distribute_vmc_state(dummy_data, x, None, key)

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
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    # do Newton's method, x <- x - f'(x) / f''(x)
    def update_param_fn(x, data, optimizer_state, key):
        dmodel = 2.0 * (x - a) + 4.0 * jnp.power(x - a, 3)
        ddmodel = 2.0 + 12.0 * jnp.power(x - a, 2)

        new_x = x - dmodel / ddmodel
        return new_x, data, optimizer_state, None, key

    min_x, _, _, _, _ = train.vmc.vmc_loop(
        x,
        optimizer_state,
        dummy_data,
        nchains,
        nepochs,
        walker_fn,
        update_param_fn,
        key,
    )
    min_x = utils.distribute.get_first(min_x)

    np.testing.assert_allclose(min_x, a)
