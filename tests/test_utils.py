"""Shared pieces for unit tests."""
import flax.core.frozen_dict as frozen_dict
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc as mcmc
from vmcnet.utils.typing import PyTree


def make_dummy_log_f():
    """Make a simple function and its log for testing."""

    def f(params, x):
        del params
        return jnp.sum(jnp.square(x) + 3 * x)

    def log_f(params, x):
        return jnp.log(jnp.abs(f(params, x)))

    return f, log_f


def make_dummy_x():
    """Make a simple array of inputs."""
    return jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def make_dummy_data_params_and_key():
    """Make some random data, params, and a key."""
    seed = 0
    key = jax.random.PRNGKey(seed)
    data = jnp.array([0, 0, 0, 0])
    params = frozen_dict.freeze(
        {"kernel_1": jnp.array([1, 2, 3]), "kernel_2": jnp.array([[4, 5], [6, 7]])}
    )

    return data, params, key


def make_dummy_metropolis_fn():
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

    metrop_step_fn = mcmc.metropolis.make_metropolis_step(
        proposal_fn, acceptance_fn, update_data_fn
    )

    return metrop_step_fn


def dummy_model_apply(params, x):
    """Model eval that outputs indices of the flattened x in the shape of x."""
    return jnp.reshape(jnp.arange(jnp.size(x)), x.shape)


def assert_pytree_allclose(
    pytree1: PyTree,
    pytree2: PyTree,
    rtol: float = 1e-7,
    atol: float = 0.0,
    verbose: bool = True,
):
    """Use jax.tree_map to assert equality at all leaves of two pytrees."""
    jax.tree_map(
        lambda l1, l2: np.testing.assert_allclose(
            l1, l2, rtol=rtol, atol=atol, verbose=verbose
        ),
        pytree1,
        pytree2,
    )
