"""Shared pieces for the test suite."""
from typing import Tuple

import flax.core.frozen_dict as frozen_dict
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc as mcmc
import vmcnet.models as models
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


def _get_log_domain_params_for_dense_layer(params):
    if "bias" not in params:
        return {"kernel": params["kernel"]}

    kernel = params["kernel"]
    bias = params["bias"]
    return {"kernel": jnp.concatenate([kernel, jnp.expand_dims(bias, 0)], axis=0)}


def get_dense_and_log_domain_dense_same_params(
    key: jnp.ndarray,
    batch: jnp.ndarray,
    dense_layer: models.core.Dense,
) -> Tuple[frozen_dict.FrozenDict, frozen_dict.FrozenDict]:
    """Get matching params for Dense and LogDomainDense layers."""
    dense_params = dense_layer.init(key, batch)
    log_domain_params = _get_log_domain_params_for_dense_layer(dense_params["params"])
    log_domain_params = frozen_dict.freeze({"params": log_domain_params})
    return dense_params, log_domain_params


def get_resnet_and_log_domain_resnet_same_params(
    key: jnp.ndarray,
    batch: jnp.ndarray,
    resnet: models.core.SimpleResNet,
) -> Tuple[frozen_dict.FrozenDict, frozen_dict.FrozenDict]:
    """Get matching params for SimpleResNet and LogDomainResnet models."""
    resnet_params = resnet.init(key, batch)
    log_domain_params = {}

    for dense_layer_key, layer_params in resnet_params["params"].items():
        log_domain_layer_params = _get_log_domain_params_for_dense_layer(layer_params)
        log_domain_params[dense_layer_key] = log_domain_layer_params

    log_domain_params = frozen_dict.freeze({"params": log_domain_params})
    return resnet_params, log_domain_params


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
