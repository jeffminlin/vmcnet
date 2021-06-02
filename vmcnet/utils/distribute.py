"""Helper functions for distributing computation to multiple devices."""
import functools
from typing import TypeVar

import jax
from jax import core
import jax.numpy as jnp

T = TypeVar("T")  # representing an arbitrary pytree

# axis name to pmap over
PMAP_AXIS_NAME = "pmap_axis"


def wrap_if_pmap(p_func):
    """Make a function run if in a pmapped context."""

    def p_func_if_pmap(obj, axis_name):
        try:
            core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


# shortcuts to use a pmapped axis called PMAP_AXIS_NAME
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

broadcast_all_local_devices = jax.pmap(lambda x: x)


def replicate_all_local_devices(obj: T) -> T:
    """Replicate a pytree on all local devices."""
    if obj is None:
        return None
    n = jax.local_device_count()
    obj_stacked = jax.tree_map(lambda x: jnp.stack([x] * n, axis=0), obj)
    return broadcast_all_local_devices(obj_stacked)


def make_different_rng_key_on_all_devices(rng: jnp.ndarray) -> jnp.ndarray:
    """Split a PRNG key to all local devices."""
    rng = jax.random.fold_in(rng, jax.host_id())
    rng = jax.random.split(rng, jax.local_device_count())
    return broadcast_all_local_devices(rng)


def get_first(obj: T) -> T:
    """Get the first object in each leaf of a pytree.

    Can be used to grab the first instance of a replicated object on the first local
    device.
    """
    return jax.tree_map(lambda x: x[0], obj)


pmean_if_pmap = functools.partial(wrap_if_pmap(jax.lax.pmean), axis_name=PMAP_AXIS_NAME)
mean_all_local_devices = lambda x: pmean_if_pmap(jnp.mean(x))


def reshape_data_leaves_for_distribution(data_leaf):
    """For a leaf of a pytree, reshape it for distributing to all local devices."""
    num_devices = jax.local_device_count()
    nchains = data_leaf.shape[0]
    if nchains % num_devices != 0:
        raise ValueError(
            "Number of chains must be divisible by number of devices, "
            "got nchains {} for {} devices.".format(nchains, num_devices)
        )
    distributed_data_shape = (num_devices, nchains // num_devices)
    data = jnp.reshape(data_leaf, distributed_data_shape + data_leaf.shape[1:])
    return data


def distribute_data(data):
    """Split data to all devices. The first axis must be divisible by ndevices."""
    data = jax.tree_map(reshape_data_leaves_for_distribution, data)
    data = broadcast_all_local_devices(data)
    return data


def distribute_data_params_optstate_and_key(data, params, optimizer_state, key):
    """Split data, replicate params and opt state, and split PRNG key to all devices."""
    data = distribute_data(data)
    params = replicate_all_local_devices(params)
    optimizer_state = replicate_all_local_devices(optimizer_state)
    sharded_key = make_different_rng_key_on_all_devices(key)

    return data, params, optimizer_state, sharded_key
