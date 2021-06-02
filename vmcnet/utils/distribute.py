"""Helper functions for distributing computation to multiple devices."""
import functools
from typing import Callable, Tuple, TypeVar

import jax
import jax.interpreters.pxla as pxla
import jax.numpy as jnp
from jax import core

T = TypeVar("T")  # representing an arbitrary pytree
D = TypeVar("D")  # represents a pytree or pytree-like object containing MCMC data
P = TypeVar("P")  # represents a pytree or pytree-like object containing model params
S = TypeVar("S")  # represents optimizer state

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

broadcast_all_local_devices = pmap(lambda x: x)


def replicate_all_local_devices(obj: T) -> T:
    """Replicate a pytree on all local devices."""
    if obj is None:
        return None
    n = jax.local_device_count()
    obj_stacked = jax.tree_map(lambda x: jnp.stack([x] * n, axis=0), obj)
    return broadcast_all_local_devices(obj_stacked)


def make_different_rng_key_on_all_devices(rng: jnp.ndarray) -> jnp.ndarray:
    """Split a PRNG key to all local devices."""
    rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.random.split(rng, jax.local_device_count())
    return broadcast_all_local_devices(rng)


def get_first(obj: T) -> T:
    """Get the first object in each leaf of a pytree.

    Can be used to grab the first instance of a replicated object on the first local
    device.
    """
    return jax.tree_map(lambda x: x[0], obj)


pmean_if_pmap = functools.partial(wrap_if_pmap(jax.lax.pmean), axis_name=PMAP_AXIS_NAME)


def mean_all_local_devices(x):
    """Compute mean over all local devices if distributed, otherwise the usual mean."""
    return pmean_if_pmap(jnp.mean(x))

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


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


def default_distribute_data(data):
    """Split all data to all devices. The first axis must be divisible by ndevices."""
    data = jax.tree_map(reshape_data_leaves_for_distribution, data)
    data = broadcast_all_local_devices(data)
    return data


def distribute_vmc_state(
    data: D,
    params: P,
    optimizer_state: S,
    key: jnp.ndarray,
    distribute_data_fn: Callable[[D], D] = default_distribute_data,
) -> Tuple[D, P, S, jnp.ndarray]:
    """Split data, replicate params and opt state, and split PRNG key to all devices.

    Args:
        data: the MCMC data to distribute
        params: model parameters
        optimizer_state: optimizer state
        key: RNG key
        distribute_data_fn: custom function for distributing the MCMC data, for the case
            where some of the data needs to be replicated instead of distributed across
            the devices. Default works if there is no data that requires replication.

    Returns:
        Tuple[D, P, O, jnp.ndarray]: tuple of data, params, optimizer_state, and key,
        each of which has been either distributed or replicated across all devices,
        as appopriate.
    """
    data = distribute_data_fn(data)
    params = replicate_all_local_devices(params)
    optimizer_state = replicate_all_local_devices(optimizer_state)
    sharded_key = make_different_rng_key_on_all_devices(key)

    return data, params, optimizer_state, sharded_key


def distribute_vmc_state_from_checkpoint(
    data: D,
    params: P,
    optimizer_state: S,
    key: jnp.ndarray,
) -> Tuple[D, P, S, jnp.ndarray]:
    """Distribute vmc state that was reloaded from a saved checkpoint.

    Data and key are saved independently for each device, so on reload
    we simply broadcast them back to the devices. Params and optimizer state are saved
    as a single copy, so on reload we replicate them to all devices.
    """
    data = broadcast_all_local_devices(data)
    params = replicate_all_local_devices(params)
    optimizer_state = replicate_all_local_devices(optimizer_state)
    key = broadcast_all_local_devices(key)

    return data, params, optimizer_state, key


def is_distributed(data) -> bool:
    """Tests whether given data has been distributed using pmap."""
    return isinstance(jax.tree_leaves(data)[0], pxla.ShardedDeviceArray)
