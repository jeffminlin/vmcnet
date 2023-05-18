"""Helper functions for distributing computation to multiple devices."""
import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import core

from vmcnet.utils.typing import Array, ArrayLike, D, P, PRNGKey, S, T


# axis name to pmap over
PMAP_AXIS_NAME = "pmap_axis"


def wrap_if_pmap(p_func: Callable) -> Callable:
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


def make_different_rng_key_on_all_devices(rng: PRNGKey) -> PRNGKey:
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


def mean_all_local_devices(x: Array) -> Array:
    """Compute mean over all local devices if distributed, otherwise the usual mean."""
    return pmean_if_pmap(jnp.mean(x))


def nanmean_all_local_devices(x: Array) -> Array:
    """Compute a nan-safe mean over all local devices."""
    return pmean_if_pmap(jnp.nanmean(x))


def get_mean_over_first_axis_fn(
    nan_safe: bool = True,
) -> Callable[[ArrayLike], ArrayLike]:
    """Get a function which averages over the first axis over all local devices.

    Args:
        nan_safe (bool, optional): whether to use jnp.nanmean or jnp.mean in the local
            average computation. Defaults to True.

    Returns:
        Callable: function which averages an array over its first axis over all local
        devices.
    """
    if nan_safe:
        local_mean_fn = functools.partial(jnp.nanmean, axis=0)
    else:
        local_mean_fn = functools.partial(jnp.mean, axis=0)

    def mean_fn(x: ArrayLike) -> ArrayLike:
        return pmean_if_pmap(local_mean_fn(x))

    return mean_fn


p_split = pmap(lambda key: tuple(jax.random.split(key)))


def split_or_psplit_key(key: PRNGKey, multi_device: bool = True) -> PRNGKey:
    """Split PRNG key, potentially on multiple devices."""
    return p_split(key) if multi_device else jax.random.split(key)


def reshape_data_leaves_for_distribution(data_leaf: Array) -> Array:
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


def default_distribute_data(data: D) -> D:
    """Split all data to all devices. The first axis must be divisible by ndevices."""
    data = jax.tree_map(reshape_data_leaves_for_distribution, data)
    data = broadcast_all_local_devices(data)
    return data


def distribute_vmc_state(
    data: D,
    params: P,
    optimizer_state: S,
    key: PRNGKey,
    distribute_data_fn: Callable[[D], D] = default_distribute_data,
) -> Tuple[D, P, S, PRNGKey]:
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
        (D, P, S, PRNGKey): tuple of data, params, optimizer_state, and key,
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
    key: PRNGKey,
) -> Tuple[D, P, S, PRNGKey]:
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
