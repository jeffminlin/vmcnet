"""Helper functions for distributing computation to multiple devices."""
import functools

import jax
import jax.numpy as jnp
from kfac_ferminet_alpha import utils as kfac_utils

# axis name to pmap over
PMAP_AXIS_NAME = "pmap_axis"

# shortcuts to use a pmapped axis called PMAP_AXIS_NAME
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
pmean_if_pmap = functools.partial(kfac_utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
replicate_all_local_devices = (
    lambda x: None if None else kfac_utils.replicate_all_local_devices(x)
)


def reshape_data_leaves_for_distribution(data_leaf):
    """For a leaf of a pytree, reshape it for distributing to all local devices."""
    num_devices = jax.device_count()
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
    data = kfac_utils.broadcast_all_local_devices(data)
    return data


def distribute_data_params_optstate_and_key(data, params, optimizer_state, key):
    """Split data, replicate params and opt state, and split PRNG key to all devices."""
    data = distribute_data(data)
    params = replicate_all_local_devices(params)
    optimizer_state = replicate_all_local_devices(optimizer_state)
    sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)

    return data, params, optimizer_state, sharded_key
