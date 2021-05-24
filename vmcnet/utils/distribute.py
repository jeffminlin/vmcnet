"""Helper functions for distributing computation to multiple devices."""
import jax
import jax.numpy as jnp

from kfac_ferminet_alpha import utils as kfac_utils


def distribute_data_params_and_key(data, params, key):
    """Broadcast data, replicate params, and split a PRNG key to all devices."""
    num_devices = jax.device_count()
    batch_size = data.shape[0]
    if batch_size % num_devices != 0:
        raise ValueError(
            "Batch size must be divisible by number of devices, "
            "got batch size {} for {} devices.".format(batch_size, num_devices)
        )
    distributed_data_shape = (num_devices, batch_size // num_devices)
    data = jnp.reshape(data, distributed_data_shape + data.shape[1:])
    data = kfac_utils.broadcast_all_local_devices(data)

    params = kfac_utils.replicate_all_local_devices(params)

    sharded_key = kfac_utils.make_different_rng_key_on_all_devices(key)

    return data, params, sharded_key
