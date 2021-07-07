"""Input/output utilities."""
import os

import flax.core.frozen_dict as frozen_dict
import numpy as np

from .distribute import get_first, is_distributed
from .typing import CheckpointData


def open_or_create(path, filename, option):
    """Opens or creates a flepath."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    return open(filepath, option)


def append_metric_to_file(new_metric, logdir, name):
    """Appends a number or list of numbers to a text file."""
    dumped_metric = np.array(new_metric).reshape((1, -1))

    with open_or_create(logdir, name + ".txt", "a") as outfile:
        np.savetxt(outfile, dumped_metric)


def process_checkpoint_data_for_saving(checkpoint_data: CheckpointData):
    """Process potentially pmapped checkpoint data to prepare for saving to disc.

    Params and opt_state are always replicated across devices, so here we save only
    only copy. Params must also be converted from a frozen_dict to a dict.
    """
    (epoch, data, params, optimizer_state, key) = checkpoint_data
    params = params.unfreeze()
    if is_distributed(params):
        params = get_first(params)
        optimizer_state = get_first(optimizer_state)
    return (epoch, data, params, optimizer_state, key)


def save_vmc_state(directory, name, checkpoint_data: CheckpointData):
    """Save a VMC state to disc.

    Data is not processed beyond splitting the data into its constituent pieces and
    saving each one to disc. Data should thus be preprocessed as needed, for example
    by getting a single copy of params that have been replicated across multiple GPUs.

    Args:
      directory (str): directory in which to write the checkpoint
      name (str): filename for the checkpoint
      checkpoint_data (CheckpointData): data to save
    """
    (epoch, data, params, optimizer_state, key) = checkpoint_data

    with open_or_create(directory, name, "wb") as file_handle:
        np.savez(
            file_handle,
            e=epoch,
            d=data,
            p=params,
            o=optimizer_state,
            k=key,
        )


def reload_vmc_state(directory: str, name: str) -> CheckpointData:
    """Reload a VMC state from a saved checkpoint."""
    with open_or_create(directory, name, "rb") as file_handle:
        # np.savez wraps non-array objects in arrays for storage, so call
        # tolist() on such objects to get them back to their original type.
        with np.load(file_handle, allow_pickle=True) as npz_data:
            epoch = npz_data["e"].tolist()

            data: np.ndarray = npz_data["d"]
            # Detect whether the data was originally an object, in which case it should
            # have dtype object, or an array, in which case it should have dtype
            # something else. This WILL BREAK if you use data that is an array of dtype
            # object.
            if data.dtype == np.dtype("object"):
                data = data.tolist()

            # Params are stored by flax as a frozen dict, so mimic that behavior here.
            params = frozen_dict.freeze(npz_data["p"].tolist())
            optimizer_state = npz_data["o"].tolist()
            key = npz_data["k"]
            return (epoch, data, params, optimizer_state, key)


def add_suffix_for_uniqueness(name, logdir, pre_suffix=""):
    """Adds a numerical suffix to keep names unique in a directory."""
    final_name = name
    i = 0
    while (final_name + pre_suffix) in os.listdir(logdir):
        i += 1
        final_name = name + "_" + str(i)
    return final_name
