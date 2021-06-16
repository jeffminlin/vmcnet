"""Input/output utilities."""
import os

import flax.core.frozen_dict as frozen_dict
import numpy as np

from .distribute import get_first


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


def save_vmc_state(
    directory: str,
    name: str,
    epoch: int,
    data,
    params,
    optimizer_state,
    key,
    pmapped: bool = True,
):
    """Save a VMC state."""
    with open_or_create(directory, name, "wb") as file_handle:

        params = params.unfreeze()
        if pmapped:
            params = get_first(params)
            optimizer_state = get_first(optimizer_state)

        np.savez(
            file_handle,
            e=epoch,
            d=data,
            # Params and opt_state are always replicated, so only save one copy.
            # Params must also be converted from a frozen_dict to a dict.
            p=params,
            o=optimizer_state,
            k=key,
        )


def reload_vmc_state(directory: str, name: str):
    """Reload a VMC state from a saved checkpoint."""
    with open_or_create(directory, name, "rb") as file_handle:
        # np.savez wraps non-array objects in arrays for storage, so call
        # tolist() on such objects to get them back to their original type.
        with np.load(file_handle, allow_pickle=True) as npz_data:
            epoch = npz_data["e"].tolist()
            data = npz_data["d"].tolist()
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
