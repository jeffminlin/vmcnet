"""Input/output utilities."""
import functools
import os
from typing import Any, Callable, Dict, IO, TypeVar

import jax
import json
from ml_collections import ConfigDict
import numpy as np

from .distribute import get_first
from .typing import CheckpointData

C = TypeVar("C", Dict, ConfigDict)


def open_existing_file(path, filename, option):
    """Opens a filepath that already exists."""
    filepath = os.path.join(path, filename)
    return open(filepath, option)


def open_or_create(path, filename, option):
    """Opens or creates a flepath."""
    os.makedirs(path, exist_ok=True)
    return open_existing_file(path, filename, option)


def append_metric_to_file(new_metric, logdir, name):
    """Appends a number or list of numbers to a text file."""
    dumped_metric = np.array(new_metric).reshape((1, -1))

    with open_or_create(logdir, name + ".txt", "a") as outfile:
        np.savetxt(outfile, dumped_metric)


def copy_txt_stats(source_dir, target_dir, truncate=None):
    """Copies and truncated energy.txt as similar stats."""
    filenames = [
        n for n in os.listdir(source_dir) if n.endswith(".txt") and "git_hash" not in n
    ]

    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if not os.path.exists(source_path):
            continue

        with open(source_path, "r") as f:
            lines = f.readlines()
            if truncate:
                lines = lines[:truncate]

        with open(target_path, "w") as f:
            f.writelines(lines)


def _config_dict_write(fp: IO[str], config: ConfigDict) -> None:
    """Write config dict to json."""
    fp.write(config.to_json(indent=4))


def _dictionary_write(fp: IO[str], dictionary: Dict) -> None:
    """Write dictionary to json."""
    json.dump(dictionary, fp, indent=4)


def _save_to_unique_json(
    info: C,
    logdir: str,
    base_filename: str,
    saving_fn: Callable[[IO[str], C], None],
) -> None:
    """Save a dict or ConfigDict to a json file, ensuring the filename is unique."""
    unique_filename = add_suffix_for_uniqueness(
        base_filename, logdir, trailing_suffix=".json"
    )
    with open_or_create(logdir, unique_filename + ".json", "w") as f:
        saving_fn(f, info)


save_config_dict_to_json: Callable[[ConfigDict, str, str], None] = functools.partial(
    _save_to_unique_json, saving_fn=_config_dict_write
)
save_dict_to_json: Callable[[Dict, str, str], None] = functools.partial(
    _save_to_unique_json, saving_fn=_dictionary_write
)


def _convert_to_tuple_if_list(leaf: Any) -> Any:
    if isinstance(leaf, list):
        return tuple(_convert_to_tuple_if_list(entry) for entry in leaf)
    return leaf


def _convert_dict_lists_to_tuples(d: Dict) -> Dict:
    return jax.tree_map(
        _convert_to_tuple_if_list,
        d,
        is_leaf=lambda li: isinstance(li, list),
    )


def load_config_dict(logdir: str, relative_path: str):
    """Load a ConfigDict from a json file.

    JSON will automatically have converted all tuples in the ConfigDict to lists, so
    when we reload here, we have to convert any lists we find back to tuples.
    """
    config_path = os.path.join(logdir, relative_path)
    with open(config_path) as json_file:
        raw_dict = json.load(json_file)
        dict_with_tuples = _convert_dict_lists_to_tuples(raw_dict)
        return ConfigDict(dict_with_tuples)


def process_checkpoint_data_for_saving(
    checkpoint_data: CheckpointData, is_distributed: bool
):
    """Process potentially pmapped checkpoint data to prepare for saving to disc.

    Params and opt_state are always replicated across devices, so here we save only
    only copy. Params must also be converted from a frozen_dict to a dict.
    """
    (epoch, data, params, optimizer_state, key) = checkpoint_data
    if is_distributed:
        params = get_first(params)
        optimizer_state = get_first(optimizer_state)
    optimizer_state = wrap_singleton(optimizer_state)
    return (epoch, data, params, optimizer_state, key)


def wrap_singleton(x):
    """Wrap tuple.

    numpy.savez does not load tuples correctly,
    so we wrap the optimizer_state.
    """
    return {"single_element": x}


def unwrap_if_singleton(x):
    """Unwrap data.

    numpy.savez does not load tuples correctly,
    so we wrap the optimizer_state.
    """
    if isinstance(x, dict) and len(x) == 1 and "single_element" in x:
        return x["single_element"]
    return x


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
    with open_existing_file(directory, name, "rb") as file_handle:
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

            params = npz_data["p"].tolist()
            optimizer_state = unwrap_if_singleton(npz_data["o"].tolist())
            key = npz_data["k"]
            return (epoch, data, params, optimizer_state, key)


def add_suffix_for_uniqueness(relative_path, base_dir="", trailing_suffix=""):
    """Adds a numerical suffix to keep names unique in a directory.

    Checks for the presence of name + trailing_suffix, name + "_1" + trailing_suffix,
    name + "_2" + trailing_suffix, etc. until an integer i >= 1 is found such that
    name + "_i" + trailing_suffix is not in logdir. Then name + "_i" is returned.

    If name + trailing_suffix is not in logdir, returns name.
    """
    final_relative_path = relative_path.rstrip(os.sep)
    i = 0
    while os.path.exists(os.path.join(base_dir, final_relative_path + trailing_suffix)):
        i += 1
        final_relative_path = relative_path + "_" + str(i)

    # This function is for ensuring uniqueness, so it doesn't check existence
    return final_relative_path


def get_vmcpath():
    """Get the path to the vmcnet package."""
    path = os.path.dirname(os.path.abspath(__file__))
    path = path.split("vmcnet")[0]
    return os.path.join(path, "vmcnet")
