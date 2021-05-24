"""Input/output utilities."""
import os

import numpy as np


def open_or_create(path, filename, option):
    """Opens or creates a flepath."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    return open(filepath, option)


def save_params(directory, name, data, params, optimizer_state):
    with open_or_create(directory, name, "wb") as file_handle:
        np.savez(
            file_handle, data=data, params=params, optimizer_state=optimizer_state,
        )


def add_suffix_for_uniqueness(name, logdir, pre_suffix=""):
    """Adds a numerical suffix to keep names unique in a directory."""
    final_name = name
    i = 0
    while (final_name + pre_suffix) in os.listdir(logdir):
        i += 1
        final_name = name + "_" + str(i)
    return final_name
