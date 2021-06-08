"""Input/output utilities."""
import os

import numpy as np
import flax.serialization as serialization


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


def save_params(directory, name, epoch, data, params, optimizer_state, key):
    """Save a VMC state."""
    with open_or_create(directory, name, "wb") as file_handle:
        file_handle.write(
            serialization.to_bytes((epoch, data, params, optimizer_state, key))
        )


def reload_params(template, directory, name):
    """Save a VMC state."""
    with open_or_create(directory, name, "rb") as file_handle:
        deserialized = serialization.from_bytes(template, file_handle.read())
        return deserialized


def add_suffix_for_uniqueness(name, logdir, pre_suffix=""):
    """Adds a numerical suffix to keep names unique in a directory."""
    final_name = name
    i = 0
    while (final_name + pre_suffix) in os.listdir(logdir):
        i += 1
        final_name = name + "_" + str(i)
    return final_name
