"""Example script to show how custom NamedTuples don't play well with serialization."""
import logging

import jax
import jax.numpy as jnp
import flax.serialization as serialization
from typing import Dict, NamedTuple, Tuple, TypeVar


class CustomData(NamedTuple):
    data: jnp.ndarray


def main():
    original_data = CustomData(jnp.array([1.0, 2.0, 3.0]))
    data_as_bytes = serialization.to_bytes(original_data)
    restored_data = serialization.from_bytes(original_data, data_as_bytes)

    new_data = CustomData(jnp.array([2.0, 3.0, 4.0]))

    def map_data(data1, data2):
        return data1 + data2

    # Works, because once we pull out the data from the tuples, it's just a jnp.ndarray,
    # in both the restored and new case.
    computed_data = map_data(restored_data.data, new_data.data)
    print("Success, result: ", computed_data)

    # Fails, because restored_data is of type flax.serialization.CustomData, whereas
    # new_data is just of type CustomData. Ideally jax.tree_map would play nicely with
    # flax.serialization by realizing that flax.serialization.CustomData is just the
    # restored version of CustomData and allowing a map over any combination of the two.
    # However, this does not seem to be the case!
    try:
        computed_data = jax.tree_map(map_data, restored_data, new_data)
        print(computed_data)
    except ValueError as e:
        print("Error: ", e)


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    main()
