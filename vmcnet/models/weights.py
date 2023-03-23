"""Functions to get weight initializers from names."""
import functools
from typing import Any, Callable, Dict, Sequence, Union

import chex
import jax.numpy as jnp
from jax.nn.initializers import (
    zeros,
    ones,
    uniform,
    normal,
    orthogonal,
    delta_orthogonal,
    xavier_normal,
    xavier_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
)
from ml_collections import ConfigDict

Key = Any
Shape = Sequence[Union[int, Any]]
Dtype = Any
WeightInitializer = Callable[[Key, Shape, Dtype], Any]

INITIALIZER_CONSTRUCTORS: Dict[str, Callable] = {
    "zeros": lambda dtype=jnp.float32: functools.partial(zeros, dtype=dtype),
    "ones": lambda dtype=jnp.float32: functools.partial(ones, dtype=dtype),
    "uniform": uniform,
    "normal": normal,
    "orthogonal": orthogonal,
    "delta_orthogonal": delta_orthogonal,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "glorot_normal": xavier_normal,
    "glorot_uniform": xavier_uniform,
    "kaiming_normal": kaiming_normal,
    "kaiming_uniform": kaiming_uniform,
    "he_normal": kaiming_normal,
    "he_uniform": kaiming_uniform,
    "lecun_normal": lecun_normal,
    "lecun_uniform": lecun_uniform,
}

VALID_KERNEL_INITIALIZERS = INITIALIZER_CONSTRUCTORS.keys()

VALID_BIAS_INITIALIZERS = ["zeros", "ones", "uniform", "normal"]


# TODO: clean up initializer getting methods
def validate_kernel_initializer(name: str) -> None:
    """Check that a kernel initializer name is in the list of supported kernel inits."""
    if name not in VALID_KERNEL_INITIALIZERS:
        raise ValueError(
            "Invalid kernel initializer requested, {} was requested, but available "
            "initializers are: ".format(name) + ", ".join(VALID_KERNEL_INITIALIZERS)
        )


def get_kernel_initializer(
    name: str, dtype=jnp.float32, **kwargs: Any
) -> WeightInitializer:
    """Get a kernel initializer."""
    validate_kernel_initializer(name)
    constructor = INITIALIZER_CONSTRUCTORS[name]
    if name == "orthogonal" or name == "delta_orthogonal":
        return constructor(scale=kwargs.get("scale", 1.0), dtype=dtype)
    else:
        return constructor(dtype=dtype)


def get_kernel_init_from_config(config: ConfigDict, dtype=jnp.float32):
    """Get a kernel initializer from a ConfigDict.

    The ConfigDict should have the key "type", as well as any other kwargs to pass
    to the initializer constructor.
    """
    return get_kernel_initializer(config.type, dtype=dtype, **config)


def validate_bias_initializer(name: str) -> None:
    """Check that a bias initializer name is in the list of supported bias inits."""
    if name not in VALID_BIAS_INITIALIZERS:
        raise ValueError(
            "Invalid bias initializer requested, {} was requested, but available "
            "initializers are: ".format(name) + ", ".join(VALID_BIAS_INITIALIZERS)
        )


def get_bias_initializer(name: str, dtype=jnp.float32) -> WeightInitializer:
    """Get a bias initializer."""
    validate_bias_initializer(name)
    return INITIALIZER_CONSTRUCTORS[name](dtype=dtype)


def get_bias_init_from_config(config, dtype=jnp.float32):
    """Get a bias initializer from a ConfigDict.

    The ConfigDict should have the key "type", as well as any other kwargs to pass
    to the initializer constructor.
    """
    return get_bias_initializer(config.type, dtype=dtype)


def get_constant_init(constant: chex.Numeric):
    """Get a weight initializer for a constant array with specified dtype, ignoring key.

    Args:
        constant (float): the number to initialize to
    """

    def init_fn(key, shape, dtype=jnp.float32):
        del key
        return jnp.ones(shape, dtype=dtype) * jnp.array(constant, dtype=dtype)

    return init_fn
