"""Process kernel initializers names."""
from typing import Any, Callable, Dict, Iterable

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

Key = Any
Shape = Iterable[int]
Dtype = Any
WeightInitializer = Callable[[Key, Shape, Dtype], jnp.ndarray]

INITIALIZER_CONSTRUCTORS: Dict[str, Callable] = {
    "zeros": lambda: zeros,
    "ones": lambda: ones,
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


def validate_kernel_initializer(name: str) -> None:
    """Check that a kernel initializer name is in the list of supported kernel inits."""
    if name not in VALID_KERNEL_INITIALIZERS:
        raise ValueError(
            "Invalid kernel initializer requested, {} was requested, but available "
            "initializers are: ".format(name) + ", ".join(VALID_KERNEL_INITIALIZERS)
        )


def get_kernel_initializer(name: str, **kwargs: Any) -> WeightInitializer:
    """Get a kernel initializer."""
    validate_kernel_initializer(name)
    constructor = INITIALIZER_CONSTRUCTORS[name]
    if name == "orthogonal" or name == "delta_orthogonal":
        return constructor(scale=kwargs.get("scale", 1.0))
    else:
        return constructor()


def validate_bias_initializer(name: str) -> None:
    """Check that a bias initializer name is in the list of supported bias inits."""
    if name not in VALID_BIAS_INITIALIZERS:
        raise ValueError(
            "Invalid bias initializer requested, {} was requested, but available "
            "initializers are: ".format(name) + ", ".join(VALID_BIAS_INITIALIZERS)
        )


def get_bias_initializer(name: str) -> WeightInitializer:
    """Get a bias initializer."""
    validate_bias_initializer(name)
    return INITIALIZER_CONSTRUCTORS[name]()
