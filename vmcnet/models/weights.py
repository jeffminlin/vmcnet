"""Process kernel initializers names."""
from typing import Any, Callable, Dict, Iterable

import jax.numpy as jnp
from jax.nn.initializers import (
    zeros,
    ones,
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


def validate_initializer(name: str) -> None:
    valid_initializers = INITIALIZER_CONSTRUCTORS.keys()
    if name not in valid_initializers:
        raise ValueError(
            "Invalid weight initializer requested, {} was requested, but available "
            "initializers are: ".format(name) + ", ".join(valid_initializers)
        )


def get_initializer(name: str, **kwargs: Any) -> WeightInitializer:
    validate_initializer(name)
    constructor = INITIALIZER_CONSTRUCTORS[name]
    if name == "orthogonal" or "delta_orthogonal":
        return constructor(kwargs.get("scale", 1.0))
    else:
        return constructor()
