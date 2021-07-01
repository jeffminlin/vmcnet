"""Type definitions that can be reused across the VMCNet codebase.

Because type-checking with numpy/jax numpy can be tricky and does not always agree with
type-checkers, this package uses types for static type-checking when possible, but
otherwise they are intended for documentation and clarity.
"""
from typing import Any, List, Tuple, TypeVar

import jax.numpy as jnp


# Currently using PyTree = Any just to improve readability in the code.
# A pytree is a "tree-like structure built out of container-like Python objects": see
# https://jax.readthedocs.io/en/latest/pytrees.html
PyTree = Any

# TypeVar for an arbitrary PyTree
T = TypeVar("T", bound=PyTree)

# TypeVar for a pytree containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D", bound=PyTree)

# TypeVar for MCMC metadata which is required to take a metropolis step.
M = TypeVar("M", bound=PyTree)

# TypeVar for a pytree containing model params
P = TypeVar("P", bound=PyTree)

# TypeVar for a pytree containing optimizer state
S = TypeVar("S", bound=PyTree)

# VMC state needed for a checkpoint. Values are:
#  1. The epoch
#  2. The MCMC walker data
#  3. The model parameters
#  4. The optimizer state
#  5. The RNG key
CheckpointData = Tuple[int, D, P, S, jnp.ndarray]

SpinSplitArray = List[jnp.ndarray]
SLArray = Tuple[jnp.ndarray, jnp.ndarray]
SpinSplitSLArray = List[SLArray]
