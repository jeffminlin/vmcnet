"""Type definitions that can be reused across the VMCNet codebase."""
from typing import Any, Tuple, TypeVar

import jax.numpy as jnp


PyTree = Any

# TypeVar for an arbitrary PyTree
T = TypeVar("T", bound=PyTree)

# TypeVar for a pytree containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D", bound=PyTree)

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
