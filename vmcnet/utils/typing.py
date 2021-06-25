"""Type definitions that can be reused across the VMCNet codebase."""

from typing import Tuple, TypeVar

import jax.numpy as jnp

# Represents a pytree or pytree-like object containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxilliary MCMC data
D = TypeVar("D")
P = TypeVar("P")  # represents a pytree or pytree-like object containing model params
S = TypeVar("S")  # represents optimizer state

"""
VMC state needed for a checkpoint. Values are:
 1. The epoch
 2. The MCMC walker data
 3. The model parameters
 4. The optimizer state
 5. The RNG key
"""
CheckpointData = Tuple[int, D, P, S, jnp.ndarray]
