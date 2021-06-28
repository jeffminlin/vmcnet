"""Type definitions that can be reused across the VMCNet codebase."""

from typing import Any, Tuple

import jax.numpy as jnp

"""
VMC state needed for a checkpoint. Values are:
 1. The epoch
 2. The MCMC walker data
 3. The model parameters
 4. The optimizer state
 5. The RNG key
"""
CheckpointData = Tuple[int, Any, Any, Any, jnp.ndarray]
