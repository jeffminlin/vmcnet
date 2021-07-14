"""Type definitions that can be reused across the VMCNet codebase.

Because type-checking with numpy/jax numpy can be tricky and does not always agree with
type-checkers, this package uses types for static type-checking when possible, but
otherwise they are intended for documentation and clarity.
"""
from typing import Any, Callable, List, Sequence, Tuple, TypeVar, Union

import jax.numpy as jnp
import kfac_ferminet_alpha.optimizer as kfac_opt
import optax


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

# Actual optimizer states currently used
OptimizerState = Union[kfac_opt.State, optax.OptState]

# VMC state needed for a checkpoint. Values are:
#  1. The epoch
#  2. The MCMC walker data
#  3. The model parameters
#  4. The optimizer state
#  5. The RNG key
CheckpointData = Tuple[int, D, P, S, jnp.ndarray]

ArrayList = List[jnp.ndarray]

# Single array in (sign, logabs) form
SLArray = Tuple[jnp.ndarray, jnp.ndarray]

SLArrayList = List[SLArray]

SpinSplit = Union[int, Sequence[int]]

ModelApply = Callable[[P, jnp.ndarray], jnp.ndarray]
