"""Type definitions that can be reused across the VMCNet codebase.

Because type-checking with numpy/jax numpy can be tricky and does not always agree with
type-checkers, this package uses types for static type-checking when possible, but
otherwise they are intended for documentation and clarity.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from jax import Array
from jax.typing import ArrayLike
import kfac_jax
import optax


PRNGKey = Array

# Currently using PyTree = Any just to improve readability in the code.
# A pytree is a "tree-like structure built out of container-like Python objects": see
# https://jax.readthedocs.io/en/latest/pytrees.html
PyTree = Any

# TypeVar for an arbitrary PyTree
T = TypeVar("T", bound=PyTree)

# TypeVar for a pytree containing MCMC data, e.g. walker positions
# and wave function amplitudes, or other auxiliary MCMC data
D = TypeVar("D", bound=PyTree)

# TypeVar for MCMC metadata which is required to take a metropolis step.
M = TypeVar("M", bound=PyTree)

# TypeVar for a pytree containing model params
P = TypeVar("P", bound=PyTree)

# TypeVar for a pytree containing optimizer state
S = TypeVar("S", bound=PyTree)

# Actual optimizer states currently used
# TODO: Figure out how to make kfac_opt.State not be interpreted by mypy as Any
OptimizerState = Union[kfac_jax.optimizer.OptimizerState, optax.OptState]

LearningRateSchedule = Callable[[Array], Array]

ModelParams = Dict[str, Any]

# VMC state needed for a checkpoint. Values are:
#  1. The epoch
#  2. The MCMC walker data
#  3. The model parameters
#  4. The optimizer state
#  5. The RNG key
CheckpointData = Tuple[int, D, P, S, PRNGKey]

ArrayList = List[Array]

# Single array in (sign, logabs) form
SLArray = Tuple[Array, Array]

SLArrayList = List[SLArray]

ParticleSplit = Union[int, Sequence[int]]

InputStreams = Tuple[Array, Optional[Array], Optional[Array], Optional[Array]]
ComputeInputStreams = Callable[[Array], InputStreams]

Backflow = Callable[[Array, Optional[Array]], Array]

Jastrow = Callable[[Array, Array, Array, Array, Array], Array]
CuspJastrowType = Callable[[Array, Array], Array]

ModelApply = Callable[[P, Array], Array]
LocalEnergyApply = Callable[[P, Array], Array]

GetPositionFromData = Callable[[D], Array]
GetAmplitudeFromData = GetPositionFromData[D]
UpdateDataFn = Callable[[D, P], D]

ClippingFn = Callable[[Array, ArrayLike], Array]
