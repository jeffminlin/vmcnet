"""Shared routines for position amplitude metropolis data."""
from typing import Any, Callable, Optional, Tuple, TypeVar, TypedDict

import jax
import jax.numpy as jnp
import vmcnet.mcmc.metropolis as metropolis
from vmcnet.utils.distribute import (
    replicate_all_local_devices,
    distribute_data,
)

# Represents a pytree or pytree-like object containing model params
P = TypeVar("P")
# Represents metadata which is required to take a metropolis step.
M = TypeVar("M")


class PositionAmplitudeWalkerData(TypedDict):
    """NamedTuple of walker data holding just positions and amplitudes.

    Holding both particle position and wavefn amplitude in the same named
    tuple allows us to simultaneously mask over both in the acceptance function.

    The first dimension of position and amplitude should match, but position can have
    more dimensions.

    Attributes:
        position (jnp.ndarray): array of shape (n, ...)
        amplitude (jnp.ndarray): array of shape (n,)
    """

    position: jnp.ndarray
    amplitude: jnp.ndarray


class PositionAmplitudeData(TypedDict):
    """NamedTuple of data holding positions, amplitudes, and optional metadata.

    Holding both particle position and wavefn amplitude in the data can be advantageous
    to avoid recalculating amplitudes in some routines, e.g. acceptance probabilities.
    Furthermore, holding additional metadata can enable more sophisticated metropolis
    algorithms such as dynamically adjusted gaussian step sizes.

    Attributes:
        walker_data (PositionAmplitudeWalkerData): the positions and amplitudes
        move_metadata (any, optional): any metadata needed for the metropolis algorithm
    """

    walker_data: PositionAmplitudeWalkerData
    move_metadata: Any


def make_position_amplitude_data(
    position: jnp.ndarray, amplitude: jnp.ndarray, move_metadata: Any
):
    """Create PositionAmplitudeData from position, amplitude, and move_metadata.

    Args:
        position (jnp.ndarray): the particle positions
        amplitude (jnp.ndarray): the wavefunction amplitudes
        move_metadata (Any): other required metadata for the metropolis algorithm

    Returns:
        PositionAmplitudeData
    """
    return PositionAmplitudeData(
        walker_data=PositionAmplitudeWalkerData(position=position, amplitude=amplitude),
        move_metadata=move_metadata,
    )


def get_position_from_data(data: PositionAmplitudeData) -> jnp.ndarray:
    """Get the position data from PositionAmplitudeData.

    Args:
        data (PositionAmplitudeData): the data

    Returns:
        jnp.ndarray: the particle positions from the data
    """
    return data["walker_data"]["position"]


def to_pam_tuple(data: PositionAmplitudeData) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
    return (
        data["walker_data"]["position"],
        data["walker_data"]["amplitude"],
        data["move_metadata"],
    )


def distribute_position_amplitude_data(
    data: PositionAmplitudeData,
) -> PositionAmplitudeData:
    """Distribute PositionAmplitudeData across devices.

    Args:
        data (PositionAmplitudeData): the data to distribute

    Returns:
        PositionAmplitudeData: the distributed data.
    """
    walker_data = data["walker_data"]
    move_metadata = data["move_metadata"]
    walker_data = distribute_data(walker_data)
    move_metadata = replicate_all_local_devices(move_metadata)
    return PositionAmplitudeData(walker_data=walker_data, move_metadata=move_metadata)


def make_position_amplitude_gaussian_proposal(
    model_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    get_std_move: Callable[[PositionAmplitudeData], jnp.float32],
) -> Callable[
    [P, PositionAmplitudeData, jnp.ndarray],
    Tuple[PositionAmplitudeData, jnp.ndarray],
]:
    """Create a gaussian proposal fn on PositionAmplitudeData.

    Positions are perturbed by a guassian; amplitudes are evaluated using the supplied
    model; move_metadata is not modified.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        get_std_move (Callable): function which gets the standard deviation of the
        gaussian move, which can optionally depend on the data. Has signature
        (PositionAmplitudeData) -> std_move

    Returns:
        Callable: proposal function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, key) -> (PositionAmplitudeData, key).
    """

    def proposal_fn(params: P, data: PositionAmplitudeData, key: jnp.float32):
        std_move = get_std_move(data)
        proposed_position, key = metropolis.gaussian_proposal(
            data["walker_data"]["position"], std_move, key
        )
        proposed_amplitude = model_apply(params, proposed_position)
        return (
            make_position_amplitude_data(
                proposed_position, proposed_amplitude, data["move_metadata"]
            ),
            key,
        )

    return proposal_fn


def make_position_amplitude_metropolis_symmetric_acceptance(
    logabs: bool = True,
) -> Callable[[P, PositionAmplitudeData, PositionAmplitudeData], jnp.ndarray]:
    """Create a Metropolis acceptance function on PositionAmplitudeData.

    Args:
        logabs (bool, optional): whether amplitudes provided to `acceptance_fn`
            represent psi (logabs = False) or log|psi| (logabs = True). Defaults to
            True.

    Returns:
        Callable: acceptance function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, PositionAmplitudeData) -> accept_ratio
    """

    def acceptance_fn(
        params: P, data: PositionAmplitudeData, proposed_data: PositionAmplitudeData
    ):
        del params
        return metropolis.metropolis_symmetric_acceptance(
            data["walker_data"]["amplitude"],
            proposed_data["walker_data"]["amplitude"],
            logabs=logabs,
        )

    return acceptance_fn


def make_position_amplitude_update(
    update_move_metadata_fn: Optional[Callable[[M, jnp.ndarray], M]] = None
) -> Callable[
    [
        PositionAmplitudeData,
        PositionAmplitudeData,
        jnp.ndarray,
    ],
    PositionAmplitudeData,
]:
    """Factory for an update to PositionAmplitudeData.

    The returned update takes a mask of approved MCMC walker moves `move_mask` and
    accepts those proposed moves from `proposed_data`, for both positions and
    amplitudes. The `std_move` gaussian step width can also be modified by an optional
    `adjust_std_move_fn`.

    The moves in `move_mask` are applied along the first axis of the position data, and
    should be the same shape as the amplitude data (one-dimensional jnp.ndarray).

    Args:
        update_move_metadata_fn (Callable): function which calculates the new
        move_metadata. Has signature (old_move_metadata, move_mask) -> new_move_metadata

    Returns:
        Callable: function with signature
            (PositionAmplitudeData, PositionAmplitudeData, jnp.ndarray) ->
                (PositionAmplitudeData),
            which takes in the original PositionAmplitudeData, the proposed
            PositionAmplitudeData, and a move mask. Uses
            the move mask to decide which proposed data to accept.
    """

    def update_position_amplitude(
        data: PositionAmplitudeData,
        proposed_data: PositionAmplitudeData,
        move_mask: jnp.ndarray,
    ) -> PositionAmplitudeData:
        def mask_on_first_dimension(old_data: jnp.ndarray, proposal: jnp.ndarray):
            shaped_mask = jnp.reshape(move_mask, (-1, *((1,) * (old_data.ndim - 1))))
            return jnp.where(shaped_mask, proposal, old_data)

        new_walker_data = jax.tree_map(
            mask_on_first_dimension, data["walker_data"], proposed_data["walker_data"]
        )

        new_move_metadata = proposed_data["move_metadata"]
        if update_move_metadata_fn is not None:
            new_move_metadata = update_move_metadata_fn(
                data["move_metadata"], move_mask
            )

        return PositionAmplitudeData(
            walker_data=new_walker_data, move_metadata=new_move_metadata
        )

    return update_position_amplitude


def make_position_amplitude_gaussian_metropolis_step(
    model_apply: Callable[[P, jnp.ndarray], jnp.ndarray],
    get_std_move: Callable[[PositionAmplitudeData], jnp.float32],
    update_move_metadata_fn: Optional[Callable[[M, jnp.ndarray], M]] = None,
    logabs: bool = True,
) -> Callable[
    [P, PositionAmplitudeData, jnp.ndarray],
    Tuple[jnp.float32, PositionAmplitudeData, jnp.ndarray],
]:
    """Make a gaussian proposal with Metropolis acceptance for PositionAmplitudeData.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        get_std_move (Callable): function which gets the standard deviation of the
            gaussian move, which can optionally depend on the data. Has signature
            (PositionAmplitudeData) -> std_move
        update_move_metadata_fn (Callable, optional): function which calculates the new
            move_metadata. Has signature
            (old_move_metadata, move_mask) -> new_move_metadata.
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        Callable: function which does a metropolis step. Has the signature
            (params, PositionAmplitudeData, key)
            -> (mean acceptance probability, PositionAmplitudeData, new_key)
    """
    proposal_fn = make_position_amplitude_gaussian_proposal(model_apply, get_std_move)
    accept_fn = make_position_amplitude_metropolis_symmetric_acceptance(logabs=logabs)
    metrop_step_fn = metropolis.make_metropolis_step(
        proposal_fn,
        accept_fn,
        make_position_amplitude_update(update_move_metadata_fn),
    )
    return metrop_step_fn
