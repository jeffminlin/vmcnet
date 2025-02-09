"""Shared routines for position amplitude metropolis data."""

from typing import Any, Callable, Optional, Tuple, TypedDict

import chex
import jax
import jax.numpy as jnp

import vmcnet.mcmc.metropolis as metropolis
from vmcnet.utils.distribute import (
    replicate_all_local_devices,
    default_distribute_data,
)
from vmcnet.utils.typing import Array, P, PRNGKey, M, ModelApply, UpdateDataFn


class PositionAmplitudeWalkerData(TypedDict):
    """TypedDict of walker data holding just positions and amplitudes.

    Holding both particle position and wavefn amplitude in the same named
    tuple allows us to simultaneously mask over both in the acceptance function.

    The first dimension of position and amplitude should match, but position can have
    more dimensions.

    Attributes:
        position (Array): array of shape (n, ...)
        amplitude (Array): array of shape (n,)
    """

    position: Array
    amplitude: Array


class PositionAmplitudeData(TypedDict):
    """TypedDict of data holding positions, amplitudes, and optional metadata.

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


def make_position_amplitude_data(position: Array, amplitude: Array, move_metadata: Any):
    """Create PositionAmplitudeData from position, amplitude, and move_metadata.

    Args:
        position (Array): the particle positions
        amplitude (Array): the wavefunction amplitudes
        move_metadata (Any): other required metadata for the metropolis algorithm

    Returns:
        PositionAmplitudeData: data containing positions, wavefn amplitudes, and move
        metadata
    """
    return PositionAmplitudeData(
        walker_data=PositionAmplitudeWalkerData(position=position, amplitude=amplitude),
        move_metadata=move_metadata,
    )


def get_position_from_data(data: PositionAmplitudeData) -> Array:
    """Get the position data from PositionAmplitudeData.

    Args:
        data (PositionAmplitudeData): the data

    Returns:
        Array: the particle positions from the data
    """
    return data["walker_data"]["position"]


def get_amplitude_from_data(data: PositionAmplitudeData) -> Array:
    """Get the amplitude data from PositionAmplitudeData.

    Args:
        data (PositionAmplitudeData): the data

    Returns:
        Array: the wave function amplitudes from the data
    """
    return data["walker_data"]["amplitude"]


def to_pam_tuple(data: PositionAmplitudeData) -> Tuple[Array, Array, Any]:
    """Returns data as a (position, amplitude, move_metadata) tuple.

    Useful for quickly assigning all three pieces to local variables for further use.
    """
    return (
        data["walker_data"]["position"],
        data["walker_data"]["amplitude"],
        data["move_metadata"],
    )


def get_update_data_fn(
    model_apply: ModelApply[P],
) -> UpdateDataFn[PositionAmplitudeData, P]:
    """Updates data based on new params, by recalculating amplitudes.

    This needs to be done in the VMC loop each time the params are updated.
    """

    def update_data_fn(data: PositionAmplitudeData, params: P) -> PositionAmplitudeData:
        position = data["walker_data"]["position"]
        amplitude = model_apply(params, position)
        return make_position_amplitude_data(position, amplitude, data["move_metadata"])

    return update_data_fn


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
    walker_data = default_distribute_data(walker_data)
    move_metadata = replicate_all_local_devices(move_metadata)
    return PositionAmplitudeData(walker_data=walker_data, move_metadata=move_metadata)


def make_position_amplitude_gaussian_proposal(
    model_apply: ModelApply[P],
    get_std_move: Callable[[PositionAmplitudeData], chex.Scalar],
) -> Callable[
    [P, PositionAmplitudeData, PRNGKey], Tuple[PositionAmplitudeData, PRNGKey]
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

    def proposal_fn(params: P, data: PositionAmplitudeData, key: PRNGKey):
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
) -> Callable[[P, PositionAmplitudeData, PositionAmplitudeData], Array]:
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
    update_move_metadata_fn: Optional[Callable[[M, Array], M]] = None,
) -> Callable[
    [
        PositionAmplitudeData,
        PositionAmplitudeData,
        Array,
    ],
    PositionAmplitudeData,
]:
    """Factory for an update to PositionAmplitudeData.

    The returned update takes a mask of approved MCMC walker moves `move_mask` and
    accepts those proposed moves from `proposed_data`, for both positions and
    amplitudes. The `std_move` gaussian step width can also be modified by an optional
    `adjust_std_move_fn`.

    The moves in `move_mask` are applied along the first axis of the position data, and
    should be the same shape as the amplitude data (one-dimensional Array).

    Args:
        update_move_metadata_fn (Callable): function which calculates the new
            move_metadata. Has signature
            (old_move_metadata, move_mask) -> new_move_metadata

    Returns:
        Callable: function with signature
            (PositionAmplitudeData, PositionAmplitudeData, Array) ->
                (PositionAmplitudeData),
            which takes in the original PositionAmplitudeData, the proposed
            PositionAmplitudeData, and a move mask. Uses
            the move mask to decide which proposed data to accept.
    """

    def update_position_amplitude(
        data: PositionAmplitudeData,
        proposed_data: PositionAmplitudeData,
        move_mask: Array,
    ) -> PositionAmplitudeData:
        def mask_on_first_dimension(old_data: Array, proposal: Array):
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
    model_apply: ModelApply[P],
    get_std_move: Callable[[PositionAmplitudeData], chex.Scalar],
    update_move_metadata_fn: Optional[Callable[[M, Array], M]] = None,
    logabs: bool = True,
) -> metropolis.MetropolisStep[P, PositionAmplitudeData]:
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
