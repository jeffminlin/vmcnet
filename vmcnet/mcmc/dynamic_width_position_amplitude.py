"""Metropolis routines for position amplitude data with dynamically sized steps."""
from typing import Callable, TypedDict

import chex
import jax

from .metropolis import MetropolisStep
from .position_amplitude_core import (
    make_position_amplitude_data,
    make_position_amplitude_gaussian_metropolis_step,
    PositionAmplitudeWalkerData,
)
from vmcnet.utils.distribute import mean_all_local_devices
from vmcnet.utils.typing import Array, P, ModelApply


class MoveMetadata(TypedDict):
    """Metadata for metropolis algorithm with dynamically sized gaussian steps.

    Attributes:
        std_move (chex.Scalar): the standard deviation of the gaussian step
        move_acceptance_sum (chex.Numeric): the sum of the move acceptance ratios of
            each step taken since the last std_move update. At update time, this sum
            will be divided by moves_since_update to get the overall average, and
            std_move will be adjusted in order to attempt to keep this value near some
            target.
        moves_since_update (int): Number of moves since the last std_move update.
            This is tracked so that the metropolis algorithm can make updates to
            std_move at fixed intervals rather than with every step.
    """

    std_move: chex.Scalar
    move_acceptance_sum: chex.Numeric
    moves_since_update: int


class DynamicWidthPositionAmplitudeData(TypedDict):
    """TypedDict holding positions and wavefunction amplitudes, plus MoveMetadata."""

    walker_data: PositionAmplitudeWalkerData
    move_metadata: MoveMetadata


DWPAData = DynamicWidthPositionAmplitudeData


def make_dynamic_width_position_amplitude_data(
    position: Array,
    amplitude: Array,
    std_move: chex.Scalar,
    move_acceptance_sum: chex.Scalar = 0.0,
    moves_since_update: int = 0,
) -> DWPAData:
    """Create instance of DynamicWidthPositionAmplitudeData.

    Args:
        position (Array): the particle positions
        amplitude (Array): the wavefunction amplitudes
        std_move (chex.Scalar): std for gaussian moves
        move_acceptance_sum (chex.Scalar): sum of the acceptance ratios of each step
            since the last update. Default of 0 should not be altered if using this
            function for initial data.
        moves_since_update (int): the number of moves since the std_move was
            last updated. Default of 0 should not be altered if using this function
            for initial data.

    Returns:
        DWPAData
    """
    return make_position_amplitude_data(
        position,
        amplitude,
        MoveMetadata(
            std_move=std_move,
            move_acceptance_sum=move_acceptance_sum,
            moves_since_update=moves_since_update,
        ),
    )


def make_threshold_adjust_std_move(
    target_acceptance_prob: chex.Scalar = 0.5,
    threshold_delta: chex.Scalar = 0.1,
    adjustment_delta: chex.Scalar = 0.1,
) -> Callable[[chex.Scalar, chex.Scalar], chex.Scalar]:
    """Create a step size adjustment fn which aims to maintain a 50% acceptance rating.

    Works by increasing the step size when the acceptance is at least some delta above
    a target, and decreasing it when the acceptance is the some delta below the target.

    Args:
        target_acceptance_prob (chex.Scalar): target value for the average acceptance
            ratio. Defaults to 0.5.
        threshold_delta (chex.Scalar): how far away from the target the acceptance ratio
            must be to trigger a compensating update. Defaults to 0.1.
        adjustment_delta (chex.Scalar): how big of an adjustment to make to the step
            width. Adjustments will multiply by either (1.0 + adjustment_delta) or
            (1.0 - adjustment_delta). Defaults to 0.1.
    """

    def adjust_std_move(
        old_std_move: chex.Scalar, avg_move_acceptance: chex.Scalar
    ) -> chex.Scalar:
        # Use jax.lax.cond since the predicates are data dependent.
        std_move = jax.lax.cond(
            avg_move_acceptance > target_acceptance_prob + threshold_delta,
            lambda old_val: old_val * (1 + adjustment_delta),
            lambda old_val: old_val,
            old_std_move,
        )
        std_move = jax.lax.cond(
            avg_move_acceptance < target_acceptance_prob - threshold_delta,
            lambda old_val: old_val * (1 - adjustment_delta),
            lambda old_val: old_val,
            std_move,
        )
        return std_move

    return adjust_std_move


def make_update_move_metadata_fn(
    nmoves_per_update: int,
    adjust_std_move_fn: Callable[[chex.Scalar, chex.Scalar], chex.Scalar],
) -> Callable[[MoveMetadata, Array], MoveMetadata]:
    """Create a function that updates the move_metadata periodically.

    Periodicity is controlled by the nmoves_per_update parameter and the logic for
    updating the std of the gaussian step is handled by adjust_std_move_fn.

    Args:
        nmoves_per_update (int): std_move will be updated every time this many
            steps are taken.
        adjust_std_move_fn (Callable): handles the logic for updating std_move.
            Has signature (old_std_move, avg_move_acceptance) -> new_std_move

    Returns:
        Callable: function with signature
            (old_move_metadata, move_mask) -> new_move_metadata
        Result can be fed into the factory for a metropolis step to handle the updating
        of the MoveMetadata.
    """

    def update_move_metadata(
        move_metadata: MoveMetadata, current_move_mask: Array
    ) -> MoveMetadata:
        std_move = move_metadata["std_move"]
        move_acceptance_sum = move_metadata["move_acceptance_sum"]
        moves_since_update = move_metadata["moves_since_update"]

        current_avg_acceptance = mean_all_local_devices(current_move_mask)
        move_acceptance_sum = move_acceptance_sum + current_avg_acceptance
        moves_since_update = moves_since_update + 1

        def update_std_move(_):
            move_acceptance_avg = move_acceptance_sum / moves_since_update
            return (adjust_std_move_fn(std_move, move_acceptance_avg), 0, 0.0)

        def skip_update_std_move(_):
            return (std_move, moves_since_update, move_acceptance_sum)

        (std_move, moves_since_update, move_acceptance_sum) = jax.lax.cond(
            moves_since_update >= nmoves_per_update,
            update_std_move,
            skip_update_std_move,
            operand=None,
        )

        return MoveMetadata(
            std_move=std_move,
            move_acceptance_sum=move_acceptance_sum,
            moves_since_update=moves_since_update,
        )

    return update_move_metadata


def make_dynamic_pos_amp_gaussian_step(
    model_apply: ModelApply[P],
    nmoves_per_update: int = 10,
    adjust_std_move_fn: Callable[
        [chex.Scalar, chex.Scalar], chex.Scalar
    ] = make_threshold_adjust_std_move(),
    logabs: bool = True,
) -> MetropolisStep:
    """Create a metropolis step with dynamic gaussian step width.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        nmoves_per_update (int): number of metropolis steps to take between each
            update to std_move
        adjust_std_move_fn (Callable): handles the logic for updating std_move. Has
            signature (old_std_move, avg_move_acceptance) -> new_std_move
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        Callable: function which runs a metropolis step. Has the signature
            (params, DWPAData, key)
            -> (mean acceptance probability, DWPAData, new_key)
    """
    update_move_metadata_fn = make_update_move_metadata_fn(
        nmoves_per_update, adjust_std_move_fn
    )

    return make_position_amplitude_gaussian_metropolis_step(
        model_apply,
        lambda data: data["move_metadata"]["std_move"],
        update_move_metadata_fn,
        logabs,
    )
