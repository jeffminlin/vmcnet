"""Test dynamic width position amplitude routines."""
import jax.numpy as jnp
import numpy as np

import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa


def test_threshold_adjust_std_move_no_adjustment():
    """Test that when mean acceptance is close to target, no adjustment is made."""
    target = 0.5
    threshold_delta = 0.1
    adjust_delta = 0.1
    adjust_std_move_fn = dwpa.make_threshold_adjust_std_move(
        target, threshold_delta, adjust_delta
    )

    old_std_move = 0.3

    mean_acceptance = 0.5
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move)

    mean_acceptance = 0.45
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move)

    mean_acceptance = 0.55
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move)


def test_threshold_adjust_std_move_increase_width():
    """Test that when mean acceptance is above target, step_width is increased."""
    target = 0.5
    threshold_delta = 0.1
    adjust_delta = 0.1
    adjust_std_move_fn = dwpa.make_threshold_adjust_std_move(
        target, threshold_delta, adjust_delta
    )

    old_std_move = 0.3

    mean_acceptance = 0.7
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move * 1.1)

    mean_acceptance = 0.9
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move * 1.1)


def test_threshold_adjust_std_move_decrease_width():
    """Test that when mean acceptance is below target, step_width is decreased."""
    target = 0.5
    threshold_delta = 0.1
    adjust_delta = 0.1
    adjust_std_move_fn = dwpa.make_threshold_adjust_std_move(
        target, threshold_delta, adjust_delta
    )

    old_std_move = 0.3

    mean_acceptance = 0.3
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move * 0.9)

    mean_acceptance = 0.1
    new_std_move = adjust_std_move_fn(old_std_move, mean_acceptance)
    np.testing.assert_allclose(new_std_move, old_std_move * 0.9)


def test_update_move_metadata_fn():
    """Test that update_move_metadata_fn works as expected."""
    nmoves_per_update = 5
    original_std_move = 0.9

    def multiplicative_adjustment(val, accept_avg):
        return val * accept_avg

    move_masks = jnp.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ]
    )
    accept_sums = jnp.array([0.5, 0.5, 1.5, 2.25, 3.0])
    std_move_after_update = 0.54  # original_std_move * mean(move_masks)

    update_metadata_fn = dwpa.make_update_move_metadata_fn(
        nmoves_per_update, multiplicative_adjustment
    )
    metadata = dwpa.MoveMetadata(
        std_move=original_std_move, move_acceptance_sum=0.0, moves_since_update=0
    )

    # Expect no change on first four updates, then multiply by average acceptance
    for i in range(0, 4):
        metadata = update_metadata_fn(metadata, move_masks[i])
        np.testing.assert_allclose(metadata["moves_since_update"], i + 1)
        np.testing.assert_allclose(metadata["move_acceptance_sum"], accept_sums[i])
        np.testing.assert_allclose(metadata["std_move"], original_std_move)

    metadata = update_metadata_fn(metadata, move_masks[4])
    np.testing.assert_allclose(metadata["moves_since_update"], 0)
    np.testing.assert_allclose(metadata["move_acceptance_sum"], 0)
    np.testing.assert_allclose(metadata["std_move"], std_move_after_update)
