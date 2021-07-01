"""Tests for slog array helpers."""
from typing import Tuple

import jax.numpy as jnp
import numpy as np

import vmcnet.utils.slog_helpers as helpers
from tests.test_utils import assert_pytree_allclose
from vmcnet.utils.typing import SLArray


def _get_array_and_slog_vals() -> Tuple[jnp.ndarray, SLArray]:
    vals = jnp.array([jnp.e, -jnp.e ** 0.5, 0, 1])
    signs = jnp.array([1, -1, 0, 1])
    logs = jnp.array([1, 0.5, -jnp.inf, 0])
    return (vals, (signs, logs))


def test_array_to_slog():
    """Test conversion from single array to slog tuple."""
    (vals, expected_slogs) = _get_array_and_slog_vals()

    slogs = helpers.array_to_slog(vals)
    assert_pytree_allclose(slogs, expected_slogs)


def test_slog_to_array():
    """Test conversion from slog tuple to single array."""
    (expected_vals, slogs) = _get_array_and_slog_vals()

    vals = helpers.array_from_slog(slogs)
    assert_pytree_allclose(vals, expected_vals)


def test_slog_multiply():
    """Test multiplication of two slog tuples."""
    slog1 = (jnp.array([1, -1, -1]), jnp.array([-1, -2, 5]))
    slog2 = (jnp.array([-1, -1, -1]), jnp.array([7, 10, 2]))
    expected_product = (jnp.array([-1, 1, 1]), jnp.array([6, 8, 7]))

    product = helpers.slog_multiply(slog1, slog2)

    assert_pytree_allclose(product, expected_product)


def test_slog_sum():
    """Test summing over an axis in an slog array."""
    vals = jnp.array([[-1, -2, -3, 6], [1, 2, 3, 4], [0.3, 0.4, 6, -2]])
    expected_sums = jnp.array([0, 10, 4.7])
    sl_vals = helpers.array_to_slog(vals)
    expected_sl_sums = helpers.array_to_slog(expected_sums)

    sl_sums = helpers.slog_sum(sl_vals, axis=-1)

    np.testing.assert_allclose(sl_sums, expected_sl_sums)
