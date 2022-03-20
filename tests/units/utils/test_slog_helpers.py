"""Tests for slog array helpers."""
from typing import Tuple

import jax.numpy as jnp

import vmcnet.utils.slog_helpers as helpers
from tests.test_utils import assert_pytree_allclose
from vmcnet.utils.typing import Array, SLArray


def _get_array_and_slog_vals() -> Tuple[Array, SLArray]:
    vals = jnp.array([jnp.e, -jnp.e**0.5, 0, 1])
    signs = jnp.array([1, -1, 0, 1])
    logs = jnp.array([1, 0.5, -jnp.inf, 0])
    return (vals, (signs, logs))


def test_array_to_slog():
    """Test conversion from array to slog tuple."""
    (vals, expected_slogs) = _get_array_and_slog_vals()

    slogs = helpers.array_to_slog(vals)
    assert_pytree_allclose(slogs, expected_slogs)


def test_slog_to_array():
    """Test conversion from slog tuple to array."""
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


def test_slog_sum_over_axis():
    """Test sum of slog array over an axis."""
    (vals, slogs) = _get_array_and_slog_vals()
    log_sum_vals = helpers.array_to_slog(jnp.sum(vals, 0, keepdims=False))

    sum_log_vals = helpers.slog_sum_over_axis(slogs, 0)

    assert_pytree_allclose(log_sum_vals, sum_log_vals, rtol=1e-6)


def test_sl_array_list_sum():
    """Test sum of an SLArrayList."""
    vals = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([-1.0, 4.0, -0.7]),
        jnp.array([0.0, 10.0, -4.0]),
    ]
    slogs = helpers.array_list_to_slog(vals)

    sum = helpers.slog_array_list_sum(slogs)
    expected_sum = helpers.array_to_slog(jnp.array([0.0, 16.0, -1.7]))
    assert_pytree_allclose(sum, expected_sum, rtol=1e-6)


def test_slog_sum():
    """Test sum of an SLArrayList."""
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([-1.0, 4.0, -0.7])
    slog_x = helpers.array_to_slog(x)
    slog_y = helpers.array_to_slog(y)

    sum = helpers.slog_sum(slog_x, slog_y)
    expected_sum = helpers.array_to_slog(jnp.array([0.0, 6.0, 2.3]))
    assert_pytree_allclose(sum, expected_sum)
