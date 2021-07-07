"""Tests for slog array helpers."""
from typing import Tuple

import chex
import jax
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


def test_log_linear_exp_shape():
    """Test output shape of log linear exp."""
    signs = jnp.ones((5, 2, 4, 3))
    vals = jnp.zeros((5, 2, 4, 3))
    weights = jnp.ones((2, 7))

    out = helpers.slog_linear_comb((signs, vals), weights, axis=-3)
    out_no_weights = helpers.slog_linear_comb((signs, vals), axis=-3)

    desired_shape = (5, 7, 4, 3)
    desired_shape_no_weights = (5, 1, 4, 3)
    chex.assert_shape(out, desired_shape)
    chex.assert_shape(out_no_weights, desired_shape_no_weights)


def test_log_linear_exp_no_overflow():
    """Test that the log-linear-exp trick avoids overflow when any vals are big."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([300.0, 100.0, 3000.0, 1.5])
    weights = jnp.reshape(jnp.array([-1.0, 2.0, 0.5, 0.6]), (4, 1))

    sign_out, log_out = helpers.slog_linear_comb((signs, vals), weights, axis=0)

    # the output should be sign_out=1.0, log_out=log|0.5 * exp(3000) + tinier stuff|
    assert jnp.isfinite(log_out)
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, jnp.log(0.5) + 3000.0)


def test_log_linear_exp_no_underflow():
    """Test that the log-linear-exp trick avoids underflow when all vals are small."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([-4000.0, -5500.0, -3000.0, -1234.5])

    sign_out, log_out = helpers.slog_linear_comb((signs, vals), axis=0)

    # the output should be sign_out=1.0, log_out=log|exp(-1234.5) + tinier stuff|
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, -1234.5)


def test_log_linear_equals_log_linear_exp_log():
    """Test that log-linear-exp of sign(x), log|x| is just log-linear."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (9, 5))
    slog_x = helpers.array_to_slog(x)

    key, subkey = jax.random.split(key)
    kernel = jax.random.normal(subkey, (5, 7))

    sign_linear_out, log_linear_out = helpers.array_to_slog(jnp.dot(x, kernel))
    sign_linear_exp_log_out, log_linear_exp_log_out = helpers.slog_linear_comb(
        slog_x, kernel, axis=-1
    )

    np.testing.assert_allclose(sign_linear_exp_log_out, sign_linear_out)
    np.testing.assert_allclose(log_linear_exp_log_out, log_linear_out, rtol=1e-5)
