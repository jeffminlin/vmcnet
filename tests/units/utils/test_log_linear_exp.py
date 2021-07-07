"""Tests for log_linear_exp function."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.utils.log_linear_exp import log_linear_exp
import vmcnet.utils.slog_helpers as slog_helpers


def test_log_linear_exp_shape():
    """Test output shape of log linear exp."""
    signs = jnp.ones((5, 2, 4, 3))
    vals = jnp.zeros((5, 2, 4, 3))
    weights = jnp.ones((2, 7))

    out = log_linear_exp(signs, vals, weights, axis=-3)
    out_no_weights = log_linear_exp(signs, vals, axis=-3)

    desired_shape = (5, 7, 4, 3)
    desired_shape_no_weights = (5, 1, 4, 3)
    chex.assert_shape(out, desired_shape)
    chex.assert_shape(out_no_weights, desired_shape_no_weights)


def test_log_linear_exp_no_overflow():
    """Test that the log-linear-exp trick avoids overflow when any vals are big."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([300.0, 100.0, 3000.0, 1.5])
    weights = jnp.reshape(jnp.array([-1.0, 2.0, 0.5, 0.6]), (4, 1))

    sign_out, log_out = log_linear_exp(signs, vals, weights, axis=0)

    # the output should be sign_out=1.0, log_out=log|0.5 * exp(3000) + tinier stuff|
    assert jnp.isfinite(log_out)
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, jnp.log(0.5) + 3000.0)


def test_log_linear_exp_no_underflow():
    """Test that the log-linear-exp trick avoids underflow when all vals are small."""
    signs = jnp.array([-1.0, -1.0, 1.0, 1.0])
    vals = jnp.array([-4000.0, -5500.0, -3000.0, -1234.5])

    sign_out, log_out = log_linear_exp(signs, vals, axis=0)

    # the output should be sign_out=1.0, log_out=log|exp(-1234.5) + tinier stuff|
    np.testing.assert_allclose(sign_out, 1.0)
    np.testing.assert_allclose(log_out, -1234.5)


def test_log_linear_equals_log_linear_exp_log():
    """Test that log-linear-exp of sign(x), log|x| is just log-linear."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (9, 5))
    sign_x, log_x = slog_helpers.array_to_slog(x)

    key, subkey = jax.random.split(key)
    kernel = jax.random.normal(subkey, (5, 7))

    sign_linear_out, log_linear_out = slog_helpers.array_to_slog(jnp.dot(x, kernel))
    sign_linear_exp_log_out, log_linear_exp_log_out = log_linear_exp(
        sign_x, log_x, kernel, axis=-1
    )

    np.testing.assert_allclose(sign_linear_exp_log_out, sign_linear_out)
    np.testing.assert_allclose(log_linear_exp_log_out, log_linear_out, rtol=1e-5)
