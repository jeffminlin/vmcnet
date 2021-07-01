"""Test core model building parts."""
import chex
import jax.numpy as jnp

import vmcnet.models as models


def test_log_linear_exp_shape():
    """Test output shape of log linear exp."""
    signs = jnp.ones((5, 2, 4, 3))
    vals = jnp.zeros((5, 2, 4, 3))
    weights = jnp.ones((2, 7))

    out = models.core.log_linear_exp(signs, vals, weights, axis=-3)
    out_no_weights = models.core.log_linear_exp(signs, vals, axis=-3)

    desired_shape = (5, 7, 4, 3)
    desired_shape_no_weights = (5, 1, 4, 3)
    chex.assert_shape(out, desired_shape)
    chex.assert_shape(out_no_weights, desired_shape_no_weights)
