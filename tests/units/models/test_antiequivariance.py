"""Test model antiequivariances."""
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models


def test_slog_cofactor_antieq_with_batches():
    base_input = jnp.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    negative_input = -base_input
    doubled_input = base_input * 2
    input = jnp.stack([base_input, negative_input, doubled_input])

    base_output = jnp.array([-3.0, 12.0, -9.0])
    base_signs = jnp.sign(base_output)
    base_logs = jnp.log(jnp.abs(base_output))
    output_signs = jnp.stack([base_signs, -base_signs, base_signs])
    output_logs = jnp.stack([base_logs, base_logs, base_logs + jnp.log(8)])

    y = models.antiequivariance.slog_cofactor_antieq(input)

    np.testing.assert_allclose(y[0], output_signs)
    np.testing.assert_allclose(y[1], output_logs, rtol=1e-6)
