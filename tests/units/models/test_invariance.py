"""Test model invariant components."""
import chex
import jax.numpy as jnp
import pytest

import vmcnet.models as models

from tests.test_utils import assert_pytree_allclose
from .utils import get_elec_and_ion_pos_from_hyperparams, get_elec_hyperparams


@pytest.mark.slow
def test_invariant_tensor():
    """Test invariance and shape of InvariantTensor."""
    nchains, nelec_total, nion, d, permutation, spin_split, _ = get_elec_hyperparams()
    key, elec_pos, perm_elec_pos, ion_pos = get_elec_and_ion_pos_from_hyperparams(
        nchains, nelec_total, nion, d, permutation
    )
    stream_1e = elec_pos
    perm_stream_1e = perm_elec_pos

    def backflow(stream_1e, _stream_2e, _r_ei):
        """Simple equivariance with linear and quadratic features."""
        return jnp.concatenate([2.0 * stream_1e, jnp.square(stream_1e)], axis=-1)

    kernel_init = models.weights.get_kernel_initializer("orthogonal")
    bias_init = models.weights.get_bias_initializer("normal")
    output_shape_per_spin = ((2, 3), (12,))
    invariant_model = models.invariance.InvariantTensor(
        spin_split=spin_split,
        output_shape_per_spin=output_shape_per_spin,
        backflow=backflow,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )

    params = invariant_model.init(key, stream_1e, None, None)

    output = invariant_model.apply(params, stream_1e, None, None)
    perm_output = invariant_model.apply(params, perm_stream_1e, None, None)

    chex.assert_shape(
        output, [(nchains,) + output_shape for output_shape in output_shape_per_spin]
    )
    assert_pytree_allclose(output, perm_output, rtol=1e-4)
