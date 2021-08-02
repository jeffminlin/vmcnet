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

    kernel_init = models.weights.get_kernel_initializer("orthogonal")
    bias_init = models.weights.get_bias_initializer("normal")
    residual_blocks = models.construct._get_residual_blocks_for_ferminet_backflow(
        spin_split=spin_split,
        ndense_list=((8,), (10,)),
        kernel_initializer_unmixed=kernel_init,
        kernel_initializer_mixed=kernel_init,
        kernel_initializer_2e_1e_stream=kernel_init,
        kernel_initializer_2e_2e_stream=kernel_init,
        bias_initializer_1e_stream=bias_init,
        bias_initializer_2e_stream=bias_init,
        activation_fn=jnp.tanh,
    )
    backflow = models.equivariance.FermiNetBackflow(
        residual_blocks, ion_pos, include_2e_stream=False
    )

    output_shape_per_spin = ((2, 3), (12,))
    invariant_model = models.invariance.InvariantTensor(
        spin_split=spin_split,
        output_shape_per_spin=output_shape_per_spin,
        backflow=backflow,
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )

    params = invariant_model.init(key, elec_pos)

    output = invariant_model.apply(params, elec_pos)
    perm_output = invariant_model.apply(params, perm_elec_pos)

    chex.assert_shape(
        output, [(nchains,) + output_shape for output_shape in output_shape_per_spin]
    )
    assert_pytree_allclose(output, perm_output, rtol=1e-4)
