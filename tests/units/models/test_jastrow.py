"""Testing jastrow factors."""
import chex
import jax.numpy as jnp
import numpy as np
import pytest

import vmcnet.models as models

from .utils import (
    get_elec_hyperparams,
    get_input_streams_from_hyperparams,
    simple_backflow,
)


def _get_ei_and_ee(elec_pos, ion_pos):
    r_ei = jnp.expand_dims(elec_pos, axis=-2) - jnp.expand_dims(ion_pos, axis=-3)
    r_ee = jnp.expand_dims(elec_pos, axis=-2) - jnp.expand_dims(elec_pos, axis=-3)
    return r_ei, r_ee


def _get_ion_and_elec_pos_and_scaled_mol_decay_jastrow():
    ion_charges = jnp.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
    ion_pos = jnp.array(
        [
            [0.0, 0.0, -2.5],
            [0.0, 0.0, -1.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.5],
        ]
    )
    elec_pos = jnp.expand_dims(
        jnp.array(
            [
                [0.0, 0.0, -2.5],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, -1.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, -0.5],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 2.5],
            ]
        ),
        axis=0,
    )
    jastrow = models.jastrow.get_two_body_decay_scaled_for_chargeless_molecules(
        ion_pos, ion_charges, trainable=False
    )

    return ion_pos, elec_pos, jastrow


def test_scaled_log_molecular_decay_jastrow_close_to_zero():
    """Test that you get 0 from the log molecular decay when all elecs are at nuclei."""
    ion_pos, elec_pos, jastrow = _get_ion_and_elec_pos_and_scaled_mol_decay_jastrow()

    r_ei, r_ee = _get_ei_and_ee(elec_pos, ion_pos)
    np.testing.assert_allclose(jastrow.apply({}, None, None, None, r_ei, r_ee), 0.0)


def test_log_molecular_decay_jastrow_close_to_linear():
    """Test that you get -Z * r from log mol decay when one elec is far from nuclei.

    Here Z means the sum of the charges of other particles (ions and electrons).
    """
    ion_pos, elec_pos, jastrow = _get_ion_and_elec_pos_and_scaled_mol_decay_jastrow()

    elec_pos = elec_pos.at[0, 0, 2].set(2e10)  # put one very far away
    elec_pos = elec_pos[:, :-2, :]  # remove two electrons
    r_ei, r_ee = _get_ei_and_ee(elec_pos, ion_pos)
    np.testing.assert_allclose(
        jastrow.apply({}, None, None, None, r_ei, r_ee), 3 * -2e10, rtol=1e-6
    )


@pytest.mark.slow
def test_backflow_based_jastrow_with_separate_backflow_is_perm_invariant():
    """Test permutation invariance of backflow-based jastrow with separate backflow."""
    nchains, nelec_total, nion, d, permutation, _, _ = get_elec_hyperparams()
    (
        input_1e,
        input_2e,
        _,
        perm_input_1e,
        perm_input_2e,
        _,
        key,
    ) = get_input_streams_from_hyperparams(nchains, nelec_total, nion, d, permutation)

    jastrow = models.jastrow.BackflowJastrow(simple_backflow)

    params = jastrow.init(key, input_1e, input_2e, None, None, None)
    out = jastrow.apply(params, input_1e, input_2e, None, None, None)
    perm_out = jastrow.apply(params, perm_input_1e, perm_input_2e, None, None, None)

    chex.assert_shape(out, (nchains,))
    np.testing.assert_allclose(out, perm_out, rtol=1e-6)
