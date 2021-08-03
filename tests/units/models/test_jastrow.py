"""Testing jastrow factors."""
import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.models as models


def _get_ei_and_ee(elec_pos, ion_pos):
    r_ei = jnp.expand_dims(elec_pos, axis=-2) - jnp.expand_dims(ion_pos, axis=-3)
    r_ee = jnp.expand_dims(elec_pos, axis=-2) - jnp.expand_dims(elec_pos, axis=-3)
    return r_ei, r_ee


def _get_ion_and_elec_pos_and_mol_decay_jastrow():
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
    jastrow = models.jastrow.get_mol_decay_scaled_for_chargeless_molecules(
        ion_pos, ion_charges
    )

    return ion_pos, elec_pos, jastrow


def test_log_molecular_decay_jastrow_close_to_zero():
    """Test that you get 0 from the log molecular decay when all elecs are at nuclei."""
    ion_pos, elec_pos, jastrow = _get_ion_and_elec_pos_and_mol_decay_jastrow()

    r_ei, r_ee = _get_ei_and_ee(elec_pos, ion_pos)
    np.testing.assert_allclose(jastrow(r_ei, r_ee), 0.0)


def test_log_molecular_decay_jastrow_close_to_linear():
    """Test that you get -Z * r from log mol decay when one elec is far from nuclei.

    Here Z means the sum of the charges of other particles (ions and electrons).
    """
    ion_pos, elec_pos, jastrow = _get_ion_and_elec_pos_and_mol_decay_jastrow()

    elec_pos = jax.ops.index_update(elec_pos, (0, 0, 2), 2e10)  # put one far away
    elec_pos = elec_pos[:, :-2, :]  # remove two electrons
    r_ei, r_ee = _get_ei_and_ee(elec_pos, ion_pos)
    np.testing.assert_allclose(jastrow(r_ei, r_ee), 3 * -2e10)
