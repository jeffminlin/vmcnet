"""Test potential energy functions."""
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics


def _get_test_elec_pos():
    return jnp.array(
        [
            [[0.0, -1.0], [0.0, 1.0]],
            [[2.0, 3.0], [-3.0, 0.0]],
        ]
    )


def _get_test_ions():
    ion_pos = jnp.array([[-4.0, 0.0], [0.0, 0.0], [2.0, 1.0]])
    ion_charges = jnp.array([1.0, 2.0, 3.0])
    return ion_pos, ion_charges


def test_electron_ion_coulomb_potential_value():
    """Test values for electron-ion potentials."""
    elec_pos = _get_test_elec_pos()
    ion_pos, ion_charges = _get_test_ions()

    individual_potentials = -1.0 * jnp.array(
        [
            [
                [1.0 / jnp.sqrt(17.0), 2.0 / 1.0, 3.0 / jnp.sqrt(8.0)],
                [1.0 / jnp.sqrt(17.0), 2.0 / 1.0, 3.0 / 2.0],
            ],
            [
                [1.0 / jnp.sqrt(45.0), 2.0 / jnp.sqrt(13.0), 3.0 / 2.0],
                [1.0 / 1.0, 2.0 / 3.0, 3.0 / jnp.sqrt(26.0)],
            ],
        ]
    )
    desired_out = jnp.sum(individual_potentials, axis=(-1, -2))
    potential_energy_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_pos, ion_charges
    )
    actual_out = potential_energy_fn(None, elec_pos)

    np.testing.assert_allclose(actual_out, desired_out)


def test_electron_ion_coulomb_potential_shape():
    """Test shape for electron-ion potentials."""
    elec_pos = _get_test_elec_pos()
    ion_pos, ion_charges = _get_test_ions()

    potential_energy_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_pos, ion_charges
    )
    out = potential_energy_fn(None, elec_pos)

    nchains = elec_pos.shape[0]
    assert out.shape == (nchains,)


def test_electron_electron_coulomb_potential():
    """Test values/shape for electron-electron potentials."""
    elec_pos = _get_test_elec_pos()
    desired_out = jnp.array([1.0 / 2.0, 1.0 / jnp.sqrt(34.0)])

    potential_energy_fn = physics.potential.create_electron_electron_coulomb_potential()
    actual_out = potential_energy_fn(None, elec_pos)

    np.testing.assert_allclose(actual_out, desired_out)


def test_ion_ion_coulomb_potential():
    """Test value/shape for an ion-ion potential."""
    ion_pos, ion_charges = _get_test_ions()

    # desired output is
    # Z_1 * Z_2 / ||R_1 - R_2|| + Z_1 * Z_3 / ||R_1 - R_3|| + Z_2 * Z_3 / ||R_2 - R_3||
    desired_out = (2.0 / 4.0) + (3.0 / jnp.sqrt(37.0)) + (6.0 / jnp.sqrt(5.0))

    potential_energy_fn = physics.potential.create_ion_ion_coulomb_potential(
        ion_pos, ion_charges
    )
    actual_out = potential_energy_fn(None, None)

    np.testing.assert_allclose(actual_out, desired_out)
