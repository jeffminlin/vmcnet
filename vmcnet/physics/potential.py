"""Potential energy terms."""
from typing import Tuple
import jax.numpy as jnp
import jax

from vmcnet.utils.typing import Array, ModelApply, ModelParams


def _compute_displacements(x: Array, y: Array) -> Array:
    """Compute the pairwise displacements between x and y in the second-to-last dim.

    Args:
        x (Array): array of shape (..., n_x, d)
        y (Array): array of shape (..., n_y, d)

    Returns:
        Array: pairwise displacements (x_i - y_j), with shape (..., n_x, n_y, d)
    """
    return jnp.expand_dims(x, axis=-2) - jnp.expand_dims(y, axis=-3)



def _compute_soft_norm(
    displacements: Array, softening_term: jnp.float32 = 0.0
) -> Array:
    """Compute an (optionally softened) norm, sqrt((sum_i x_i^2) + softening_term^2).

    Args:
        displacements (Array): array of shape (..., d)
        softening_term (jnp.float32, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt. The smaller this term, the closer the
            derivative gets to a step function (but the derivative is continuous except
            for for softening term exactly equal to zero!). When zero, gives the usual
            vector 2-norm. Defaults to 0.0.

    Returns:
        Array: array with shape displacements.shape[:-1]
    """
    return jnp.sqrt(
        jnp.sum(jnp.square(displacements), axis=-1) + jnp.square(softening_term)
    )


def _get_ion_ion_info(ion_locations: Array, ion_charges: Array) -> Tuple[Array, Array]:
    """Get pairwise ion-ion displacements and charge-charge products."""
    ion_ion_displacements = _compute_displacements(ion_locations, ion_locations)
    charge_charge_prods = jnp.expand_dims(ion_charges, axis=-1) * ion_charges
    return ion_ion_displacements, charge_charge_prods


def create_electron_ion_coulomb_potential(
    ion_locations: Array,
    ion_charges: Array,
    strength: jnp.float32 = 1.0,
    softening_term: jnp.float32 = 0.0,
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential attraction between electron and ion.

    Args:
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron)
        strength (jnp.float32, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.
        softening_term (jnp.float32, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt in the norm calculation. When zero, the
            usual vector 2-norm is used to compute distance. Defaults to 0.0.

    Returns:
        Callable: function which computes the potential energy due to the attraction
        between electrons and ion. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params
        electron_ion_displacements = _compute_displacements(x, ion_locations)
        electron_ion_distances = _compute_soft_norm(
            electron_ion_displacements, softening_term=softening_term
        )
        coulomb_attraction = ion_charges / electron_ion_distances
        return -strength * jnp.sum(coulomb_attraction, axis=(-1, -2))

    return potential_fn


def create_electron_electron_coulomb_potential(
    strength: jnp.float32 = 1.0, softening_term: jnp.float32 = 0.0
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential repulsion between pairs of electrons.

    Args:
        strength (jnp.float32, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.
        softening_term (jnp.float32, optional): this amount squared is added to
            sum_i x_i^2 before taking the sqrt in the norm calculation. When zero, the
            usual vector 2-norm is used to compute distance. Defaults to 0.0.

    Returns:
        Callable: function which computes the potential energy due to the repulsion
        between pairs of electrons. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params
        electron_electron_displacements = _compute_displacements(x, x)
        electron_electron_distances = _compute_soft_norm(
            electron_electron_displacements, softening_term=softening_term
        )
        return jnp.sum(
            jnp.triu(strength / electron_electron_distances, k=1), axis=(-1, -2)
        )

    return potential_fn


def create_ion_ion_coulomb_potential(
    ion_locations: Array, ion_charges: Array
) -> ModelApply[ModelParams]:
    """Computes the total coulomb potential repulsion between stationary ions.

    Args:
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron)

    Returns:
        Callable: function which computes the potential energy due to the attraction
        between electrons and ion. Has the signature
        (params, electron_positions of shape (..., n_elec, d))
        -> array of potential energies of shape electron_positions.shape[:-2]
    """
    ion_ion_displacements, charge_charge_prods = _get_ion_ion_info(
        ion_locations, ion_charges
    )
    ion_ion_distances = _compute_soft_norm(ion_ion_displacements)
    constant_potential = jnp.sum(
        jnp.triu(charge_charge_prods / ion_ion_distances, k=1), axis=(-1, -2)
    )

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params, x
        return constant_potential

    return potential_fn



def create_hubbard_potential(
    N_up:int,
    strength: jnp.float32 = 1.0
) -> ModelApply[ModelParams]:
    """Computes the total Hubbard repulsion between pairs of electrons.

    Args:
        N_up (int): number of spin-up electrons
        strength (jnp.float32, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.

    Returns:
        Callable: function which computes the potential energy due to the repulsion
        between pairs of electrons. Has the signature
        (params, electron_positions of shape (..., n_elec))
        -> array of potential energies of shape electron_positions.shape[:-1]
    """

    is_zero=lambda x:jnp.heaviside(x,1)

    def _compute_1d_displacements(x: Array, y: Array) -> Array:
        return jnp.expand_dims(x, axis=-1) - jnp.expand_dims(y, axis=-2)

    def potential_fn(params: ModelParams, x: Array) -> Array:
        del params
        N = x.shape[-1]
        up_confs=jnp.take(x,indices=jnp.arange(0,N_up),axis=-1)
        down_confs=jnp.take(x,indices=jnp.arange(N_up,N),axis=-1)
        electron_electron_displacements = _compute_1d_displacements( up_confs, down_confs )
        energy_contributions=is_zero(electron_electron_displacements)

        return strength*jnp.sum( energy_contributions, axis=(-1, -2) )

    return potential_fn
