"""Energy terms related to the integration-by-parts approach."""

from typing import Callable, List

import chex
import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, ArrayLike, LocalEnergyApply, ModelApply, P

from .potential import (
    compute_displacements,
    compute_soft_norm,
    create_electron_electron_coulomb_potential,
    create_electron_ion_coulomb_potential,
    create_ion_ion_coulomb_potential,
)
from .kinetic import create_laplacian_kinetic_energy
from .core import combine_local_energy_terms


def create_gradient_squared_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
) -> ModelApply[P]:
    """Create the local kinetic energy function in the gradient squared form.

    This form is given by (params, x) -> 0.5 (nabla psi(x) / psi(x))^2, which comes
    from applying integration by parts to the Laplacian form.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), in gradient squared form
        i.e. 0.5 (nabla psi(x) / psi(x))^2. Evaluates on only a single configuration
        so must be externally vmapped to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        return 0.5 * jnp.sum(grad_log_psi_apply(params, x) ** 2)

    return kinetic_energy_fn


def create_ibp_energy_term(
    log_psi_apply: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
    include_kinetic: bool = True,
    include_ei: bool = True,
    ei_softening: chex.Scalar = 0.0,
    include_ee: bool = True,
    ee_softening: chex.Scalar = 0.0,
) -> ModelApply[P]:
    """Create the integrated-by-parts (IBP) portion of the local energy.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
           inputs x. It is okay for it to produce batch outputs on batches of x as long
           as it produces a single number for single x. Has the signature
           (params, single_x_in) -> log|psi(single_x_in)|
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron).
        include_kinetic (bool): whether to include the kinetic energy in the IBP energy.
            Defaults to True.
        include_ei (bool): whether to include the electron-ion Coulomb interaction in
            the IBP energy. Defaults to True.
        ei_softening (chex.Scalar): softening term to add to the electron-ion
            interaction to smooth out the singularity. Defaults to 0.0.
        include_ee (bool): whether to include the electron-electron Coulomb interaction
            in the IBP energy. Defaults to True.
        ee_softening (chex.Scalar): softening term to add to the electron-electron
            interaction to smooth out the singularity. Defaults to 0.0.

    Returns:
        Callable: function which computes the IBP part of the local energy, with the
            requested terms and parameters.
    """

    def U(x: ArrayLike):
        """Evaluate a potential whose Laplacian is the Coulomb potential."""
        result = jnp.array(0.0)

        if include_ei:
            electron_ion_displacements = _ = compute_displacements(x, ion_locations)
            electron_ion_distances = compute_soft_norm(
                electron_ion_displacements, ei_softening
            )
            ei_coulomb_cone = ion_charges * electron_ion_distances / 2
            ei_contribution = -jnp.sum(ei_coulomb_cone, axis=(-1, -2))
            result += ei_contribution

        if include_ee:
            electron_electron_displacements = compute_displacements(x, x)
            electron_electron_distances = compute_soft_norm(
                electron_electron_displacements, ee_softening
            )
            ee_coulomb_cone = electron_electron_distances / 4
            ee_contribution = jnp.sum(
                jnp.triu(ee_coulomb_cone, k=1),
                axis=(-1, -2),
            )
            result += ee_contribution

        return result

    gradU_apply = jax.grad(U)
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def local_energy_fn(params: P, x: Array) -> Array:
        grad_log_psi = grad_log_psi_apply(params, x)
        gradU = gradU_apply(x)

        result = -2 * jnp.sum(grad_log_psi * gradU)
        if include_kinetic:
            result += 0.5 * jnp.sum(grad_log_psi**2)

        return result

    return local_energy_fn


def create_ibp_local_energy(
    log_psi_apply: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
    ibp_kinetic: bool = True,
    ibp_ei: bool = True,
    ei_softening: chex.Scalar = 0.0,
    ibp_ee: bool = True,
    ee_softening: chex.Scalar = 0.0,
) -> LocalEnergyApply[P]:
    """Create the full local energy for the integration by parts (IBP) method.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
           inputs x. It is okay for it to produce batch outputs on batches of x as long
           as it produces a single number for single x. Has the signature
           (params, single_x_in) -> log|psi(single_x_in)|
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron).
        ibp_kinetic (bool): whether to apply IBP to the kinetic energy. Defaults to
            True.
        ibp_ei (bool): whether to apply IBP to the electron-ion Coulomb interaction.
            Defaults to True.
        ei_softening (chex.Scalar): softening term to add to the electron-ion
            interaction to smooth out the singularity. Defaults to 0.0.
        ibp_ee (bool): whether to apply IBP to the electron-electron Coulomb
            interaction. Defaults to True.
        ee_softening (chex.Scalar): softening term to add to the electron-electron
            interaction to smooth out the singularity. Defaults to 0.0.

    Returns:
        Callable: function which computes the full local energy for the IBP method,
            with the requested terms integrated-by-parts and the other terms included
            in their standard formulation. Evaluates on only a single configuration
            so must be externally vmapped to be applied to a batch of walkers.
    """
    ibp_energy = create_ibp_energy_term(
        log_psi_apply,
        ion_locations,
        ion_charges,
        ibp_kinetic,
        ibp_ei,
        ei_softening,
        ibp_ee,
        ee_softening,
    )

    ii_potential_fn = create_ion_ion_coulomb_potential(ion_locations, ion_charges)

    energy_terms: List[ModelApply[P]] = [ibp_energy, ii_potential_fn]

    if not ibp_kinetic:
        energy_terms.append(create_laplacian_kinetic_energy(log_psi_apply))
    if not ibp_ei:
        energy_terms.append(
            create_electron_ion_coulomb_potential(
                ion_locations, ion_charges, softening_term=ei_softening
            )
        )
    if not ibp_ee:
        energy_terms.append(
            create_electron_electron_coulomb_potential(softening_term=ee_softening)
        )

    local_energy_fn: LocalEnergyApply[P] = combine_local_energy_terms(energy_terms)

    return local_energy_fn
