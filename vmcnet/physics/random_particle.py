"""Energy terms related to the random particle approach."""
from typing import Callable, List

import chex
import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, LocalEnergyApply, ModelApply, P, PRNGKey

from .core import laplacian_psi_over_psi, per_particle_laplace
from .kinetic import create_laplacian_kinetic_energy
from .potential import (
    create_electron_electron_coulomb_potential,
    create_electron_electron_per_particle_potential,
    create_electron_ion_coulomb_potential,
    create_electron_ion_per_particle_potential,
    create_ion_ion_coulomb_potential,
)

RandomParticleKineticEnergy = Callable[[P, Array, Array], Array]


def create_per_particle_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array], surrogate_params
) -> RandomParticleKineticEnergy:
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(x: Array) -> Array:
        return -0.5 * per_particle_laplace(grad_log_psi_apply, surrogate_params, x)

    return kinetic_energy_fn


def create_random_particle_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array], nparticles: int = 1
) -> RandomParticleKineticEnergy:
    """Create a local kinetic energy which samples nparticles random particles.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|
        nparticles (int, Optional): the number of random particles to evaluate the
            kinetic energy of. Defaults to 1.

    Returns:
        Callable: function which computes a random particle estimator of the local
        kinetic energy, i.e. -0.5 nabla^2 psi / psi. Evaluates on only a single
        configuration so must be externally vmapped to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: Array, particle_perm: Array) -> Array:
        return -0.5 * laplacian_psi_over_psi(
            grad_log_psi_apply, params, x, nparticles, particle_perm
        )

    return kinetic_energy_fn


def assemble_random_particle_local_energy(
    kinetic_term,
    potential_terms: List[ModelApply[P]],
    sample_kinetic: bool,
    surrogate=None,
    nparticles: int = 1,
) -> LocalEnergyApply[P]:
    """Assembles the random particle local energy from kinetic and potential terms."""

    def local_energy_fn(
        wf_params: P, surrogate_params: P, positions: Array, key: PRNGKey
    ):
        if surrogate is None:
            total_particles = positions.shape[-2]
            perm = jax.random.permutation(key, jnp.arange(total_particles))
            permuted_positions = positions[perm, :]

            wf_energy = (
                kinetic_term(wf_params, positions, perm)
                if sample_kinetic
                else kinetic_term(wf_params, positions)
            )

            for potential_term in potential_terms:
                wf_energy += potential_term(wf_params, permuted_positions)

            return wf_energy

        else:
            total_particles = positions.shape[-2]
            surrogate_energies = surrogate(surrogate_params, positions)

            perm = jax.random.permutation(key, jnp.arange(total_particles))

            permuted_surrogate_energies = surrogate_energies[perm]
            surrogate_corrections = (
                jnp.mean(permuted_surrogate_energies, axis=-1, keepdims=True)
                - permuted_surrogate_energies
            )

            permuted_positions = positions[perm, :]
            total_correction = jnp.sum(surrogate_corrections[:nparticles])

            wf_energy = (
                kinetic_term(wf_params, positions, perm)
                if sample_kinetic
                else kinetic_term(wf_params, positions)
            )

            for potential_term in potential_terms:
                wf_energy += potential_term(wf_params, permuted_positions)

            sg_energy = jnp.sum(permuted_surrogate_energies[:nparticles])
            return wf_energy + total_correction, jnp.square(wf_energy - sg_energy)

    return local_energy_fn


def create_molecular_random_particle_local_energy(
    log_psi_apply: Callable[[P, Array], Array],
    surrogate: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
    nparticles: int = 1,
    sample_kinetic: bool = True,
    sample_ei: bool = True,
    ei_softening: chex.Scalar = 0.0,
    sample_ee: bool = True,
    ee_softening: chex.Scalar = 0.0,
) -> LocalEnergyApply[P]:
    """Create the full local energy for the random particle method for molecules.

    This method evaluates the local energy using a randomly selected subset of the
    particles for each walker, to obtain a cheaper but more noisy estimate of the
    local energy.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
           inputs x. It is okay for it to produce batch outputs on batches of x as long
           as it produces a single number for single x. Has the signature
           (params, single_x_in) -> log|psi(single_x_in)|
        surrogate (Callable): Surrogate.
        ion_locations (Array): an (n, d) array of ion positions, where n is the
            number of ion positions and d is the dimension of the space they live in
        ion_charges (Array): an (n,) array of ion charges, in units of one
            elementary charge (the charge of one electron).
        nparticles (int): the number of particles to select to estimate the local
            energy. Defaults to 1.
        sample_kinetic (bool): whether to sample the kinetic energy. Defaults to
            True.
        sample_ei (bool): whether to sample the electron-ion Coulomb interaction.
            Defaults to True.
        ei_softening (chex.Scalar): softening term to add to the electron-ion
            interaction to smooth out the singularity. Defaults to 0.0.
        sample_ee (bool): whether to sample the electron-electron Coulomb interaction.
            Defaults to True.
        ee_softening (chex.Scalar): softening term to add to the electron-electron
            interaction to smooth out the singularity. Defaults to 0.0.

    Returns:
        Callable: function which computes the full local energy for the IBP method,
            with the requested terms integrated-by-parts and the other terms included
            in their standard formulation.
    """
    ii_potential_fn = create_ion_ion_coulomb_potential(ion_locations, ion_charges)

    ei_potential_fn = create_electron_ion_coulomb_potential(
        ion_locations,
        ion_charges,
        softening_term=ei_softening,
        nparticles=nparticles if sample_ei else None,
    )

    ee_potential_fn = create_electron_electron_coulomb_potential(
        softening_term=ee_softening, nparticles=nparticles if sample_ee else None
    )

    kinetic_energy = (
        create_random_particle_kinetic_energy(log_psi_apply, nparticles)
        if sample_kinetic
        else create_laplacian_kinetic_energy(log_psi_apply)
    )

    return assemble_random_particle_local_energy(
        kinetic_energy,
        [ii_potential_fn, ei_potential_fn, ee_potential_fn],
        sample_kinetic,
        surrogate,
        nparticles,
    )


def assemble_random_particle_surrogate_energy(
    terms: List[ModelApply[P]],
) -> LocalEnergyApply[P]:
    """Assembles the random particle surrogate energy from kinetic and potential terms."""

    def local_energy_fn(positions: Array) -> Array:
        result = terms[0](positions)

        for term in terms[1:]:
            result += term(positions)

        return result

    return local_energy_fn


def create_molecular_random_particle_surrogate_energy(
    surrogate_log_psi_apply: Callable[[Array], Array],
    surrogate_params,
    ion_locations: Array,
    ion_charges: Array,
    ei_softening: chex.Scalar = 0.0,
    ee_softening: chex.Scalar = 0.0,
) -> LocalEnergyApply[P]:
    ei_potential_fn = create_electron_ion_per_particle_potential(
        ion_locations,
        ion_charges,
        softening_term=ei_softening,
    )

    ee_potential_fn = create_electron_electron_per_particle_potential(
        softening_term=ee_softening
    )

    kinetic_energy = create_per_particle_kinetic_energy(
        surrogate_log_psi_apply, surrogate_params
    )

    return assemble_random_particle_surrogate_energy(
        [ei_potential_fn, ee_potential_fn, kinetic_energy]
    )