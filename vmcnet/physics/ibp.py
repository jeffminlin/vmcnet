"""Energy terms related to the integration-by-parts approach."""
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, ArrayLike, P, ModelApply

from .potential import (
    compute_displacements,
    compute_soft_norm,
    create_ion_ion_coulomb_potential,
)
from .core import combine_local_energy_terms


def create_gradient_squared_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array]
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
    ei_softening: chex.Scalar = 0.0,
    ee_softening: chex.Scalar = 0.0,
) -> ModelApply[P]:
    def U(x: ArrayLike):
        electron_ion_displacements = _ = compute_displacements(x, ion_locations)
        electron_ion_distances = compute_soft_norm(
            electron_ion_displacements, ei_softening
        )
        ei_coulomb_cone = ion_charges * electron_ion_distances / 2
        ei_contribution = -jnp.sum(ei_coulomb_cone, axis=(-1, -2))

        electron_electron_displacements = compute_displacements(x, x)
        electron_electron_distances = compute_soft_norm(
            electron_electron_displacements, ee_softening
        )
        ee_coulomb_cone = electron_electron_distances / 4
        ee_contribution = jnp.sum(
            jnp.triu(ee_coulomb_cone, k=1),
            axis=(-1, -2),
        )

        return ei_contribution + ee_contribution

    gradU_apply = jax.grad(U)
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def local_energy_fn(params: P, x: Array) -> Array:
        grad_log_psi = grad_log_psi_apply(params, x)
        gradU = gradU_apply(x)
        return 0.5 * jnp.sum(grad_log_psi**2) - 2 * jnp.sum(grad_log_psi * gradU)

    return local_energy_fn


def create_ibp_local_energy(
    log_psi_apply: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
    ei_softening: chex.Scalar = 0.0,
    ee_softening: chex.Scalar = 0.0,
):
    ii_potential_fn = create_ion_ion_coulomb_potential(ion_locations, ion_charges)
    ibp_energy = create_ibp_energy_term(
        log_psi_apply, ion_locations, ion_charges, ei_softening, ee_softening
    )

    local_energy_fn: ModelApply[P] = combine_local_energy_terms(
        [ibp_energy, ii_potential_fn]
    )

    return local_energy_fn
