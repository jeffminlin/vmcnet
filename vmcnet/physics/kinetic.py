"""Kinetic energy terms."""
from typing import Callable

import chex
import jax
import jax.numpy as jnp

import vmcnet.physics as physics
from vmcnet.utils.typing import Array, ArrayLike, P, ModelApply


def create_laplacian_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array]
) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> -0.5 (nabla^2 psi(x) / psi(x)).

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on only a single configuration so must be externally vmapped
        to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        return -0.5 * physics.core.laplacian_psi_over_psi(grad_log_psi_apply, params, x)

    return kinetic_energy_fn


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


def _compute_displacements(x: ArrayLike, y: ArrayLike) -> Array:
    """Compute the pairwise displacements between x and y in the second-to-last dim.

    Args:
        x (Array): array of shape (..., n_x, d)
        y (Array): array of shape (..., n_y, d)

    Returns:
        Array: pairwise displacements (x_i - y_j), with shape (..., n_x, n_y, d)
    """
    return jnp.expand_dims(x, axis=-2) - jnp.expand_dims(y, axis=-3)


def _compute_soft_norm(
    displacements: ArrayLike, softening_term: chex.Scalar = 0.0
) -> Array:
    """Compute an (optionally softened) norm, sqrt((sum_i x_i^2) + softening_term^2).

    Args:
        displacements (Array): array of shape (..., d)
        softening_term (chex.Scalar, optional): this amount squared is added to
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


def create_ibp_energy(
    log_psi_apply: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
) -> ModelApply[P]:
    def U(x: ArrayLike):
        electron_ion_displacements = _compute_displacements(x, ion_locations)
        electron_ion_distances = _compute_soft_norm(electron_ion_displacements)
        ei_coulomb_cone = ion_charges * electron_ion_distances / 2
        ei_contribution = jnp.sum(ei_coulomb_cone, axis=(-1, -2))

        electron_electron_displacements = _compute_displacements(x, x)
        electron_electron_distances = _compute_soft_norm(
            electron_electron_displacements
        )
        ee_coulomb_cone = electron_electron_distances / 4
        ee_contribution = jnp.sum(jnp.triu(ee_coulomb_cone, k=1), axis=(-1, -2))

        return ei_contribution + ee_contribution

    gradU_apply = jax.grad(U)
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def local_energy_fn(params: P, x: Array) -> Array:
        grad_log_psi = grad_log_psi_apply(params, x)
        gradU = gradU_apply(x)
        return 0.5 * jnp.linalg.norm(grad_log_psi) ** 2 - jnp.dot(
            grad_log_psi_apply, gradU
        )

    return local_energy_fn
