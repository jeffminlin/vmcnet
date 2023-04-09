"""Kinetic energy terms."""
from functools import partial
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


@jax.custom_vjp
def _differentiable_norm(displacements: ArrayLike) -> Array:
    """Compute an (optionally softened) norm, sqrt((sum_i x_i^2) + softening_term^2).

    Args:
        displacements (Array): array of shape (..., d)

    Returns:
        Array: array with shape displacements.shape[:-1]
    """
    return jnp.sqrt(jnp.sum(jnp.square(displacements), axis=-1))


def f_fwd(displacements):
    norm = _differentiable_norm(displacements)
    return (
        norm,
        (
            norm,
            displacements,
        ),
    )


def f_bwd(res, g):
    (norm, displacements) = res
    norm = jnp.expand_dims(norm, axis=-1)
    return (jnp.expand_dims(g, -1) * jnp.where(norm == 0, 0.0, displacements / norm),)


_differentiable_norm.defvjp(f_fwd, f_bwd)


def create_ibp_energy(
    log_psi_apply: Callable[[P, Array], Array],
    ion_locations: Array,
    ion_charges: Array,
) -> ModelApply[P]:
    def U(x: ArrayLike):
        electron_ion_displacements = _compute_displacements(x, ion_locations)
        electron_ion_distances = jnp.linalg.norm(electron_ion_displacements, axis=-1)
        ei_coulomb_cone = ion_charges * electron_ion_distances / 2
        print(f"ei cone: {ei_coulomb_cone.shape}")
        ei_contribution = -jnp.sum(ei_coulomb_cone, axis=(-1, -2))
        print(f"ei contribution: {ei_contribution.shape}")

        electron_electron_displacements = _compute_displacements(x, x)
        electron_electron_distances = _differentiable_norm(
            electron_electron_displacements
        )
        ee_coulomb_cone = electron_electron_distances / 4
        print(f"ee cone: {ee_coulomb_cone.shape}")
        ee_coulomb_cone = jnp.triu(ee_coulomb_cone, k=1)
        print(f"ee cone: {ee_coulomb_cone.shape}")
        ee_contribution = jnp.sum(
            ee_coulomb_cone,
            axis=(-1, -2),
        )
        print(f"ee contribution: {ee_contribution.shape}")

        return ei_contribution + ee_contribution

    gradU_apply = jax.grad(U)
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def local_energy_fn(params: P, x: Array) -> Array:
        grad_log_psi = grad_log_psi_apply(params, x)
        gradU = gradU_apply(x)
        print(f"Grad U: {gradU.shape}")
        print(f"Grad log psi: {grad_log_psi.shape}")
        return 0.5 * jnp.sum(grad_log_psi**2) - 2 * jnp.sum(grad_log_psi * gradU)

    return local_energy_fn
