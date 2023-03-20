"""Exactly solvable single-electron hydrogen-like atom."""
from typing import Callable

import chex
import flax
import jax.numpy as jnp

import vmcnet.models as models
import vmcnet.physics as physics
from vmcnet.utils.typing import Array, P, ModelApply


class HydrogenLikeWavefunction(models.core.Module):
    """Model which computes -decay_rate * r, with trainable decay_rate.

    This model returns log|psi(x)|, where psi is the 1-s orbital exp(-decay_rate * r).
    This psi is an eigenfunction of the hydrogen-like Hamiltonian (one ion and one
    electron) when the decay rate is equal to 2 * nuclear_charge / (d - 1), with d > 1,
    where d is the dimension of the system.

    Thus when the system is 3-d, then this model computes the ground-state wavefunction
    precisely when decay_rate = nuclear_charge.

    Attributes:
        decay_rate (chex.Scalar): initial decay rate in the model
    """

    init_decay_rate: chex.Scalar

    @flax.linen.compact
    def __call__(self, x: Array) -> Array:  # type: ignore[override]
        """Log of isotropic exponential decay. Computes -decay_rate * ||x||.

        Args:
            x (Array): single electron positions, of shape (..., 1, d), where d
                is the dimension of the system

        Returns:
            Array: log of exponential decay wavefunction, with shape x.shape[:-2]
        """
        r = jnp.linalg.norm(x, axis=-1)
        scaled_r = models.core.Dense(
            1,
            kernel_init=lambda key, shape, **kwargs: jnp.array(
                [[self.init_decay_rate]]
            ),
            use_bias=False,
        )(r)
        return -jnp.squeeze(scaled_r, axis=-1)


def make_hydrogen_like_local_energy(
    log_psi_apply: Callable[[P, Array], Array],
    charge: chex.Scalar,
    d: int = 3,
) -> ModelApply[P]:
    """Local energy calculation for the hydrogen-like atom in general dimension d.

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|
        charge (chex.Scalar): charge of the nucleus
        d (int, optional): Dimension of the system (number of coordinates that each
            ion and electron has). Defaults to 3.

    Returns:
        Callable: local energy function which computes -0.5 nabla^2 psi / psi + (Z / r),
        where psi is the wavefn, and Z is the charge of the nucleus. Has the signature
        (params, x) -> local energy array of shape (x.shape[0],)
    """
    ion_location = jnp.zeros((1, d))
    ion_charge = jnp.array([charge])

    kinetic_fn = physics.kinetic.create_continuous_kinetic_energy(log_psi_apply)
    potential_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_location, ion_charge
    )

    return physics.core.combine_local_energy_terms([kinetic_fn, potential_fn])
