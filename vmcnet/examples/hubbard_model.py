from typing import Callable, Union

import flax
import jax.numpy as jnp

import vmcnet.physics as physics
import vmcnet.models as models
from vmcnet.utils.typing import P, ModelApply


class HubbardAnsatz(flax.linen.Module):

    N:int
    N_up:int
    side_length:int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return models.core.SimpleSlater(self.N,self.N_up,ndense_inner=self.N,side_length=self.side_length)(x)


def make_hubbard_local_energy(
    psi_apply: Callable[[P, jnp.ndarray], Union[jnp.float32, jnp.ndarray]],
    side_length: int,
) -> ModelApply[P]:

    kinetic_fn = physics.kinetic.create_hubbard_kinetic_energy(psi_apply,side_length)
    potential_fn = physics.potential.create_hubbard_potential(2)

    return physics.core.combine_local_energy_terms([kinetic_fn, potential_fn])
