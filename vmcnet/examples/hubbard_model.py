from typing import Callable, Union

import flax
import jax.numpy as jnp

import vmcnet.physics as physics
import vmcnet.models as models
from vmcnet.utils.typing import P, ModelApply


class HubbardAnsatz(flax.linen.Module):
    """Ansatz given by a Slater determinant
    Spinful basis functions psi_i can be viewed as

                / phi_up_i(x), s=up
    psi_i(x,s)={
               \ phi_down_i(x), s=down

    We define two SimpleResNets phi_list_up/down where
    phi_list_up(x)=(phi_up_1(x),...,phi_up_{N_up}(x))
    phi_list_down(x)=(phi_down_{N_up+1}(x),...,phi_down_{N}(x))

    Then we represent the Slater determinant as

    |psi_1(x_1,up)....psi_1(x_{N_up},up),psi_1(x_{N_up+1},down)....psi_1(x_N,down)|
    |                                                                             |
    |                                                                             |
    |                                                                             |
    |psi_N(x_1,up)....psi_N(x_{N_up},up),psi_N(x_{N_up+1},down)....psi_N(x_N,down)|

    =

    |phi_list_up(x_1)....phi_list_up(x_{N_up}),phi_list_down(x_{N_up+1})....phi_list_down(x_N)|


    Attributes:
        N (int): number of electrons
        N_up (int): number of spin-up electrons
        side_length (int): side length of finite lattice with periodic boundary conditions
    """

    N:int
    N_up:int
    side_length:int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x (Array): electron configurations, of shape (..., N)

        Returns:
            Array: Value of wave function (not in log domain) modeled as Slater determinant, with shape x.shape[:-1]
        """
        return models.core.SimpleSlater(self.N,self.N_up,ndense_inner=self.N,side_length=self.side_length)(x)


def make_hubbard_local_energy(
    psi_apply: Callable[[P, jnp.ndarray], Union[jnp.float32, jnp.ndarray]],
    side_length: int,
) -> ModelApply[P]:

    kinetic_fn = physics.kinetic.create_hubbard_kinetic_energy(psi_apply,side_length)
    potential_fn = physics.potential.create_hubbard_potential(2)

    return physics.core.combine_local_energy_terms([kinetic_fn, potential_fn])
