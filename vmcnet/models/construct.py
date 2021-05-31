"""Combine pieces to form full models."""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax.numpy as jnp

from vmcnet.models.antisymmetry import logdet_product
from vmcnet.models.equivariance import FermiNetBackflow, FermiNetOrbitalLayer
from vmcnet.models.weights import WeightInitializer

Activation = Callable[[jnp.ndarray], jnp.ndarray]


class ComposedModel(flax.linen.Module):
    """A model made from composable parts.

    Attributes:
        submodels (Sequence[Union[Callable, flax.linen.Module]]): a sequence of
            functions or flax.linen.Modules which can be composed sequentially
    """

    submodels: Sequence[Union[Callable, flax.linen.Module]]

    @flax.linen.compact
    def __call__(self, x):
        outputs = x
        for model in self.submodels:
            outputs = model(outputs)
        return outputs


class SingleDeterminantFermiNet(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    ndense_list: List[Tuple[int, int]]
    kernel_initializer_unmixed: WeightInitializer
    kernel_initializer_mixed: WeightInitializer
    kernel_initializer_2e_1e_stream: WeightInitializer
    kernel_initializer_2e_2e_stream: WeightInitializer
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_1e_stream: WeightInitializer
    bias_initializer_2e_stream: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    activation_fn: Activation
    ion_pos: Optional[jnp.ndarray] = None
    include_2e_stream: bool = True
    include_ei_norm: bool = True
    include_ee_norm: bool = True
    streams_use_bias: bool = True
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False
    skip_connection: bool = True
    cyclic_spins: bool = True

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        nelec_total = elec_pos.shape[-2]
        if isinstance(self.spin_split, Sequence):
            nall_but_last = functools.reduce(lambda a, b: a + b, self.spin_split)
            norbitals = tuple(self.spin_split) + (nelec_total - nall_but_last,)
        else:
            norbitals = (nelec_total // self.spin_split,) * self.spin_split
        stream_1e, r_ei = FermiNetBackflow(
            spin_split=self.spin_split,
            ndense_list=self.ndense_list,
            kernel_initializer_unmixed=self.kernel_initializer_unmixed,
            kernel_initializer_mixed=self.kernel_initializer_mixed,
            kernel_initializer_2e_1e_stream=self.kernel_initializer_2e_1e_stream,
            kernel_initializer_2e_2e_stream=self.kernel_initializer_2e_2e_stream,
            bias_initializer_1e_stream=self.bias_initializer_1e_stream,
            bias_initializer_2e_stream=self.bias_initializer_2e_stream,
            activation_fn=self.activation_fn,
            ion_pos=self.ion_pos,
            include_2e_stream=self.include_2e_stream,
            include_ei_norm=self.include_ei_norm,
            include_ee_norm=self.include_ee_norm,
            use_bias=self.streams_use_bias,
            skip_connection=self.skip_connection,
            cyclic_spins=self.cyclic_spins,
        )(elec_pos)
        orbitals = FermiNetOrbitalLayer(
            spin_split=self.spin_split,
            norbitals=norbitals,
            kernel_initializer_linear=self.kernel_initializer_orbital_linear,
            kernel_initializer_envelope_dim=self.kernel_initializer_envelope_dim,
            kernel_initializer_envelope_ion=self.kernel_initializer_envelope_ion,
            bias_initializer_linear=self.bias_initializer_orbital_linear,
            use_bias=self.orbitals_use_bias,
            isotropic_decay=self.isotropic_decay,
        )(stream_1e, r_ei)
        return logdet_product(orbitals)
