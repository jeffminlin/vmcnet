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
        """Call submodels on the output of the previous one one at a time."""
        outputs = x
        for model in self.submodels:
            outputs = model(outputs)
        return outputs


class SingleDeterminantFermiNet(flax.linen.Module):
    """Single determinant FermiNet model. Multiple dets to be added.

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_list: (list of (int, int)): number of dense nodes in each of the
            residual blocks, (ndense_1e, ndense_2e). The length of this list determines
            the number of residual blocks which are composed on top of the input streams
        kernel_initializer_unmixed (WeightInitializer): kernel initializer for the
            unmixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the previous one-electron stream output. Has
            signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_1e_stream (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_2e_stream (WeightInitializer): kernel initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer for the
            decay rate in the exponential envelopes. If `isotropic_decay` is True, then
            this initializes a single decay rate number per ion and orbital. If
            `isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer for the
            linear combination over the ions of exponential envelopes. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_1e_stream (WeightInitializer): bias initializer for the
            one-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_2e_stream (WeightInitializer): bias initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function in the electron streams. Has
            the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        ion_pos (jnp.ndarray, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_2e_stream (bool, optional): whether to compute pairwise electron
            displacements/distances. Defaults to True.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.
        streams_use_bias (bool, optional): whether to add a bias term in the electron
            streams. Defaults to True.
        orbitals_use_bias (bool, optional): whether to add a bias term in the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and outputs of the streams match. Defaults
            to True.
        cyclic_spins (bool, optional): whether the the concatenation in the one-electron
            stream should satisfy a cyclic equivariance structure, i.e. if there are
            three spins (1, 2, 3), then in the mixed part of the stream, after averaging
            but before the linear transformation, cyclic equivariance means the inputs
            are [(1, 2, 3), (2, 3, 1), (3, 1, 2)]. If False, then the inputs are
            [(1, 2, 3), (1, 2, 3), (1, 2, 3)] (as in the original FermiNet).
            When there are only two spins (spin-1/2 case), then this is equivalent to
            true spin equivariance. Defaults to False (original FermiNet).
    """

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
        """Compose FermiNet backflow -> orbitals -> log determinant product."""
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
