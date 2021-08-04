"""Combine pieces to form full models."""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from vmcnet.models import antiequivariance
from vmcnet.utils.slog_helpers import slog_sum_over_axis
from vmcnet.utils.typing import ArrayList, Backflow, Jastrow, SpinSplit
from .antisymmetry import (
    ComposedBruteForceAntisymmetrize,
    SplitBruteForceAntisymmetrize,
    slogdet_product,
)
from .core import Activation, SimpleResNet, get_nelec_per_spin
from .equivariance import (
    FermiNetBackflow,
    FermiNetOneElectronLayer,
    FermiNetOrbitalLayer,
    FermiNetResidualBlock,
    FermiNetTwoElectronLayer,
)
from .invariance import InvariantTensor
from .jastrow import get_mol_decay_scaled_for_chargeless_molecules
from .sign_symmetry import (
    make_array_list_fn_sign_covariant,
    make_array_list_fn_sign_invariant,
)
from .weights import (
    WeightInitializer,
    get_bias_init_from_config,
    get_kernel_init_from_config,
)


# TODO: figure out a way to add other options in a scalable way
def _get_named_activation_fn(name):
    if name == "tanh":
        return jnp.tanh
    elif name == "gelu":
        return jax.nn.gelu
    else:
        raise ValueError("Activations besides tanh and gelu are not yet supported.")


def _get_dtype_init_constructors(dtype):
    kernel_init_constructor = functools.partial(
        get_kernel_init_from_config, dtype=dtype
    )
    bias_init_constructor = functools.partial(get_bias_init_from_config, dtype=dtype)
    return kernel_init_constructor, bias_init_constructor


def get_model_from_config(
    model_config: ConfigDict,
    nelec: jnp.ndarray,
    ion_pos: jnp.ndarray,
    ion_charges: jnp.ndarray,
    dtype=jnp.float32,
) -> flax.linen.Module:
    """Get a model from a hyperparameter config."""
    spin_split = tuple(jnp.cumsum(nelec)[:-1])

    backflow = get_backflow_from_config(
        model_config.backflow,
        ion_pos,
        spin_split,
        dtype=dtype,
    )

    kernel_init_constructor, bias_init_constructor = _get_dtype_init_constructors(dtype)

    if model_config.type in ["ferminet", "embedded_particle_ferminet"]:
        determinant_fn_builder = None
        if model_config.use_det_resnet:
            resnet_config = model_config.det_resnet
            determinant_fn_builder = get_resnet_determinant_fn_builder_for_ferminet(
                resnet_config.ndense,
                resnet_config.nlayers,
                _get_named_activation_fn(resnet_config.activation),
                kernel_init_constructor(resnet_config.kernel_init),
                bias_init_constructor(resnet_config.bias_init),
                resnet_config.use_bias,
                resnet_config.register_kfac,
            )

        if model_config.type == "ferminet":
            return FermiNet(
                spin_split,
                backflow,
                model_config.ndeterminants,
                kernel_initializer_orbital_linear=kernel_init_constructor(
                    model_config.kernel_init_orbital_linear
                ),
                kernel_initializer_envelope_dim=kernel_init_constructor(
                    model_config.kernel_init_envelope_dim
                ),
                kernel_initializer_envelope_ion=kernel_init_constructor(
                    model_config.kernel_init_envelope_ion
                ),
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
                determinant_fn_builder=determinant_fn_builder,
                make_odd_fn_from_even=model_config.make_odd_fn_from_even,
            )
        elif model_config.type == "embedded_particle_ferminet":
            invariance_config = model_config.invariance
            invariance_backflow = get_backflow_from_config(
                invariance_config.backflow,
                ion_pos,
                spin_split,
                dtype=dtype,
            )
            return EmbeddedParticleFermiNet(
                spin_split,
                model_config.nhidden_fermions_per_spin,
                backflow,
                model_config.ndeterminants,
                kernel_initializer_orbital_linear=kernel_init_constructor(
                    model_config.kernel_init_orbital_linear
                ),
                kernel_initializer_envelope_dim=kernel_init_constructor(
                    model_config.kernel_init_envelope_dim
                ),
                kernel_initializer_envelope_ion=kernel_init_constructor(
                    model_config.kernel_init_envelope_ion
                ),
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                invariance_backflow=invariance_backflow,
                invariance_kernel_initializer=kernel_init_constructor(
                    invariance_config.kernel_initializer
                ),
                invariance_bias_initializer=bias_init_constructor(
                    invariance_config.bias_initializer
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
                invariance_use_bias=invariance_config.use_bias,
                invariance_register_kfac=invariance_config.register_kfac,
                determinant_fn_builder=determinant_fn_builder,
                make_odd_fn_from_even=model_config.make_odd_fn_from_even,
            )

    elif model_config.type in ["orbital_cofactor_net", "per_particle_dets_net"]:
        if model_config.type == "orbital_cofactor_net":
            antieq_layer = antiequivariance.OrbitalCofactorAntiequivarianceLayer(
                spin_split,
                kernel_initializer_orbital_linear=kernel_init_constructor(
                    model_config.kernel_init_orbital_linear
                ),
                kernel_initializer_envelope_dim=kernel_init_constructor(
                    model_config.kernel_init_envelope_dim
                ),
                kernel_initializer_envelope_ion=kernel_init_constructor(
                    model_config.kernel_init_envelope_ion
                ),
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
            )
        elif model_config.type == "per_particle_dets_net":
            antieq_layer = antiequivariance.PerParticleDeterminantAntiequivarianceLayer(
                spin_split,
                kernel_initializer_orbital_linear=kernel_init_constructor(
                    model_config.kernel_init_orbital_linear
                ),
                kernel_initializer_envelope_dim=kernel_init_constructor(
                    model_config.kernel_init_envelope_dim
                ),
                kernel_initializer_envelope_ion=kernel_init_constructor(
                    model_config.kernel_init_envelope_ion
                ),
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
            )

        def array_list_equivariance(x: ArrayList) -> jnp.ndarray:
            concat_x = jnp.concatenate(x, axis=-2)
            return get_backflow_from_config(
                model_config.invariance,
                ion_pos=None,
                spin_split=spin_split,
                dtype=dtype,
            )(concat_x)[0]

        return AntiequivarianceNet(backflow, antieq_layer, array_list_equivariance)
    elif model_config.type == "brute_force_antisym":
        # TODO(Jeffmin): make interface more flexible w.r.t. different types of Jastrows
        jastrow = get_mol_decay_scaled_for_chargeless_molecules(ion_pos, ion_charges)
        if model_config.antisym_type == "rank_one":
            return SplitBruteForceAntisymmetryWithDecay(
                spin_split,
                backflow,
                jastrow,
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=kernel_init_constructor(
                    model_config.kernel_init_resnet
                ),
                kernel_initializer_jastrow=kernel_init_constructor(
                    model_config.kernel_init_jastrow
                ),
                bias_initializer_resnet=bias_init_constructor(
                    model_config.bias_init_resnet
                ),
                activation_fn_resnet=_get_named_activation_fn(
                    model_config.activation_fn_resnet
                ),
                resnet_use_bias=model_config.resnet_use_bias,
            )
        elif model_config.antisym_type == "double":
            return ComposedBruteForceAntisymmetryWithDecay(
                spin_split,
                backflow,
                jastrow,
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=kernel_init_constructor(
                    model_config.kernel_init_resnet
                ),
                kernel_initializer_jastrow=kernel_init_constructor(
                    model_config.kernel_init_jastrow
                ),
                bias_initializer_resnet=bias_init_constructor(
                    model_config.bias_init_resnet
                ),
                activation_fn_resnet=_get_named_activation_fn(
                    model_config.activation_fn_resnet
                ),
                resnet_use_bias=model_config.resnet_use_bias,
            )
        else:
            raise ValueError(
                "Unsupported brute-force antisymmetry type; {} was requested".format(
                    model_config.antisym_type
                )
            )
    else:
        raise ValueError(
            "Unsupported model type; {} was requested".format(model_config.type)
        )


def get_backflow_from_config(
    backflow_config,
    ion_pos,
    spin_split,
    dtype=jnp.float32,
) -> flax.linen.Module:
    """Get a FermiNet backflow from a model configuration."""
    kernel_init_constructor, bias_init_constructor = _get_dtype_init_constructors(dtype)

    residual_blocks = get_residual_blocks_for_ferminet_backflow(
        spin_split,
        backflow_config.ndense_list,
        kernel_initializer_unmixed=kernel_init_constructor(
            backflow_config.kernel_init_unmixed
        ),
        kernel_initializer_mixed=kernel_init_constructor(
            backflow_config.kernel_init_mixed
        ),
        kernel_initializer_2e_1e_stream=kernel_init_constructor(
            backflow_config.kernel_init_2e_1e_stream
        ),
        kernel_initializer_2e_2e_stream=kernel_init_constructor(
            backflow_config.kernel_init_2e_2e_stream
        ),
        bias_initializer_1e_stream=bias_init_constructor(
            backflow_config.bias_init_1e_stream
        ),
        bias_initializer_2e_stream=bias_init_constructor(
            backflow_config.bias_init_2e_stream
        ),
        activation_fn=_get_named_activation_fn(backflow_config.activation_fn),
        use_bias=backflow_config.use_bias,
        skip_connection=backflow_config.skip_connection,
        cyclic_spins=backflow_config.cyclic_spins,
    )

    backflow = FermiNetBackflow(
        residual_blocks,
        ion_pos=ion_pos,
        include_2e_stream=backflow_config.include_2e_stream,
        include_ei_norm=backflow_config.include_ei_norm,
        include_ee_norm=backflow_config.include_ee_norm,
    )

    return backflow


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


def get_residual_blocks_for_ferminet_backflow(
    spin_split: SpinSplit,
    ndense_list: List[Tuple[int, ...]],
    kernel_initializer_unmixed: WeightInitializer,
    kernel_initializer_mixed: WeightInitializer,
    kernel_initializer_2e_1e_stream: WeightInitializer,
    kernel_initializer_2e_2e_stream: WeightInitializer,
    bias_initializer_1e_stream: WeightInitializer,
    bias_initializer_2e_stream: WeightInitializer,
    activation_fn: Activation,
    use_bias: bool = True,
    skip_connection: bool = True,
    cyclic_spins: bool = True,
) -> List[FermiNetResidualBlock]:
    """Construct a list of FermiNet residual blocks composed by FermiNetBackflow.

    Arguments:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_list: (list of (int, ...)): number of dense nodes in each of the
            residual blocks, (ndense_1e, optional ndense_2e). The length of this list
            determines the number of residual blocks which are composed on top of the
            input streams. If ndense_2e is specified, then then a dense layer is applied
            to the two-electron stream with an optional skip connection, otherwise
            the two-electron stream is mixed into the one-electron stream but no
            transformation is done.
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
        bias_initializer_1e_stream (WeightInitializer): bias initializer for the
            one-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_2e_stream (WeightInitializer): bias initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function in the electron streams. Has
            the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether to add a bias term in the electron streams.
            Defaults to True.
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
    residual_blocks = []
    for ndense in ndense_list:
        one_electron_layer = FermiNetOneElectronLayer(
            spin_split,
            ndense[0],
            kernel_initializer_unmixed,
            kernel_initializer_mixed,
            kernel_initializer_2e_1e_stream,
            bias_initializer_1e_stream,
            activation_fn,
            use_bias,
            skip_connection,
            cyclic_spins,
        )
        two_electron_layer = None
        if len(ndense) > 1:
            two_electron_layer = FermiNetTwoElectronLayer(
                ndense[1],
                kernel_initializer_2e_2e_stream,
                bias_initializer_2e_stream,
                activation_fn,
                use_bias,
                skip_connection,
            )
        residual_blocks.append(
            FermiNetResidualBlock(one_electron_layer, two_electron_layer)
        )
    return residual_blocks


DeterminantFn = Callable[[ArrayList], jnp.ndarray]
DeterminantFnBuilder = Callable[[int], DeterminantFn]


def get_resnet_determinant_fn_builder_for_ferminet(
    ndense: int,
    nlayers: int,
    activation: Activation,
    kernel_initializer: WeightInitializer,
    bias_initializer: WeightInitializer,
    use_bias: bool = True,
    register_kfac: bool = False,
) -> DeterminantFnBuilder:
    """Get a resnet-based determinant function for FermiNet construction.

    This is used as a more general way to combine the determinant outputs into the
    final wavefunction value, relative to the original method of a sum of products.

    Note: the parameters for this network should be chosen so as to make it not too
    close to an odd function. If the Resnet were to be perfectly odd, for example by
    using a tanh activation function and no biases, then the wavefunction would end
    up being uniformly zero when there are an even number of spins, after the sign
    covariant symmetrization is applied. This somewhat paradoxical phenomenon comes from
    the fact that a per-spin odd function is actually even with respect to the inputs as
    a whole, since for two spins for example f(s1, s2) = -f(-s1, s2) = f(-s1, -s2).

    Args:
        ndense (int): the number of neurons in the dense layers
            of the ResNet.
        nlayers (int): the number of layers in the ResNet.
        activation (Activation): the activation function to use for the  resnet.
        kernel_initializer (WeightInitializer): kernel initializer for the resnet.
        bias_initializer (WeightInitializer): bias initializer for the resnet.
        use_bias(bool): Whether to use a bias in the ResNet. Defaults to True.
        register_kfac (bool): Whether to register the ResNet Dense layers with KFAC.
            Currently, params for this ResNet explode to huge values and cause nans
            if register_kfac is True, so this flag defaults to false and should only be
            overridden with care. The reason for the instability is not known.

    Returns:
        (Callable[[ArrayList, jnp.ndarray]): A resnet-based function which takes as
        input a list of nspins arrays of shape (..., ndeterminants) and outputs a
        single array of shape (..., 1).
    """

    def get_fn(nout: int) -> DeterminantFn:
        def fn(det_values: ArrayList) -> jnp.ndarray:
            concat_values = jnp.concatenate(det_values, axis=-1)
            return SimpleResNet(
                ndense,
                nout,
                nlayers,
                activation,
                kernel_initializer,
                bias_initializer,
                use_bias,
                register_kfac=register_kfac,
            )(concat_values)

        return fn

    return get_fn


class FermiNet(flax.linen.Module):
    """FermiNet/generalized Slater determinant model.

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            elec pos of shape (..., n, d)
                -> (
                    stream_1e of shape (..., n, d'),
                    r_ei of shape (..., n, nion, d),
                    r_ee of shape (..., n, n, d),
                )
        ndeterminants (int): number of determinants in the FermiNet model, i.e. the
            number of distinct orbital layers applied
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
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term in the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
        determinant_fn_builder (DeterminantFnBuilder, optional): An arbitrary
            function that can map an ArrayList of nspins determinant outputs of shape
            (..., ndeterminants) to a single output array of shape (..., 1). If
            provided, this function will be symmetrized to be sign-covariant (odd) with
            respect to each spin, and will then be used to combine the orbital
            determinants into the final wavefunction value. If not provided, the final
            wavefunction value is calculated as the sum of the product of the
            determinants, where the product is taken across the nspins spins and the sum
            is taken across the ndeterminants determinants. Defaults to None.
        make_odd_fn_from_even (bool, optional): whether to make determinant_fn odd (or
            sign-covariant) by starting with an even function and multiplying by odd
            linear terms. For example, for ndeterminants=1, and the up and down spin
            determinants are d1 and d2, they will be combined into d1*d2*g(d1, d2),
            where g(d1, d2) is an even function made by symmetrizing the provided
            determinant_fn. For more determinants, many such terms are combined.
            Defaults to False.
    """

    spin_split: SpinSplit
    backflow: Backflow
    ndeterminants: int
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False
    determinant_fn_builder: Optional[DeterminantFnBuilder] = None
    make_odd_fn_from_even: bool = False

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._sign_cov_det_fn = None
        self._sign_inv_det_fn = None
        if self.determinant_fn_builder is not None:
            if self.make_odd_fn_from_even:
                det_fn = self.determinant_fn_builder(self.ndeterminants ** 2)
                self._sign_inv_det_fn = make_array_list_fn_sign_invariant(det_fn)
            else:
                det_fn = self.determinant_fn_builder(1)
                self._sign_cov_det_fn = make_array_list_fn_sign_covariant(det_fn)

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        """Compose FermiNet backflow -> orbitals -> logabs determinant product.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: FermiNet output; logarithm of the absolute value of a
            anti-symmetric function of elec_pos, where the anti-symmetry is with respect
            to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        nelec_total = elec_pos.shape[-2]
        norbitals_per_spin = get_nelec_per_spin(self.spin_split, nelec_total)
        stream_1e, r_ei, _ = self._backflow(elec_pos)
        orbitals = [
            FermiNetOrbitalLayer(
                spin_split=self.spin_split,
                norbitals_per_spin=norbitals_per_spin,
                kernel_initializer_linear=self.kernel_initializer_orbital_linear,
                kernel_initializer_envelope_dim=self.kernel_initializer_envelope_dim,
                kernel_initializer_envelope_ion=self.kernel_initializer_envelope_ion,
                bias_initializer_linear=self.bias_initializer_orbital_linear,
                use_bias=self.orbitals_use_bias,
                isotropic_decay=self.isotropic_decay,
            )(stream_1e, r_ei)
            for _ in range(self.ndeterminants)
        ]
        # Orbitals shape is [nspins: (ndeterminants, ..., nelec[i], nelec[i])]
        orbitals = jax.tree_map(lambda *args: jnp.stack(args), *orbitals)

        if self._sign_cov_det_fn is not None:
            # dets is ArrayList of shape [nspins: (ndeterminants, ...)]
            dets = jax.tree_map(jnp.linalg.det, orbitals)
            # Swap axes to get shape [nspins: (..., ndeterminants)]
            fn_inputs = jax.tree_map(lambda x: jnp.swapaxes(x, 0, -1), dets)
            psi = jnp.squeeze(self._sign_cov_det_fn(fn_inputs), -1)
            return jnp.log(jnp.abs(psi))
        elif self._sign_inv_det_fn is not None:
            # dets is ArrayList of shape [nspins: (ndeterminants, ...)]
            dets = jax.tree_map(jnp.linalg.det, orbitals)
            # Swap axes to get shape [nspins: (..., ndeterminants)]
            fn_inputs = jax.tree_map(lambda x: jnp.swapaxes(x, 0, -1), dets)

            # Shape (..., d) where d hopefully = ndets^2
            even_outputs = self._sign_inv_det_fn(fn_inputs)

            up_dets = jnp.expand_dims(fn_inputs[0], -1)
            down_dets = jnp.expand_dims(fn_inputs[1], -2)
            prod_dets = up_dets * down_dets
            shaped_prod_dets = jnp.reshape(
                prod_dets,
                (
                    *prod_dets.shape[:-2],
                    prod_dets.shape[-1] * prod_dets.shape[-2],
                ),
            )
            psi = jnp.sum(shaped_prod_dets * even_outputs, axis=-1)
            return jnp.log(jnp.abs(psi))

        # slog_det_prods is SLArray of shape (ndeterminants, ...)
        slog_det_prods = slogdet_product(orbitals)
        _, log_psi = slog_sum_over_axis(slog_det_prods)
        return log_psi


class EmbeddedParticleFermiNet(flax.linen.Module):
    """Model that expands its inputs with extra hidden particles, then applies FermiNet.

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        nhidden_fermions_per_spin (Sequence[int]): number of hidden fermions to
            generate for each spin. Must have length nspins.
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            elec pos of shape (..., n, d)
                -> (
                    stream_1e of shape (..., n, d'),
                    r_ei of shape (..., n, nion, d),
                    r_ee of shape (..., n, n, d),
                )
        ndeterminants (int): number of determinants in the FermiNet model, i.e. the
            number of distinct orbital layers applied
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
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        invariance_backflow (Callable): backflow function to be used for the invariance
            which generates the hidden fermion positions.
        invariance_kernel_initializer (WeightInitializer): kernel initializer for the
            invariance dense layer. Has signature (key, shape, dtype) -> jnp.ndarray
        invariance_bias_initializer (WeightInitializer): bias initializer for the
            invariance dense layer. Has signature (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term in the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
        invariance_use_bias: (bool, optional): whether to add a bias term in the dense
            layer of the invariance. Defaults to True.
        invariance_register_kfac (bool, optional): whether to register the dense layer
            of the invariance with KFAC. Defaults to True.
        determinant_fn (Callable[[ArrayList], jnp.ndarray], optional): An arbitrary
            function that can map an ArrayList of nspins determinant outputs of shape
            (..., ndeterminants) to a single output array of shape (..., 1). If
            provided, this function will be symmetrized to be sign-covariant (odd) with
            respect to each spin, and will then be used to combine the orbital
            determinants into the final wavefunction value. If not provided, the final
            wavefunction value is calculated as the sum of the product of the
            determinants, where the product is taken across the nspins spins and the sum
            is taken across the ndeterminants determinants. Defaults to None.
    """

    spin_split: SpinSplit
    nhidden_fermions_per_spin: Sequence[int]
    backflow: Backflow
    ndeterminants: int
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    invariance_backflow: Backflow
    invariance_kernel_initializer: WeightInitializer
    invariance_bias_initializer: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False
    invariance_use_bias: bool = True
    invariance_register_kfac: bool = True
    determinant_fn: Optional[Callable[[ArrayList], jnp.ndarray]] = None

    def _get_invariant_tensor(
        self, output_shape_per_spin: Sequence[Tuple[int, int]]
    ) -> InvariantTensor:
        return InvariantTensor(
            self.spin_split,
            output_shape_per_spin,
            self.invariance_backflow,
            self.invariance_kernel_initializer,
            self.invariance_bias_initializer,
            self.invariance_use_bias,
            self.invariance_register_kfac,
        )

    def _get_ferminet(self, total_spin_split: SpinSplit) -> FermiNet:
        return FermiNet(
            total_spin_split,
            self.backflow,
            self.ndeterminants,
            self.kernel_initializer_orbital_linear,
            self.kernel_initializer_envelope_dim,
            self.kernel_initializer_envelope_ion,
            self.bias_initializer_orbital_linear,
            self.orbitals_use_bias,
            self.isotropic_decay,
            self.determinant_fn,
        )

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        """Use invariance to generate hidden particles, then Ferminet to calculate Psi.

        In this model, the set of input particles is expanded for each spin with a set
        of permutation invariant "hidden" fermions. These extra hidden fermions
        effectively move the problem into a higher dimensional space, resulting in
        larger orbital matrices and potentially a more expressive ansatz than a regular
        FermiNet. Due to the invariance of these hidden fermions with respect
        to the original input particle permutation, the ansatz is still antisymmetric
        with respect to the original input particles only.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: FermiNet output; logarithm of the absolute value of a
            anti-symmetric function of elec_pos, where the anti-symmetry is with respect
            to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        visible_nelec_total = elec_pos.shape[-2]
        d = elec_pos.shape[-1]
        visible_nelec_per_spin = get_nelec_per_spin(
            self.spin_split, visible_nelec_total
        )
        nspins = len(visible_nelec_per_spin)
        if len(self.nhidden_fermions_per_spin) != nspins:
            raise ValueError(
                "Length of nhidden_fermions_per_spin does not match number of spins. "
                "Provided {} for {} spins.".format(
                    len(self.nhidden_fermions_per_spin), nspins
                )
            )

        invariance_output_shape_per_spin = [
            (n, d) for n in self.nhidden_fermions_per_spin
        ]
        invariance = self._get_invariant_tensor(invariance_output_shape_per_spin)

        total_nelec_per_spin = [
            n + self.nhidden_fermions_per_spin[i]
            for i, n in enumerate(visible_nelec_per_spin)
        ]
        # Using numpy not jnp here to avoid Jax thinking this is a dynamic value and
        # complaining when it gets used within the constructed FermiNet.
        total_spin_split = tuple(np.cumsum(np.array(total_nelec_per_spin))[:-1])
        ferminet = self._get_ferminet(total_spin_split)

        split_input_particles = jnp.split(elec_pos, self.spin_split, axis=-2)
        split_hidden_particles = invariance(elec_pos)
        # Create list of [visible_pos_spin1, hidden_pos_spin1, visible_pos_spin2, ...],
        # so that the full input positions can be created with a single concatenation.
        split_all_particles = [
            p
            for i in range(nspins)
            for p in [split_input_particles[i], split_hidden_particles[i]]
        ]
        concat_all_particles = jnp.concatenate(split_all_particles, axis=-2)

        return ferminet(concat_all_particles)


class AntiequivarianceNet(flax.linen.Module):
    """Antisymmetry from anti-equivariance, backflow -> antieq -> odd invariance.

    Attributes:
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            elec pos of shape (..., n, d)
                -> (
                    stream_1e of shape (..., n, d'),
                    r_ei of shape (..., n, nion, d),
                    r_ee of shape (..., n, n, d),
                )
        antiequivariant_layer (Callable): function which computes antiequivariances-per-
            spin. Has the signature
            (stream_1e of shape (..., n, d_backflow), r_ei of shape (..., n, nion, d))
                -> (antieqs of shapes [spin: (..., n[spin], d_antieq)])
        array_list_equivariance (Callable): function which is equivariant-per-spin. Has
            the signature
            (list of arrays of shapes [spin: (..., n[spin], d_antieq)])
                -> (array of shape (..., n, d_equiv))
            All outputs of this function are made covariant with respect to each input
            sign, and an odd invariance is created by summing over the last two
            dimensions of the output.
    """

    backflow: Backflow
    antiequivariant_layer: Callable[[jnp.ndarray, jnp.ndarray], ArrayList]
    array_list_equivariance: Callable[[ArrayList], jnp.ndarray]

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._antiequivariant_layer = self.antiequivariant_layer
        self._sign_cov_equivariance = make_array_list_fn_sign_covariant(
            self.array_list_equivariance, axis=-3
        )

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        """Compose backflow -> antiequivariance -> sign covariant equivariance -> sum.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: log(abs(psi)), where psi is a general odd invariance of an
            anti-equivariant backflow. If the inputs have shape (batch_dims, nelec, d),
            then the output has shape (batch_dims,).
        """
        backflow_out, r_ei, _ = self._backflow(elec_pos)
        antiequivariant_out = self._antiequivariant_layer(backflow_out, r_ei)
        transformed_out = self._sign_cov_equivariance(antiequivariant_out)
        return jnp.log(jnp.abs(jnp.sum(transformed_out, axis=(-1, -2))))


class SplitBruteForceAntisymmetryWithDecay(flax.linen.Module):
    """Model with FermiNet backflow and a product of brute-force antisym. ResNets.

    A simple isotropic exponential decay is added to ensure square-integrability (and
    because this is asymptotically correct for molecules).

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            elec pos of shape (..., n, d)
                -> (
                    stream_1e of shape (..., n, d'),
                    r_ei of shape (..., n, nion, d),
                    r_ee of shape (..., n, n, d),
                )
        jastrow (Callable): function which computes a Jastrow factor from displacements.
            Has the signature
            (
                r_ei of shape (batch_dims, n, nion, d),
                r_ee of shape (batch_dims, n, n, d),
            )
                -> log jastrow of shape (batch_dims,)
        ndense_resnet (int): number of dense nodes in each layer of each antisymmetrized
            ResNet
        nlayers_resnet (int): number of layers in each antisymmetrized ResNet
        kernel_initializer_resnet (WeightInitializer): kernel initializer for
            the dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_jastrow (WeightInitializer): kernel initializer for the
            decay rates in the exponential decay Jastrow factor. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNets. Has the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        resnet_use_bias (bool, optional): whether to add a bias term in the dense layers
            of the antisymmetrized ResNets. Defaults to True.
    """

    spin_split: SpinSplit
    backflow: Backflow
    jastrow: Jastrow
    ndense_resnet: int
    nlayers_resnet: int
    kernel_initializer_resnet: WeightInitializer
    kernel_initializer_jastrow: WeightInitializer
    bias_initializer_resnet: WeightInitializer
    activation_fn_resnet: Activation
    resnet_use_bias: bool = True

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._jastrow = self.jastrow

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        """Compose FermiNet backflow -> antisymmetrized ResNets -> logabs product.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: spinful antisymmetrized output; logarithm of the absolute value
            of a anti-symmetric function of elec_pos, where the anti-symmetry is with
            respect to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        stream_1e, r_ei, r_ee = self._backflow(elec_pos)
        split_spins = jnp.split(stream_1e, self.spin_split, axis=-2)
        antisymmetric_part = SplitBruteForceAntisymmetrize(
            [
                SimpleResNet(
                    self.ndense_resnet,
                    1,
                    self.nlayers_resnet,
                    self.activation_fn_resnet,
                    self.kernel_initializer_resnet,
                    self.bias_initializer_resnet,
                    use_bias=self.resnet_use_bias,
                )
                for _ in split_spins
            ]
        )(split_spins)
        jastrow_part = self._jastrow(r_ei, r_ee)

        return antisymmetric_part + jastrow_part


class ComposedBruteForceAntisymmetryWithDecay(flax.linen.Module):
    """Model with FermiNet backflow and a single antisymmetrized ResNet.

    The ResNet is antisymmetrized with respect to each spin split separately (i.e. the
    antisymmetrization operators for each spin are composed and applied).

    A simple isotropic exponential decay is added to ensure square-integrability (and
    because this is asymptotically correct for molecules).

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            elec pos of shape (..., n, d)
                -> (
                    stream_1e of shape (..., n, d'),
                    r_ei of shape (..., n, nion, d),
                    r_ee of shape (..., n, n, d),
                )
        jastrow (Callable): function which computes a Jastrow factor from displacements.
            Has the signature
            (
                r_ei of shape (batch_dims, n, nion, d),
                r_ee of shape (batch_dims, n, n, d),
            )
                -> log jastrow of shape (batch_dims,)
        ndense_resnet (int): number of dense nodes in each layer of the ResNet
        nlayers_resnet (int): number of layers in each antisymmetrized ResNet
        kernel_initializer_resnet (WeightInitializer): kernel initializer for
            the dense layers in the antisymmetrized ResNet. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_jastrow (WeightInitializer): kernel initializer for the
            decay rates in the exponential decay Jastrow factor. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNet. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNet. Has the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        resnet_use_bias (bool, optional): whether to add a bias term in the dense layers
            of the antisymmetrized ResNet. Defaults to True.
    """

    spin_split: SpinSplit
    backflow: Backflow
    jastrow: Jastrow
    ndense_resnet: int
    nlayers_resnet: int
    kernel_initializer_resnet: WeightInitializer
    kernel_initializer_jastrow: WeightInitializer
    bias_initializer_resnet: WeightInitializer
    activation_fn_resnet: Activation
    resnet_use_bias: bool = True

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._jastrow = self.jastrow

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> jnp.ndarray:
        """Compose FermiNet backflow -> antisymmetrized ResNet -> logabs.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: spinful antisymmetrized output; logarithm of the absolute value
            of a anti-symmetric function of elec_pos, where the anti-symmetry is with
            respect to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        stream_1e, r_ei, r_ee = self._backflow(elec_pos)
        split_spins = jnp.split(stream_1e, self.spin_split, axis=-2)
        antisymmetric_part = ComposedBruteForceAntisymmetrize(
            SimpleResNet(
                self.ndense_resnet,
                1,
                self.nlayers_resnet,
                self.activation_fn_resnet,
                self.kernel_initializer_resnet,
                self.bias_initializer_resnet,
                use_bias=self.resnet_use_bias,
            )
        )(split_spins)
        jastrow_part = self._jastrow(r_ei, r_ee)

        return antisymmetric_part + jastrow_part
