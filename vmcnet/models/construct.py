"""Combine pieces to form full models."""
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from vmcnet.utils.slog_helpers import slog_sum_over_axis
from vmcnet.utils.typing import ArrayList, SpinSplit
from .antisymmetry import (
    ComposedBruteForceAntisymmetrize,
    SplitBruteForceAntisymmetrize,
    slogdet_product,
)
from .core import (
    Activation,
    SimpleResNet,
    get_nelec_per_spin,
)
from .equivariance import (
    FermiNetBackflow,
    FermiNetOneElectronLayer,
    FermiNetOrbitalLayer,
    FermiNetResidualBlock,
    FermiNetTwoElectronLayer,
)
from .jastrow import IsotropicAtomicExpDecay
from .sign_covariance import make_array_list_fn_sign_covariant
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


def get_model_from_config(
    model_config: ConfigDict,
    nelec: jnp.ndarray,
    ion_pos: jnp.ndarray,
    dtype=jnp.float32,
) -> flax.linen.Module:
    """Get a model from a hyperparameter config."""
    spin_split = tuple(jnp.cumsum(nelec)[:-1])

    backflow = get_backflow_from_model_config(
        model_config,
        ion_pos,
        spin_split,
        dtype=dtype,
    )

    if model_config.type == "ferminet":
        determinant_fn = None
        if model_config.use_det_resnet:
            resnet_config = model_config.det_resnet
            determinant_fn = get_resnet_determinant_fn_for_ferminet(
                resnet_config.ndense,
                resnet_config.nlayers,
                _get_named_activation_fn(resnet_config.activation),
                get_kernel_init_from_config(resnet_config.kernel_init, dtype=dtype),
                get_bias_init_from_config(resnet_config.bias_init, dtype=dtype),
                resnet_config.use_bias,
                resnet_config.register_kfac
            )
        return FermiNet(
            spin_split,
            backflow,
            model_config.ndeterminants,
            kernel_initializer_orbital_linear=get_kernel_init_from_config(
                model_config.kernel_init_orbital_linear, dtype=dtype
            ),
            kernel_initializer_envelope_dim=get_kernel_init_from_config(
                model_config.kernel_init_envelope_dim, dtype=dtype
            ),
            kernel_initializer_envelope_ion=get_kernel_init_from_config(
                model_config.kernel_init_envelope_ion, dtype=dtype
            ),
            bias_initializer_orbital_linear=get_bias_init_from_config(
                model_config.bias_init_orbital_linear, dtype=dtype
            ),
            orbitals_use_bias=model_config.orbitals_use_bias,
            isotropic_decay=model_config.isotropic_decay,
            determinant_fn=determinant_fn,
        )
    elif model_config.type == "brute_force_antisym":
        if model_config.antisym_type == "rank_one":
            return SplitBruteForceAntisymmetryWithDecay(
                spin_split,
                backflow,
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=get_kernel_init_from_config(
                    model_config.kernel_init_resnet, dtype=dtype
                ),
                kernel_initializer_jastrow=get_kernel_init_from_config(
                    model_config.kernel_init_jastrow, dtype=dtype
                ),
                bias_initializer_resnet=get_bias_init_from_config(
                    model_config.bias_init_resnet, dtype=dtype
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
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=get_kernel_init_from_config(
                    model_config.kernel_init_resnet, dtype=dtype
                ),
                kernel_initializer_jastrow=get_kernel_init_from_config(
                    model_config.kernel_init_jastrow, dtype=dtype
                ),
                bias_initializer_resnet=get_bias_init_from_config(
                    model_config.bias_init_resnet, dtype=dtype
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


def get_backflow_from_model_config(
    model_config,
    ion_pos,
    spin_split,
    dtype=jnp.float32,
):
    """Get a FermiNet backflow from a model configuration."""
    residual_blocks = _get_residual_blocks_for_ferminet_backflow(
        spin_split,
        model_config.backflow.ndense_list,
        kernel_initializer_unmixed=get_kernel_init_from_config(
            model_config.backflow.kernel_init_unmixed, dtype=dtype
        ),
        kernel_initializer_mixed=get_kernel_init_from_config(
            model_config.backflow.kernel_init_mixed, dtype=dtype
        ),
        kernel_initializer_2e_1e_stream=get_kernel_init_from_config(
            model_config.backflow.kernel_init_2e_1e_stream, dtype=dtype
        ),
        kernel_initializer_2e_2e_stream=get_kernel_init_from_config(
            model_config.backflow.kernel_init_2e_2e_stream, dtype=dtype
        ),
        bias_initializer_1e_stream=get_bias_init_from_config(
            model_config.backflow.bias_init_1e_stream, dtype=dtype
        ),
        bias_initializer_2e_stream=get_bias_init_from_config(
            model_config.backflow.bias_init_2e_stream, dtype=dtype
        ),
        activation_fn=_get_named_activation_fn(model_config.backflow.activation_fn),
        use_bias=model_config.backflow.use_bias,
        skip_connection=model_config.backflow.skip_connection,
        cyclic_spins=model_config.backflow.cyclic_spins,
    )

    backflow = FermiNetBackflow(
        residual_blocks,
        ion_pos=ion_pos,
        include_2e_stream=model_config.backflow.include_2e_stream,
        include_ei_norm=model_config.backflow.include_ei_norm,
        include_ee_norm=model_config.backflow.include_ee_norm,
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


def _get_residual_blocks_for_ferminet_backflow(
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


def get_resnet_determinant_fn_for_ferminet(
    ndense: int,
    nlayers: int,
    activation: Activation,
    kernel_initializer: WeightInitializer,
    bias_initializer: WeightInitializer,
    use_bias: bool = True,
    register_kfac: bool = False,
) -> Callable[[ArrayList], jnp.ndarray]:
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

    def fn(det_values: ArrayList) -> jnp.ndarray:
        concat_values = jnp.concatenate(det_values, axis=-1)
        return SimpleResNet(
            ndense,
            1,
            nlayers,
            activation,
            kernel_initializer,
            bias_initializer,
            use_bias,
            register_kfac=register_kfac,
        )(concat_values)

    return fn


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
            (elec pos of shape (..., n, d))
                -> (stream_1e of shape (..., n, d'), r_ei of shape (..., n, nion, d))
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

    spin_split: Union[int, Sequence[int]]
    backflow: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    ndeterminants: int
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False
    determinant_fn: Optional[Callable[[ArrayList], jnp.ndarray]] = None

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._determinant_fn = self.determinant_fn

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
        stream_1e, r_ei = self._backflow(elec_pos)
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

        if self._determinant_fn is not None:
            # dets is ArrayList of shape [nspins: (ndeterminants, ...)]
            dets = jax.tree_map(jnp.linalg.det, orbitals)
            # Swap axes to get shape [nspins: (..., ndeterminants)]
            fn_inputs = jax.tree_map(lambda x: jnp.swapaxes(x, 0, -1), dets)
            # Symmetrize the resnet to be sign covariant with respect to each spin.
            sign_cov_det_fn = make_array_list_fn_sign_covariant(self._determinant_fn)
            psi = jnp.squeeze(sign_cov_det_fn(fn_inputs), -1)
            return jnp.log(jnp.abs(psi))

        # slog_det_prods is SLArray of shape (ndeterminants, ...)
        slog_det_prods = slogdet_product(orbitals)
        _, log_psi = slog_sum_over_axis(slog_det_prods)
        return log_psi


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
            (elec pos of shape (..., n, d))
                -> (stream_1e of shape (..., n, d'), r_ei of shape (..., n, nion, d))
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
    backflow: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
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
        stream_1e, r_ei = self._backflow(elec_pos)
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
        jastrow_part = IsotropicAtomicExpDecay(self.kernel_initializer_jastrow)(r_ei)

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
            (elec pos of shape (..., n, d))
                -> (stream_1e of shape (..., n, d'), r_ei of shape (..., n, nion, d))
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
    backflow: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
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
        stream_1e, r_ei = self._backflow(elec_pos)
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
        jastrow_part = IsotropicAtomicExpDecay(self.kernel_initializer_jastrow)(r_ei)

        return antisymmetric_part + jastrow_part
