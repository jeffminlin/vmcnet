"""Combine pieces to form full models."""

# TODO (ggoldsh): split this file into smaller component files
import functools
from typing import Callable, List, Optional, Sequence, Tuple, cast

import chex
import flax
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from vmcnet.utils.slog_helpers import slog_sum_over_axis, array_to_slog
from vmcnet.utils.typing import (
    Array,
    ArrayList,
    Backflow,
    ComputeInputStreams,
    Jastrow,
    ParticleSplit,
    SLArray,
)
from .antisymmetry import (
    GenericAntisymmetrize,
    FactorizedAntisymmetrize,
    slogdet_product,
)
from .core import (
    Activation,
    AddedModel,
    SimpleResNet,
    Module,
    get_nelec_per_split,
    get_nsplits,
    get_spin_split,
    split,
)
from .equivariance import (
    FermiNetBackflow,
    FermiNetOneElectronLayer,
    FermiNetOrbitalLayer,
    FermiNetResidualBlock,
    FermiNetTwoElectronLayer,
    compute_input_streams,
)
from .jastrow import (
    BackflowJastrow,
    OneBodyExpDecay,
    get_two_body_decay_scaled_for_chargeless_molecules,
)
from .weights import (
    WeightInitializer,
    get_bias_init_from_config,
    get_kernel_init_from_config,
)

VALID_JASTROW_TYPES = [
    "one_body_decay",
    "two_body_decay",
    "backflow_based",
    "two_body_decay_and_backflow_based",
]


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


def slog_psi_to_log_psi_apply(slog_psi_apply) -> Callable[..., Array]:
    """Get a log|psi| model apply callable from a sign(psi), log|psi| apply callable."""

    def log_psi_apply(*args) -> Array:
        return slog_psi_apply(*args)[1]

    return log_psi_apply


def get_model_from_config(
    model_config: ConfigDict,
    nelec: Array,
    ion_pos: Array,
    ion_charges: Array,
    dtype=jnp.float32,
) -> Module:
    """Get a model from a hyperparameter config."""
    spin_split = get_spin_split(nelec)

    compute_input_streams = get_compute_input_streams_from_config(
        model_config.input_streams, ion_pos
    )
    backflow = get_backflow_from_config(
        model_config.backflow,
        spin_split,
        dtype=dtype,
    )

    kernel_init_constructor, bias_init_constructor = _get_dtype_init_constructors(dtype)

    if model_config.type == "ferminet":
        return FermiNet(
            spin_split,
            compute_input_streams,
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
            envelope_softening=model_config.envelope_softening,
            bias_initializer_orbital_linear=bias_init_constructor(
                model_config.bias_init_orbital_linear
            ),
            orbitals_use_bias=model_config.orbitals_use_bias,
            isotropic_decay=model_config.isotropic_decay,
            full_det=model_config.full_det,
        )
    elif model_config.type == "explicit_antisym":
        jastrow_config = model_config.jastrow

        def _get_two_body_decay_jastrow():
            return get_two_body_decay_scaled_for_chargeless_molecules(
                ion_pos,
                ion_charges,
                init_ee_strength=jastrow_config.two_body_decay.init_ee_strength,
                trainable=jastrow_config.two_body_decay.trainable,
            )

        def _get_backflow_based_jastrow():
            if jastrow_config.backflow_based.use_separate_jastrow_backflow:
                jastrow_backflow = get_backflow_from_config(
                    jastrow_config.backflow_based.backflow,
                    spin_split,
                    dtype=dtype,
                )
            else:
                jastrow_backflow = None
            return BackflowJastrow(backflow=jastrow_backflow)

        if jastrow_config.type == "one_body_decay":
            jastrow: Jastrow = OneBodyExpDecay(
                kernel_initializer=kernel_init_constructor(
                    jastrow_config.one_body_decay.kernel_init
                )
            )
        elif jastrow_config.type == "two_body_decay":
            jastrow = _get_two_body_decay_jastrow()
        elif jastrow_config.type == "backflow_based":
            jastrow = _get_backflow_based_jastrow()
        elif jastrow_config.type == "two_body_decay_and_backflow_based":
            two_body_decay_jastrow = _get_two_body_decay_jastrow()
            backflow_jastrow = _get_backflow_based_jastrow()
            jastrow = AddedModel([two_body_decay_jastrow, backflow_jastrow])
        else:
            raise ValueError(
                "Unsupported jastrow type; {} was requested, but the only supported "
                "types are: {}".format(
                    jastrow_config.type, ", ".join(VALID_JASTROW_TYPES)
                )
            )
        if model_config.antisym_type == "factorized":
            return FactorizedAntisymmetry(
                spin_split,
                compute_input_streams,
                backflow,
                jastrow,
                rank=model_config.rank,
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=kernel_init_constructor(
                    model_config.kernel_init_resnet
                ),
                bias_initializer_resnet=bias_init_constructor(
                    model_config.bias_init_resnet
                ),
                activation_fn_resnet=_get_named_activation_fn(
                    model_config.activation_fn_resnet
                ),
                resnet_use_bias=model_config.resnet_use_bias,
            )
        elif model_config.antisym_type == "generic":
            return GenericAntisymmetry(
                spin_split,
                compute_input_streams,
                backflow,
                jastrow,
                ndense_resnet=model_config.ndense_resnet,
                nlayers_resnet=model_config.nlayers_resnet,
                kernel_initializer_resnet=kernel_init_constructor(
                    model_config.kernel_init_resnet
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
                "Unsupported explicit antisymmetry type; {} was requested".format(
                    model_config.antisym_type
                )
            )
    else:
        raise ValueError(
            "Unsupported model type; {} was requested".format(model_config.type)
        )


def get_compute_input_streams_from_config(
    input_streams_config: ConfigDict, ion_pos: Optional[Array] = None
) -> ComputeInputStreams:
    """Get a function for computing input streams from a model configuration."""
    return functools.partial(
        compute_input_streams,
        ion_pos=ion_pos,
        include_2e_stream=input_streams_config.include_2e_stream,
        include_ei_norm=input_streams_config.include_ei_norm,
        include_ee_norm=input_streams_config.include_ee_norm,
    )


def get_backflow_from_config(
    backflow_config,
    spin_split,
    dtype=jnp.float32,
) -> Module:
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
        one_electron_skip=backflow_config.one_electron_skip,
        one_electron_skip_scale=backflow_config.one_electron_skip_scale,
        two_electron_skip=backflow_config.two_electron_skip,
        two_electron_skip_scale=backflow_config.two_electron_skip_scale,
    )

    return FermiNetBackflow(residual_blocks)


def get_residual_blocks_for_ferminet_backflow(
    spin_split: ParticleSplit,
    ndense_list: List[Tuple[int, ...]],
    kernel_initializer_unmixed: WeightInitializer,
    kernel_initializer_mixed: WeightInitializer,
    kernel_initializer_2e_1e_stream: WeightInitializer,
    kernel_initializer_2e_2e_stream: WeightInitializer,
    bias_initializer_1e_stream: WeightInitializer,
    bias_initializer_2e_stream: WeightInitializer,
    activation_fn: Activation,
    use_bias: bool = True,
    one_electron_skip: bool = True,
    one_electron_skip_scale: float = 1.0,
    two_electron_skip: bool = True,
    two_electron_skip_scale: float = 1.0,
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
            signature (key, shape, dtype) -> Array
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> Array
        kernel_initializer_2e_1e_stream (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> Array
        kernel_initializer_2e_2e_stream (WeightInitializer): kernel initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> Array
        bias_initializer_1e_stream (WeightInitializer): bias initializer for the
            one-electron stream. Has signature (key, shape, dtype) -> Array
        bias_initializer_2e_stream (WeightInitializer): bias initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> Array
        activation_fn (Activation): activation function in the electron streams. Has
            the signature Array -> Array (shape is preserved)
        use_bias (bool, optional): whether to add a bias term in the electron streams.
            Defaults to True.
        one_electron_skip (bool, optional): whether to add a residual skip connection to
            the one-electron layer whenever the shapes of the input and output match.
            Defaults to True.
        one_electron_skip_scale (float, optional): quantity to scale the one-electron
            output by if a skip connection is added. Defaults to 1.0.
        two_electron_skip (bool, optional): whether to add a residual skip connection to
            the two-electron layer whenever the shapes of the input and output match.
            Defaults to True.
        two_electron_skip_scale (float, optional): quantity to scale the two-electron
            output by if a skip connection is added. Defaults to 1.0.
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
            skip_connection=one_electron_skip,
            skip_connection_scale=one_electron_skip_scale,
        )
        two_electron_layer = None
        if len(ndense) > 1:
            two_electron_layer = FermiNetTwoElectronLayer(
                ndense[1],
                kernel_initializer_2e_2e_stream,
                bias_initializer_2e_stream,
                activation_fn,
                use_bias,
                skip_connection=two_electron_skip,
                skip_connection_scale=two_electron_skip_scale,
            )
        residual_blocks.append(
            FermiNetResidualBlock(one_electron_layer, two_electron_layer)
        )
    return residual_blocks


DeterminantFn = Callable[[int, ArrayList], Array]


def get_resnet_determinant_fn_for_ferminet(
    ndense: int,
    nlayers: int,
    activation: Activation,
    kernel_initializer: WeightInitializer,
    bias_initializer: WeightInitializer,
    use_bias: bool = True,
) -> DeterminantFn:
    """Get a resnet-based determinant function for FermiNet construction.

    The returned function is used as a more general way to combine the determinant
    outputs into the final wavefunction value, relative to the original method of a
    sum of products. The function takes as its first argument the number of requested
    output features because several variants of this method are supported, and each
    requires the function to generate a different output size.

    Args:
        ndense (int): the number of neurons in the dense layers
            of the ResNet.
        nlayers (int): the number of layers in the ResNet.
        activation (Activation): the activation function to use for the  resnet.
        kernel_initializer (WeightInitializer): kernel initializer for the resnet.
        bias_initializer (WeightInitializer): bias initializer for the resnet.
        use_bias (bool): Whether to use a bias in the ResNet. Defaults to True.

    Returns:
        (DeterminantFn): A resnet-based function. Has the signature
        dout, [nspins: (..., ndeterminants)] -> (..., dout).
    """

    def fn(dout: int, det_values: ArrayList) -> Array:
        concat_values = jnp.concatenate(det_values, axis=-1)
        return SimpleResNet(
            ndense,
            dout,
            nlayers,
            activation,
            kernel_initializer,
            bias_initializer,
            use_bias,
        )(concat_values)

    return fn


def _reshape_raw_ferminet_orbitals(
    orbitals: ArrayList, ndeterminants: int
) -> ArrayList:
    """Reshape orbitals returned from FermiNetOrbitalLayer for use by FermiNet.

    Args:
        orbitals (Arraylist): orbitals generated from the FermiNetOrbitalLayer,
            of shape [norb_splits: (..., nelec[i], norbitals[i] * ndeterminants)].
        ndeterminants (int): the number of determinants used.

    Returns:
        ArrayList: input orbitals reshaped to
        [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
    """
    # Reshape to [norb_splits: (..., nhidden[i], ndeterminants, norbitals[i])]
    orbitals = [
        jnp.reshape(
            orb,
            (
                *orb.shape[:-1],
                # This ordering ensures elements corresponding to a single determinant
                # come from adjacent blocks of the raw orbitals.
                ndeterminants,
                orb.shape[-1] // ndeterminants,
            ),
        )
        for orb in orbitals
    ]
    # Move axis to [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
    return [jnp.moveaxis(orb, -2, 0) for orb in orbitals]


class FermiNet(Module):
    """FermiNet/generalized Slater determinant model.

    This model was first introduced in the following papers:
        https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429
        https://arxiv.org/abs/2011.07125
    Their repository can be found at https://github.com/deepmind/ferminet, which
    includes a JAX branch.

    Attributes:
        spin_split (ParticleSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        compute_input_streams (ComputeInputStreams): function to compute input
            streams from electron positions. Has the signature
            (elec_pos of shape (..., n, d)) -> (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
                optional r_ei of shape (..., n, nion, d),
                optional r_ee of shape (..., n, n, d),
            )
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d')
        ndeterminants (int): number of determinants in the FermiNet model, i.e. the
            number of distinct orbital layers applied
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear part of the orbitals. Has signature
            (key, shape, dtype) -> Array
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer for the
            decay rate in the exponential envelopes. If `isotropic_decay` is True, then
            this initializes a single decay rate number per ion and orbital. If
            `isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> Array
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer for the
            linear combination over the ions of exponential envelopes. Has signature
            (key, shape, dtype) -> Array
        envelope_softening (float): amount by which to soften the cusp of the
            exponential envelope. If set to c, then an ei distance of r is replaced by
            sqrt(r^2 + c^2) - c.
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> Array
        orbitals_use_bias (bool): whether to add a bias term in the linear part of the
            orbitals.
        isotropic_decay (bool): whether the decay for each ion should be anisotropic
            (w.r.t. the dimensions of the input), giving envelopes of the form
            exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
        full_det (bool): If True, the model will use a single, "full" determinant with
            orbitals from particles of all spins. For example, for a spin_split of
            (2,2), the original FermiNet with ndeterminants=1 would calculate two
            separate 2x2 orbital matrices and multiply their determinants together. A
            full determinant model would instead calculate a single 4x4 matrix, with the
            first two particle indices corresponding to the up-spin particles and the
            last two particle indices corresponding to the down-spin particles. The
            output of the model would then be the determinant of that single matrix, if
            ndeterminants=1, or the sum of multiple such determinants if
            ndeterminants>1.
    """

    spin_split: ParticleSplit
    compute_input_streams: ComputeInputStreams
    backflow: Backflow
    ndeterminants: int
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    envelope_softening: chex.Scalar
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool
    isotropic_decay: bool
    full_det: bool
    bosons: bool = False

    def setup(self):
        """Setup backflow and compute_input_streams."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._compute_input_streams = self.compute_input_streams
        self._backflow = self.backflow

    def _get_elec_pos_and_orbitals_split(
        self, elec_pos: Array
    ) -> Tuple[Array, ParticleSplit]:
        return elec_pos, self.spin_split

    def _get_norbitals_per_split(
        self, elec_pos: Array, orbitals_split: ParticleSplit
    ) -> Tuple[int, ...]:
        nelec_total = elec_pos.shape[-2]

        if self.full_det:
            nsplits = get_nsplits(orbitals_split)
            return (nelec_total,) * nsplits

        return get_nelec_per_split(orbitals_split, nelec_total)

    def _eval_orbitals(
        self,
        orbitals_split: ParticleSplit,
        norbitals_per_split: Sequence[int],
        input_stream_1e: Array,
        input_stream_2e: Optional[Array],
        stream_1e: Array,
        r_ei: Optional[Array],
    ) -> ArrayList:
        # Input streams unused in orbitals for regular FermiNet
        del input_stream_1e
        del input_stream_2e
        # Multiply norbitals by ndeterminants to generate all orbitals in one go.
        norbitals_per_split = [n * self.ndeterminants for n in norbitals_per_split]
        # orbitals is shape [norb_splits: (..., nelec[i], norbitals[i] * ndeterminants)]
        orbitals = FermiNetOrbitalLayer(
            orbitals_split=orbitals_split,
            norbitals_per_split=norbitals_per_split,
            kernel_initializer_linear=self.kernel_initializer_orbital_linear,
            kernel_initializer_envelope_dim=self.kernel_initializer_envelope_dim,
            kernel_initializer_envelope_ion=self.kernel_initializer_envelope_ion,
            bias_initializer_linear=self.bias_initializer_orbital_linear,
            envelope_softening=self.envelope_softening,
            use_bias=self.orbitals_use_bias,
            isotropic_decay=self.isotropic_decay,
        )(stream_1e, r_ei)
        # Reshape to [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
        return _reshape_raw_ferminet_orbitals(orbitals, self.ndeterminants)

    @flax.linen.compact
    def __call__(self, elec_pos: Array) -> SLArray:  # type: ignore[override]
        """Compose FermiNet backflow -> orbitals -> logabs determinant product.

        Args:
            elec_pos (Array): array of particle positions (..., nelec, d)

        Returns:
            Array: FermiNet output; logarithm of the absolute value of a
            anti-symmetric function of elec_pos, where the anti-symmetry is with respect
            to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        elec_pos, orbitals_split = self._get_elec_pos_and_orbitals_split(elec_pos)

        input_stream_1e, input_stream_2e, r_ei, _ = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)

        norbitals_per_split = self._get_norbitals_per_split(elec_pos, orbitals_split)
        # orbitals is [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
        orbitals = self._eval_orbitals(
            orbitals_split,
            norbitals_per_split,
            input_stream_1e,
            input_stream_2e,
            stream_1e,
            r_ei,
        )

        if self.full_det:
            orbitals = [jnp.concatenate(orbitals, axis=-2)]

            if self.bosons:
                amplitudes = jnp.sum(
                    jnp.prod(jnp.sum(orbitals[0], axis=-1), axis=-1), axis=0
                )
                return array_to_slog(amplitudes)

        # slog_det_prods is SLArray of shape (ndeterminants, ...)
        slog_det_prods = slogdet_product(orbitals)
        return slog_sum_over_axis(slog_det_prods)


class FactorizedAntisymmetry(Module):
    """A sum of products of explicitly antisymmetrized ResNets, composed with backflow.

    This connects the computational graph between a backflow, a factorized
    antisymmetrized ResNet, and a jastrow.

    See https://arxiv.org/abs/2112.03491 for a description of the factorized
    antisymmetric layer.

    Attributes:
        spin_split (ParticleSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        compute_input_streams (ComputeInputStreams): function to compute input
            streams from electron positions. Has the signature
            (elec_pos of shape (..., n, d)) -> (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
                optional r_ei of shape (..., n, nion, d),
                optional r_ee of shape (..., n, n, d),
            )
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d')
        jastrow (Callable): function which computes a Jastrow factor from displacements.
            Has the signature
            (
                r_ei of shape (batch_dims, n, nion, d),
                r_ee of shape (batch_dims, n, n, d),
            )
                -> log jastrow of shape (batch_dims,)
        rank (int): The rank of the explicit antisymmetry. In practical terms, the
            number of resnets to antisymmetrize for each spin. This is analogous to
            ndeterminants for regular FermiNet.
        ndense_resnet (int): number of dense nodes in each layer of each antisymmetrized
            ResNet
        nlayers_resnet (int): number of layers in each antisymmetrized ResNet
        kernel_initializer_resnet (WeightInitializer): kernel initializer for
            the dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> Array
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> Array
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNets. Has the signature Array -> Array (shape is preserved)
        resnet_use_bias (bool, optional): whether to add a bias term in the dense layers
            of the antisymmetrized ResNets. Defaults to True.
    """

    spin_split: ParticleSplit
    compute_input_streams: ComputeInputStreams
    backflow: Backflow
    jastrow: Jastrow
    rank: int
    ndense_resnet: int
    nlayers_resnet: int
    kernel_initializer_resnet: WeightInitializer
    bias_initializer_resnet: WeightInitializer
    activation_fn_resnet: Activation
    resnet_use_bias: bool = True

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._compute_input_streams = self.compute_input_streams
        self._backflow = self.backflow
        self._jastrow = self.jastrow

    @flax.linen.compact
    def __call__(self, elec_pos: Array) -> SLArray:  # type: ignore[override]
        """Compose FermiNet backflow -> antisymmetrized ResNets -> logabs product.

        Args:
            elec_pos (Array): array of particle positions (..., nelec, d)

        Returns:
            Array: spinful antisymmetrized output; logarithm of the absolute value
            of a anti-symmetric function of elec_pos, where the anti-symmetry is with
            respect to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        input_stream_1e, input_stream_2e, r_ei, r_ee = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)
        split_spins = split(stream_1e, self.spin_split, axis=-2)

        def fn_to_antisymmetrize(x_one_spin):
            resnet_outputs = [
                SimpleResNet(
                    self.ndense_resnet,
                    1,
                    self.nlayers_resnet,
                    self.activation_fn_resnet,
                    self.kernel_initializer_resnet,
                    self.bias_initializer_resnet,
                    use_bias=self.resnet_use_bias,
                )(x_one_spin)
                for _ in range(self.rank)
            ]
            return jnp.concatenate(resnet_outputs, axis=-1)

        # TODO (ggoldsh/jeffminlin): better typing for the Array vs SLArray version of
        # this model to avoid having to cast the return type.
        slog_antisyms = cast(
            SLArray,
            FactorizedAntisymmetrize([fn_to_antisymmetrize for _ in split_spins])(
                split_spins
            ),
        )

        sign_psi, log_antisyms = slog_sum_over_axis(slog_antisyms, axis=-1)

        jastrow_part = self._jastrow(
            input_stream_1e, input_stream_2e, stream_1e, r_ei, r_ee
        )

        return sign_psi, log_antisyms + jastrow_part


class GenericAntisymmetry(Module):
    """A single ResNet antisymmetrized over all input leaves, composed with backflow.

    The ResNet is antisymmetrized with respect to each spin split separately (i.e. the
    antisymmetrization operators for each spin are composed and applied).

    This connects the computational graph between a backflow, a generic antisymmetrized
    ResNet, and a jastrow.

    See https://arxiv.org/abs/2112.03491 for a description of the generic antisymmetric
    layer.

    Attributes:
        spin_split (ParticleSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        compute_input_streams (ComputeInputStreams): function to compute input
            streams from electron positions. Has the signature
            (elec_pos of shape (..., n, d)) -> (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
                optional r_ei of shape (..., n, nion, d),
                optional r_ee of shape (..., n, n, d),
            )
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d')
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
            (key, shape, dtype) -> Array
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNet. Has signature
            (key, shape, dtype) -> Array
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNet. Has the signature Array -> Array (shape is preserved)
        resnet_use_bias (bool, optional): whether to add a bias term in the dense layers
            of the antisymmetrized ResNet. Defaults to True.
    """

    spin_split: ParticleSplit
    compute_input_streams: ComputeInputStreams
    backflow: Backflow
    jastrow: Jastrow
    ndense_resnet: int
    nlayers_resnet: int
    kernel_initializer_resnet: WeightInitializer
    bias_initializer_resnet: WeightInitializer
    activation_fn_resnet: Activation
    resnet_use_bias: bool = True

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._compute_input_streams = self.compute_input_streams
        self._backflow = self.backflow
        self._jastrow = self.jastrow
        self._activation_fn_resnet = self.activation_fn_resnet

    @flax.linen.compact
    def __call__(self, elec_pos: Array) -> SLArray:  # type: ignore[override]
        """Compose FermiNet backflow -> antisymmetrized ResNet -> logabs.

        Args:
            elec_pos (Array): array of particle positions (..., nelec, d)

        Returns:
            Array: spinful antisymmetrized output; logarithm of the absolute value
            of a anti-symmetric function of elec_pos, where the anti-symmetry is with
            respect to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        input_stream_1e, input_stream_2e, r_ei, r_ee = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)
        split_spins = split(stream_1e, self.spin_split, axis=-2)
        sign_psi, log_antisym = GenericAntisymmetrize(
            SimpleResNet(
                self.ndense_resnet,
                1,
                self.nlayers_resnet,
                self._activation_fn_resnet,
                self.kernel_initializer_resnet,
                self.bias_initializer_resnet,
                use_bias=self.resnet_use_bias,
            )
        )(split_spins)
        jastrow_part = self._jastrow(
            input_stream_1e, input_stream_2e, stream_1e, r_ei, r_ee
        )

        return sign_psi, log_antisym + jastrow_part
