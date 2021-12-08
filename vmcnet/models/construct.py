"""Combine pieces to form full models."""
# TODO (ggoldsh): split this file into smaller component files
from enum import Enum
import functools
from typing import Callable, List, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from vmcnet.models import antiequivariance
from vmcnet.utils.slog_helpers import array_to_slog, slog_sum_over_axis
from vmcnet.utils.typing import (
    ArrayList,
    Backflow,
    ComputeInputStreams,
    Jastrow,
    ParticleSplit,
    SLArray,
)
from .antisymmetry import (
    GenericAntisymmetrize,
    FactoredAntisymmetrize,
    slogdet_product,
)
from .core import (
    Activation,
    AddedModel,
    SimpleResNet,
    get_nelec_per_split,
    get_nsplits,
    get_spin_split,
)
from .equivariance import (
    FermiNetBackflow,
    FermiNetOneElectronLayer,
    FermiNetOrbitalLayer,
    FermiNetResidualBlock,
    FermiNetTwoElectronLayer,
    compute_input_streams,
)
from .invariance import InvariantTensor
from .jastrow import (
    BackflowJastrow,
    OneBodyExpDecay,
    get_two_body_decay_scaled_for_chargeless_molecules,
)
from .sign_symmetry import (
    ProductsSignCovariance,
    make_array_list_fn_sign_covariant,
    make_array_list_fn_sign_invariant,
)
from .weights import (
    WeightInitializer,
    get_bias_init_from_config,
    get_kernel_init_from_config,
    get_kernel_initializer,
)

VALID_JASTROW_TYPES = [
    "one_body_decay",
    "two_body_decay",
    "backflow_based",
    "two_body_decay_and_backflow_based",
]


class DeterminantFnMode(Enum):
    """Enum specifying how to use determinant resnet in FermiNet model."""

    SIGN_COVARIANCE = 0
    PARALLEL_EVEN = 1
    PAIRWISE_EVEN = 2


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


def slog_psi_to_log_psi_apply(
    slog_psi_apply: Callable[..., SLArray]
) -> Callable[..., jnp.ndarray]:
    """Get a log|psi| model apply callable from a sign(psi), log|psi| apply callable."""

    def log_psi_apply(*args) -> jnp.ndarray:
        return slog_psi_apply(*args)[1]

    return log_psi_apply


def get_model_from_config(
    model_config: ConfigDict,
    nelec: jnp.ndarray,
    ion_pos: jnp.ndarray,
    ion_charges: jnp.ndarray,
    dtype=jnp.float32,
) -> flax.linen.Module:
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

    if model_config.type in [
        "ferminet",
        "embedded_particle_ferminet",
        "extended_orbital_matrix_ferminet",
    ]:
        determinant_fn = None
        resnet_config = model_config.det_resnet
        if model_config.use_det_resnet:
            determinant_fn = get_resnet_determinant_fn_for_ferminet(
                resnet_config.ndense,
                resnet_config.nlayers,
                _get_named_activation_fn(resnet_config.activation),
                kernel_init_constructor(resnet_config.kernel_init),
                bias_init_constructor(resnet_config.bias_init),
                resnet_config.use_bias,
                resnet_config.register_kfac,
            )

        # TODO(Jeffmin): make interface more flexible w.r.t. different types of Jastrows
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
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
                determinant_fn=determinant_fn,
                determinant_fn_mode=DeterminantFnMode[resnet_config.mode.upper()],
                full_det=model_config.full_det,
            )
        elif model_config.type == "embedded_particle_ferminet":
            total_nelec = jnp.array(model_config.nhidden_fermions_per_spin) + nelec
            total_spin_split = get_spin_split(total_nelec)

            backflow = get_backflow_from_config(
                model_config.backflow,
                total_spin_split,
                dtype=dtype,
            )
            invariance_config = model_config.invariance
            invariance_compute_input_streams = get_compute_input_streams_from_config(
                invariance_config.input_streams, ion_pos
            )
            invariance_backflow = get_backflow_from_config(
                invariance_config.backflow,
                spin_split,
                dtype=dtype,
            )

            return EmbeddedParticleFermiNet(
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
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
                determinant_fn=determinant_fn,
                determinant_fn_mode=DeterminantFnMode[resnet_config.mode.upper()],
                full_det=model_config.full_det,
                nhidden_fermions_per_spin=model_config.nhidden_fermions_per_spin,
                invariance_compute_input_streams=invariance_compute_input_streams,
                invariance_backflow=invariance_backflow,
                invariance_kernel_initializer=kernel_init_constructor(
                    invariance_config.kernel_initializer
                ),
                invariance_bias_initializer=bias_init_constructor(
                    invariance_config.bias_initializer
                ),
                invariance_use_bias=invariance_config.use_bias,
                invariance_register_kfac=invariance_config.register_kfac,
            )
        elif model_config.type == "extended_orbital_matrix_ferminet":
            invariance_config = model_config.invariance
            if model_config.use_separate_invariance_backflow:
                invariance_backflow = get_backflow_from_config(
                    invariance_config.backflow,
                    spin_split,
                    dtype=dtype,
                )
            else:
                invariance_backflow = None
            return ExtendedOrbitalMatrixFermiNet(
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
                bias_initializer_orbital_linear=bias_init_constructor(
                    model_config.bias_init_orbital_linear
                ),
                orbitals_use_bias=model_config.orbitals_use_bias,
                isotropic_decay=model_config.isotropic_decay,
                determinant_fn=determinant_fn,
                determinant_fn_mode=DeterminantFnMode[resnet_config.mode.upper()],
                full_det=model_config.full_det,
                nhidden_fermions_per_spin=model_config.nhidden_fermions_per_spin,
                invariance_backflow=invariance_backflow,
                invariance_kernel_initializer=kernel_init_constructor(
                    invariance_config.kernel_initializer
                ),
                invariance_bias_initializer=bias_init_constructor(
                    invariance_config.bias_initializer
                ),
                invariance_use_bias=invariance_config.use_bias,
                invariance_register_kfac=invariance_config.register_kfac,
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

        array_list_sign_covariance = get_sign_covariance_from_config(
            model_config, spin_split, kernel_init_constructor, dtype
        )

        return AntiequivarianceNet(
            spin_split,
            compute_input_streams,
            backflow,
            antieq_layer,
            array_list_sign_covariance,
            multiply_by_eq_features=model_config.multiply_by_eq_features,
        )

    elif model_config.type == "brute_force_antisym":
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
            jastrow = OneBodyExpDecay(
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
        if model_config.antisym_type == "factored":
            return FactoredAntisymmetry(
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
                "Unsupported brute-force antisymmetry type; {} was requested".format(
                    model_config.antisym_type
                )
            )
    else:
        raise ValueError(
            "Unsupported model type; {} was requested".format(model_config.type)
        )


def get_compute_input_streams_from_config(
    input_streams_config: ConfigDict, ion_pos: Optional[jnp.ndarray] = None
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

    return FermiNetBackflow(residual_blocks)


def get_sign_covariance_from_config(
    model_config: ConfigDict,
    spin_split: ParticleSplit,
    kernel_init_constructor: Callable[[ConfigDict], WeightInitializer],
    dtype: jnp.dtype,
) -> Callable[[ArrayList], jnp.ndarray]:
    """Get a sign covariance from a model config, for use in AntiequivarianceNet."""
    if model_config.use_products_covariance:
        return ProductsSignCovariance(
            1,
            kernel_init_constructor(model_config.products_covariance.kernel_init),
            model_config.products_covariance.register_kfac,
            use_weights=model_config.products_covariance.use_weights,
        )

    else:

        def backflow_based_equivariance(x: ArrayList) -> jnp.ndarray:
            concat_x = jnp.concatenate(x, axis=-2)
            return get_backflow_from_config(
                model_config.invariance,
                spin_split=spin_split,
                dtype=dtype,
            )(concat_x)

        odd_equivariance = make_array_list_fn_sign_covariant(
            backflow_based_equivariance, axis=-3
        )
        return lambda x: jnp.sum(odd_equivariance(x), axis=-2)


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


DeterminantFn = Callable[[int, ArrayList], jnp.ndarray]


def get_resnet_determinant_fn_for_ferminet(
    ndense: int,
    nlayers: int,
    activation: Activation,
    kernel_initializer: WeightInitializer,
    bias_initializer: WeightInitializer,
    use_bias: bool = True,
    register_kfac: bool = False,
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
        use_bias(bool): Whether to use a bias in the ResNet. Defaults to True.
        register_kfac (bool): Whether to register the ResNet Dense layers with KFAC.
            Currently, params for this ResNet explode to huge values and cause nans
            if register_kfac is True, so this flag defaults to false and should only be
            overridden with care. The reason for the instability is not known.

    Returns:
        (DeterminantFn): A resnet-based function. Has the signature
        dout, [nspins: (..., ndeterminants)] -> (..., dout).
    """

    def fn(dout: int, det_values: ArrayList) -> jnp.ndarray:
        concat_values = jnp.concatenate(det_values, axis=-1)
        return SimpleResNet(
            ndense,
            dout,
            nlayers,
            activation,
            kernel_initializer,
            bias_initializer,
            use_bias,
            register_kfac=register_kfac,
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


class FermiNet(flax.linen.Module):
    """FermiNet/generalized Slater determinant model.

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
        orbitals_use_bias (bool): whether to add a bias term in the linear part of the
            orbitals.
        isotropic_decay (bool): whether the decay for each ion should be anisotropic
            (w.r.t. the dimensions of the input), giving envelopes of the form
            exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
        determinant_fn (DeterminantFn or None): A optional function with signature
            dout, [nspins: (..., ndeterminants)] -> (..., dout).

            If not None, the function will be used to calculate Psi based on the
            outputs of the orbital matrix determinants. Depending on the
            determinant_fn_mode selected, this function can be used in one of several
            ways.

            If the mode is SIGN_COVARIANCE, the function will use d=1
            and will be explicitly symmetrized over the sign group, on a per-spin basis,
            to be sign-covariant (odd). If PARALLEL_EVEN or PAIRWISE_EVEN are
            selected, the function will be symmetrized to be spin-wise sign invariant
            (even).

            For PARALLEL_EVEN, the function will use d=ndeterminants, and each
            output will be multiplied by the product of corresponding determinants. That
            is, for 2 spins, with up determinants u_i and down determinants d_i, the
            ansatz will be sum_{i}(u_i * d_i * f_i(u,d)), where f_i(u,d) is the
            symmetrized determinant function.

            For PAIRWISE_EVEN, the function will use
            d=ndeterminants**nspins, and each output will again be multiplied by a
            product of determinants, but this time the determinants will range over all
            pairs. That is, for 2 spins, the ansatz will be
            sum_{i, j}(u_i * d_j * f_{i,j}(u,d)). Currently, PAIRWISE_EVEN mode only
            supports nspins = 2.

            If None, the equivalent of PARALLEL_EVEN mode (overriding any set
            determinant_fn_mode) is used without a symmetrized resnet (so the output,
            before any log-transformations, is a sum of products of determinants).
        determinant_fn_mode (DeterminantFnMode): One of SIGN_COVARIANCE,
            PARALLEL_EVEN, or PAIRWISE_EVEN. Used to decide how exactly to use the
            provided determinant_fn to calculate an ansatz for Psi; irrelevant
            if determinant_fn is set to None.
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
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool
    isotropic_decay: bool
    determinant_fn: Optional[DeterminantFn]
    determinant_fn_mode: DeterminantFnMode
    full_det: bool

    def _get_bad_determinant_fn_mode_error(self) -> ValueError:
        raise ValueError(
            "Only supported determinant function modes are SIGN_COVARIANCE, "
            "PARALLEL_EVEN, and PAIRWISE_EVEN. Received {}.".format(
                self.determinant_fn_mode
            )
        )

    def setup(self):
        """Setup backflow and symmetrized determinant function."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._compute_input_streams = self.compute_input_streams
        self._backflow = self.backflow
        self._symmetrized_det_fn = None
        if self.determinant_fn is not None:
            if self.determinant_fn_mode == DeterminantFnMode.SIGN_COVARIANCE:
                self._symmetrized_det_fn = make_array_list_fn_sign_covariant(
                    functools.partial(self.determinant_fn, 1)
                )
            elif self.determinant_fn_mode == DeterminantFnMode.PARALLEL_EVEN:
                self._symmetrized_det_fn = make_array_list_fn_sign_invariant(
                    functools.partial(self.determinant_fn, self.ndeterminants)
                )
            elif self.determinant_fn_mode == DeterminantFnMode.PAIRWISE_EVEN:
                # TODO (ggoldsh): build support for PAIRWISE_EVEN for nspins != 2
                self._symmetrized_det_fn = make_array_list_fn_sign_invariant(
                    functools.partial(self.determinant_fn, self.ndeterminants ** 2)
                )
            else:
                raise self._get_bad_determinant_fn_mode_error()

    def _calculate_psi_parallel_even(self, fn_inputs: ArrayList):
        """Calculate psi as an even fn. times products of corresponding determinants.

        Arguments:
            fn_inputs (ArrayList): input data of shape
                [norb_splits: (..., ndeterminants)]
        """
        # even_outputs has shape (..., ndeterminantss)
        even_outputs = self._symmetrized_det_fn(fn_inputs)
        # stacked_dets has shape (..., ndeterminants, norb_splits)
        stacked_dets = jnp.stack(fn_inputs, axis=-1)
        # prod_dets has shape (..., ndeterminants)
        prod_dets = jnp.prod(stacked_dets, axis=-1)
        return jnp.sum(prod_dets * even_outputs, axis=-1)

    def _calculate_psi_pairwise_even(self, fn_inputs: ArrayList):
        """Calculate psi as an even fn. times products of all pairs of determinants.

        Arguments:
            fn_inputs (ArrayList): input data of shape
                [norb_splits: (..., ndeterminants)]
        """
        if len(fn_inputs) != 2:
            raise ValueError(
                "For PAIRWISE_EVEN determinant_fn_mode, only norb_splits=2 is "
                "supported, but received {} orbital splits.".format(len(fn_inputs))
            )

        # even_outputs is shape (..., ndeterminants**2)
        even_outputs = self._symmetrized_det_fn(fn_inputs)

        # up_dets, down_dets are shape (..., ndeterminants, 1),  (..., 1, ndeterminants)
        up_dets = jnp.expand_dims(fn_inputs[0], -1)
        down_dets = jnp.expand_dims(fn_inputs[1], -2)
        # prod_dets is shape (..., ndeterminants, ndeterminants)
        prod_dets = up_dets * down_dets
        # Reshape prod_dets to (..., ndeterminants**2)
        prod_dets = jnp.reshape(
            prod_dets,
            (
                *prod_dets.shape[:-2],
                prod_dets.shape[-1] * prod_dets.shape[-2],
            ),
        )
        return jnp.sum(prod_dets * even_outputs, axis=-1)

    def _get_elec_pos_and_orbitals_split(
        self, elec_pos: jnp.ndarray
    ) -> Tuple[jnp.ndarray, ParticleSplit]:
        return elec_pos, self.spin_split

    def _get_norbitals_per_split(
        self, elec_pos: jnp.ndarray, orbitals_split: ParticleSplit
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
        input_stream_1e: jnp.ndarray,
        input_stream_2e: Optional[jnp.ndarray],
        stream_1e: jnp.ndarray,
        r_ei: Optional[jnp.ndarray],
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
            use_bias=self.orbitals_use_bias,
            isotropic_decay=self.isotropic_decay,
        )(stream_1e, r_ei)
        # Reshape to [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
        return _reshape_raw_ferminet_orbitals(orbitals, self.ndeterminants)

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> SLArray:
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

        if self._symmetrized_det_fn is not None:
            # dets is ArrayList of shape [norb_splits: (ndeterminants, ...)]
            dets = jax.tree_map(jnp.linalg.det, orbitals)
            # Move axis to get shape [norb_splits: (..., ndeterminants)]
            fn_inputs = jax.tree_map(lambda x: jnp.moveaxis(x, 0, -1), dets)
            if self.determinant_fn_mode == DeterminantFnMode.SIGN_COVARIANCE:
                psi = jnp.squeeze(self._symmetrized_det_fn(fn_inputs), -1)
            elif self.determinant_fn_mode == DeterminantFnMode.PARALLEL_EVEN:
                psi = self._calculate_psi_parallel_even(fn_inputs)
            elif self.determinant_fn_mode == DeterminantFnMode.PAIRWISE_EVEN:
                psi = self._calculate_psi_pairwise_even(fn_inputs)
            else:
                raise self._get_bad_determinant_fn_mode_error()
            return array_to_slog(psi)

        # slog_det_prods is SLArray of shape (ndeterminants, ...)
        slog_det_prods = slogdet_product(orbitals)
        return slog_sum_over_axis(slog_det_prods)


class EmbeddedParticleFermiNet(FermiNet):
    """Model that expands its inputs with extra hidden particles, then applies FermiNet.

    Note: the backflow argument supplied for the construction of this model should use
    the spin_split for the TOTAL number of particles, visible and hidden, for each spin,
    not the standard spin_split for the visible particles only.

    Attributes:
        nhidden_fermions_per_spin (Sequence[int]): number of hidden fermions to
            generate for each spin. Must have length nspins.
        invariance_compute_input_streams (ComputeInputStreams): function to compute
            input streams from electron positions, for the invariance that is used to
            generate the hidden particle positions. Has the signature
            (elec_pos of shape (..., n, d)) -> (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
                optional r_ei of shape (..., n, nion, d),
                optional r_ee of shape (..., n, n, d),
            )
        invariance_backflow (Callable): backflow function to be used for the invariance
            which generates the hidden fermion positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d')
        invariance_kernel_initializer (WeightInitializer): kernel initializer for the
            invariance dense layer. Has signature (key, shape, dtype) -> jnp.ndarray
        invariance_bias_initializer (WeightInitializer): bias initializer for the
            invariance dense layer. Has signature (key, shape, dtype) -> jnp.ndarray
        invariance_use_bias: (bool, optional): whether to add a bias term in the dense
            layer of the invariance. Defaults to True.
        invariance_register_kfac (bool, optional): whether to register the dense layer
            of the invariance with KFAC. Defaults to True.
    """

    nhidden_fermions_per_spin: Sequence[int]
    invariance_compute_input_streams: ComputeInputStreams
    invariance_backflow: Backflow
    invariance_kernel_initializer: WeightInitializer
    invariance_bias_initializer: WeightInitializer
    invariance_use_bias: bool = True
    invariance_register_kfac: bool = True

    def setup(self):
        """Setup EmbeddedParticleFermiNet."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        super().setup()
        self._invariance_compute_input_streams = self.invariance_compute_input_streams

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

    def _get_elec_pos_and_orbitals_split(
        self, elec_pos: jnp.ndarray
    ) -> Tuple[jnp.ndarray, ParticleSplit]:
        visible_nelec_total = elec_pos.shape[-2]
        d = elec_pos.shape[-1]
        visible_nelec_per_spin = get_nelec_per_split(
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
        orbitals_split = get_spin_split(total_nelec_per_spin)

        split_input_particles = jnp.split(elec_pos, self.spin_split, axis=-2)
        (
            invariance_stream_1e,
            invariance_stream_2e,
            _,
            _,
        ) = self._invariance_compute_input_streams(elec_pos)
        split_hidden_particles = invariance(invariance_stream_1e, invariance_stream_2e)
        # Create list of [visible_pos_spin1, hidden_pos_spin1, visible_pos_spin2, ...],
        # so that the full input positions can be created with a single concatenation.
        split_all_particles = [
            p
            for i in range(nspins)
            for p in [split_input_particles[i], split_hidden_particles[i]]
        ]
        concat_all_particles = jnp.concatenate(split_all_particles, axis=-2)

        return concat_all_particles, orbitals_split


class ExtendedOrbitalMatrixFermiNet(FermiNet):
    """FermiNet-based model with larger orbital matrices via padding with invariance.

    Attributes:
        nhidden_fermions_per_spin (Sequence[int], optional): sequence of integers
            specifying how many extra hidden particle dimensions and corresponding
            virtual orbitals to add to the orbital matrices. If not None, must have
            length nspins. Defaults to None (no extra dims added, equivalent to
            FermiNet).
        invariance_kernel_initializer (WeightInitializer, optional): kernel initializer
            for the invariance dense layer. Has signature
                (key, shape, dtype) -> jnp.ndarray.
            Defaults to an orthogonal initializer.
        invariance_bias_initializer (WeightInitializer, optional): bias initializer for
            the invariance dense layer. Has signature
                (key, shape, dtype) -> jnp.ndarray.
            Defaults to a scaled random normal initializer.
        invariance_use_bias: (bool, optional): whether to add a bias term in the dense
            layer of the invariance. Defaults to True.
        invariance_register_kfac (bool, optional): whether to register the dense layer
            of the invariance with KFAC. Defaults to True.
        invariance_backflow (Callable, optional): backflow function to be used for the
            invariance which generates the hidden fermion positions. If None, the
            outputs of the regular FermiNet backflow are used instead to form an
            invariance. Defaults to None.
    """

    nhidden_fermions_per_spin: Sequence[int]
    invariance_kernel_initializer: WeightInitializer = get_kernel_initializer(
        "orthogonal"
    )
    invariance_bias_initializer: WeightInitializer = get_kernel_initializer("normal")
    invariance_use_bias: bool = True
    invariance_register_kfac: bool = True
    invariance_backflow: Optional[Backflow] = None

    def _get_invariance(
        self,
        norbitals_per_split: Sequence[int],
        input_stream_1e: jnp.ndarray,
        input_stream_2e: Optional[jnp.ndarray],
        stream_1e: jnp.ndarray,
    ) -> ArrayList:
        invariant_shape_per_spin = [
            (nhidden, norbitals_per_split[i] * self.ndeterminants)
            for i, nhidden in enumerate(self.nhidden_fermions_per_spin)
        ]

        if self.invariance_backflow is not None:
            # Compute the backflow a single time here rather than in the
            # InvariantTensors below to avoid duplicating the computation for each
            # determinant and having kfac complain about duplicate registrations.
            stream_1e = self.invariance_backflow(input_stream_1e, input_stream_2e)

        invariance = InvariantTensor(
            split=self.spin_split,
            output_shape_per_split=invariant_shape_per_spin,
            backflow=None,
            kernel_initializer=self.invariance_kernel_initializer,
            bias_initializer=self.invariance_bias_initializer,
            use_bias=self.invariance_use_bias,
            register_kfac=self.invariance_register_kfac,
        )(stream_1e, None)
        # Reshape to [norb_splits: (ndeterminants, ..., nhidden[i], norbitals[i])]
        return _reshape_raw_ferminet_orbitals(invariance, self.ndeterminants)

    def _get_norbitals_per_split(
        self, elec_pos: jnp.ndarray, orbitals_split: ParticleSplit
    ) -> Tuple[int, ...]:
        visible_nelec_total = elec_pos.shape[-2]

        if self.full_det:
            all_nelec_total = visible_nelec_total + sum(self.nhidden_fermions_per_spin)
            nsplits = get_nsplits(orbitals_split)
            return (all_nelec_total,) * nsplits

        nelec_per_split = get_nelec_per_split(orbitals_split, visible_nelec_total)
        return tuple(
            nelec_i + self.nhidden_fermions_per_spin[i]
            for i, nelec_i in enumerate(nelec_per_split)
        )

    def _eval_orbitals(
        self,
        orbitals_split: ParticleSplit,
        norbitals_per_split: Sequence[int],
        input_stream_1e: jnp.ndarray,
        input_stream_2e: Optional[jnp.ndarray],
        stream_1e: jnp.ndarray,
        r_ei: Optional[jnp.ndarray],
    ) -> ArrayList:
        # Shape is [norb_splits: (ndeterminants, ..., nvisible[i], norbitals[i])]
        equivariant_part = super()._eval_orbitals(
            orbitals_split,
            norbitals_per_split,
            input_stream_1e,
            input_stream_2e,
            stream_1e,
            r_ei,
        )
        # Shape is [norb_splits: (ndeterminants, ..., nhidden[i], norbitals[i])]
        invariant_part = self._get_invariance(
            norbitals_per_split,
            input_stream_1e,
            input_stream_2e,
            stream_1e,
        )
        # Return shape [norb_splits: (ndeterminants, ..., ntotal[i], norbitals[i])]
        return [
            jnp.concatenate([eq, inv], axis=-2)
            for (eq, inv) in zip(equivariant_part, invariant_part)
        ]


class AntiequivarianceNet(flax.linen.Module):
    """Antisymmetry from anti-equivariance, backflow -> antieq -> odd invariance.

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
        antiequivariant_layer (Callable): function which computes antiequivariances-per-
            spin. Has the signature
            (stream_1e of shape (..., n, d_backflow), r_ei of shape (..., n, nion, d))
                -> (antieqs of shapes [spin: (..., n[spin], d_antieq)])
        array_list_sign_covariance (Callable): function which is sign-
            covariant with respect to each spin. Has the signature
            [(..., nelec[spin], d_antieq)]  -> (..., d_antisym). Since this function
            is sign covariant, its outputs are antisymmetric, so Psi can be calculated
            by summing over the final axis of the result.
        multiply_by_eq_features (bool, optional): If True, the antiequivariance from the
            antiequivariant_layer is multiplied by the equivariant features from the
            backflow before being fed into the sign covariant function. If False, the
            antiequivariance is processed directly by the sign covariant function.
            Defaults to False.
    """

    spin_split: ParticleSplit
    compute_input_streams: ComputeInputStreams
    backflow: Backflow
    antiequivariant_layer: Callable[[jnp.ndarray, jnp.ndarray], ArrayList]
    array_list_sign_covariance: Callable[[ArrayList], jnp.ndarray]
    multiply_by_eq_features: bool = False

    def setup(self):
        """Setup backflow."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._compute_input_streams = self.compute_input_streams
        self._backflow = self.backflow
        self._antiequivariant_layer = self.antiequivariant_layer
        self._array_list_sign_covariance = self.array_list_sign_covariance

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> SLArray:
        """Compose backflow -> antiequivariance -> sign covariant equivariance -> sum.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            jnp.ndarray: log(abs(psi)), where psi is a general odd invariance of an
            anti-equivariant backflow. If the inputs have shape (batch_dims, nelec, d),
            then the output has shape (batch_dims,).
        """
        stream_1e, stream_2e, r_ei, _ = self._compute_input_streams(elec_pos)
        backflow_out = self._backflow(stream_1e, stream_2e)
        antiequivariant_out = self._antiequivariant_layer(backflow_out, r_ei)

        if self.multiply_by_eq_features:
            antiequivariant_out = antiequivariance.multiply_antieq_by_eq_features(
                antiequivariant_out, backflow_out, self.spin_split
            )

        antisym_vector = self._array_list_sign_covariance(antiequivariant_out)
        return array_to_slog(jnp.sum(antisym_vector, axis=-1))


class FactoredAntisymmetry(flax.linen.Module):
    """A sum of products of brute-force antisymmetrized ResNets, composed with backflow.

    This connects the computational graph between a backflow, a factored antisymmetrized
    ResNet, and a jastrow.

    See https://arxiv.org/abs/2112.03491 for a description of the factored antisymmetric
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
        rank (int): The rank of the brute-force antisymmetry. In practical terms, the
            number of resnets to antisymmetrize for each spin. This is analogous to
            ndeterminants for regular FermiNet.
        ndense_resnet (int): number of dense nodes in each layer of each antisymmetrized
            ResNet
        nlayers_resnet (int): number of layers in each antisymmetrized ResNet
        kernel_initializer_resnet (WeightInitializer): kernel initializer for
            the dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNets. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNets. Has the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
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
    def __call__(self, elec_pos: jnp.ndarray) -> SLArray:
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
        input_stream_1e, input_stream_2e, r_ei, r_ee = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)
        split_spins = jnp.split(stream_1e, self.spin_split, axis=-2)

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

        slog_antisyms = FactoredAntisymmetrize(
            [fn_to_antisymmetrize for _ in split_spins]
        )(split_spins)
        sign_psi, log_antisyms = slog_sum_over_axis(slog_antisyms, axis=-1)

        jastrow_part = self._jastrow(
            input_stream_1e, input_stream_2e, stream_1e, r_ei, r_ee
        )

        return sign_psi, log_antisyms + jastrow_part


class GenericAntisymmetry(flax.linen.Module):
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
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_resnet (WeightInitializer): bias initializer for the
            dense layers in the antisymmetrized ResNet. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn_resnet (Activation): activation function in the antisymmetrized
            ResNet. Has the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
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

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> SLArray:
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
        input_stream_1e, input_stream_2e, r_ei, r_ee = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)
        split_spins = jnp.split(stream_1e, self.spin_split, axis=-2)
        sign_psi, log_antisym = GenericAntisymmetrize(
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
        jastrow_part = self._jastrow(
            input_stream_1e, input_stream_2e, stream_1e, r_ei, r_ee
        )

        return sign_psi, log_antisym + jastrow_part
