# copied from
# ----------------------------------------------------------------------------------------------------
# vmcnet.models.construct
# ----------------------------------------------------------------------------------------------------
#
# changed to get *nonsym* model. TODO: maybe merge back into vmcnet.models.construct.get_model_from_config

from vmcnet.models.construct import (
    _get_dtype_init_constructors,
    _get_named_activation_fn,
    get_compute_input_streams_from_config,
    get_spin_split,
    get_backflow_from_config,
    get_resnet_determinant_fn_for_ferminet,
    DeterminantFnMode,
    VALID_JASTROW_TYPES
)
from .nonsymmetry import FermiNet_Nonsym
import jax.numpy as jnp
from ml_collections import ConfigDict
from vmcnet.utils.typing import Array
from vmcnet.models.core import (
    Module,
    get_spin_split,
)

def get_model_from_config_nonsym(
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

    ferminet_model_types = [
        "ferminet",
        "embedded_particle_ferminet",
        "extended_orbital_matrix_ferminet",
    ]
    if model_config.type in ferminet_model_types:
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
        if model_config.type == "ferminet":
            return FermiNet_Nonsym(
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
        else:
            raise ValueError( "Unsupported model type; {} was requested".format(model_config.type))
    else:
        raise ValueError( "Unsupported model type; {} was requested".format(model_config.type))

