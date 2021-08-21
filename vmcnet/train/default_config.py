"""Create configuration of hyperparameters."""
import copy
import os
from typing import Dict

from ml_collections import ConfigDict, FieldReference

from vmcnet.utils.checkpoint import CHECKPOINT_FILE_NAME

NO_RELOAD_LOG_DIR = "NONE"
DEFAULT_CONFIG_FILE_NAME = "config.json"


def get_default_reload_config() -> ConfigDict:
    """Make a default reload configuration (no logdir but valid defaults otherwise)."""
    return ConfigDict(
        {
            "logdir": NO_RELOAD_LOG_DIR,
            "use_config_file": True,
            "config_relative_file_path": DEFAULT_CONFIG_FILE_NAME,
            "use_checkpoint_file": True,
            "checkpoint_relative_file_path": CHECKPOINT_FILE_NAME,
        }
    )


def get_default_config() -> ConfigDict:
    """Make a default configuration (single det FermiNet on LiH)."""
    config = ConfigDict(
        {
            "problem": get_default_molecular_config(),
            "model": get_default_model_config(),
            "vmc": get_default_vmc_config(),
            "eval": get_default_eval_config(),
            "logdir": os.path.join(
                os.curdir,  # this will be relative to the calling script
                "logs",
            ),
            # if save_to_current_datetime_subfolder=True, will log into a subfolder
            # named according to the datetime at start
            "save_to_current_datetime_subfolder": True,
            "logging_level": "WARNING",
            "dtype": "float32",
            "distribute": True,
            "debug_nans": False,  # If true, OVERRIDES config.distribute to be False
            "initial_seed": 0,
        }
    )
    return config


def choose_model_type_in_model_config(model_config):
    """Given a model config with a specified type, select the specified model.

    The default config contains architecture hyperparameters for several types of models
    (in order to support command-line overwriting via absl.flags), but only one needs to
    be retained after the model type is chosen at the beginning of a run, so this
    function returns a ConfigDict with only the hyperparams associated with the model in
    model_config.type.
    """
    model_type = model_config.type
    model_config = model_config[model_type]
    model_config.type = model_type
    return model_config


def get_default_model_config() -> Dict:
    """Get a default model configuration from a model type."""
    orthogonal_init = {"type": "orthogonal", "scale": 1.0}
    normal_init = {"type": "normal"}

    # tie together the values of ferminet_backflow.cyclic_spins and
    # invariance.cyclic_spins
    cyclic_spins = FieldReference(False)

    input_streams = {
        "include_2e_stream": True,
        "include_ei_norm": True,
        "include_ee_norm": True,
    }

    base_backflow_config = {
        "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_mixed": orthogonal_init.copy(),
        "kernel_init_2e_1e_stream": orthogonal_init.copy(),
        "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
        "bias_init_1e_stream": normal_init.copy(),
        "bias_init_2e_stream": normal_init.copy(),
        "activation_fn": "tanh",
        "use_bias": True,
        "skip_connection": True,
        "cyclic_spins": cyclic_spins,
    }

    ferminet_backflow = {
        "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
        **base_backflow_config,
    }

    determinant_resnet = {
        "ndense": 10,
        "nlayers": 3,
        "activation": "gelu",
        "kernel_init": {"type": "orthogonal", "scale": 2.0},
        "bias_init": normal_init.copy(),
        "use_bias": True,
        "register_kfac": False,
        "mode": "parallel_even",
    }

    base_ferminet_config = {
        "input_streams": input_streams.copy(),
        "backflow": copy.deepcopy(ferminet_backflow),
        "ndeterminants": 1,
        "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_envelope_dim": {"type": "ones"},
        "kernel_init_envelope_ion": {"type": "ones"},
        "bias_init_orbital_linear": normal_init.copy(),
        "orbitals_use_bias": True,
        "isotropic_decay": True,
        "use_det_resnet": False,
        "det_resnet": copy.deepcopy(determinant_resnet),
        "determinant_fn_mode": "parallel_even",
        "full_det": False,
    }

    invariance_for_antieq = {
        "ndense_list": ((32,), (32,), (1,)),
        **base_backflow_config,
    }

    antieq_config = {
        "input_streams": input_streams.copy(),
        "backflow": copy.deepcopy(ferminet_backflow),
        "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_envelope_dim": {"type": "ones"},
        "kernel_init_envelope_ion": {"type": "ones"},
        "bias_init_orbital_linear": normal_init.copy(),
        "orbitals_use_bias": True,
        "isotropic_decay": True,
        "use_products_covariance": True,
        "invariance": copy.deepcopy(invariance_for_antieq),
        "products_covariance": {
            "kernel_init": {"type": "orthogonal", "scale": 2.0},
            "register_kfac": True,
            "use_weights": False,
        },
        "multiply_by_eq_features": False,
    }

    config = {
        "type": "ferminet",
        "ferminet": base_ferminet_config,
        "embedded_particle_ferminet": {
            **base_ferminet_config,
            "nhidden_fermions_per_spin": (2, 2),
            "invariance": {
                "input_streams": input_streams.copy(),
                "backflow": copy.deepcopy(ferminet_backflow.copy()),
                "kernel_initializer": {"type": "orthogonal", "scale": 2.0},
                "bias_initializer": normal_init.copy(),
                "use_bias": True,
                "register_kfac": True,
            },
        },
        "extended_orbital_matrix_ferminet": {
            **base_ferminet_config,
            "nhidden_fermions_per_spin": (2, 2),
            "use_separate_invariance_backflow": False,
            "invariance": {
                "backflow": copy.deepcopy(ferminet_backflow),
                "kernel_initializer": {"type": "orthogonal", "scale": 2.0},
                "bias_initializer": normal_init.copy(),
                "use_bias": True,
                "register_kfac": True,
            },
        },
        # TODO (ggoldsh): these two should probably be subtypes of a single
        # "antiequivariance" model type
        "orbital_cofactor_net": antieq_config,
        "per_particle_dets_net": antieq_config,
        "brute_force_antisym": {
            "input_streams": input_streams.copy(),
            "backflow": copy.deepcopy(ferminet_backflow),
            "antisym_type": "double",
            "ndense_resnet": 64,
            "nlayers_resnet": 2,
            "kernel_init_resnet": {"type": "orthogonal", "scale": 2.0},
            "kernel_init_jastrow": {"type": "ones"},
            "bias_init_resnet": normal_init.copy(),
            "activation_fn_resnet": "tanh",
            "resnet_use_bias": True,
        },
    }
    return config


def get_default_molecular_config() -> Dict:
    """Get a default molecular configuration (LiH)."""
    problem_config = {
        "ion_pos": ((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621)),
        "ion_charges": (1.0, 3.0),
        "nelec": (2, 2),
    }
    return problem_config


def get_default_vmc_config() -> Dict:
    """Get a default VMC training configuration."""
    vmc_config = {
        "nchains": 2000,
        "nepochs": 200000,
        "nburn": 5000,
        "nsteps_per_param_update": 10,
        "nmoves_per_width_update": 100,
        "std_move": 0.25,
        "checkpoint_every": 5000,
        "best_checkpoint_every": 100,
        "checkpoint_dir": "checkpoints",
        "checkpoint_variance_scale": 10,
        "checkpoint_if_nans": False,
        "only_checkpoint_first_nans": True,
        "nhistory_max": 200,
        "record_amplitudes": False,
        "record_param_l1_norm": False,
        "clip_threshold": 5.0,
        "nan_safe": True,
        "optimizer_type": "kfac",
        "optimizer": {
            "kfac": {
                "l2_reg": 0.0,
                "norm_constraint": 0.001,
                "curvature_ema": 0.95,
                "inverse_update_period": 1,
                "min_damping": 1e-4,
                "register_only_generic": False,
                "estimation_mode": "fisher_exact",
                "damping": 0.001,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "adam": {
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "eps_root": 0.0,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "sgd": {
                "momentum": 0.0,
                "nesterov": False,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "sr": {
                "damping": 1.0,  # needs to be tuned with everything else
                "maxiter": 10,  # when maxiter <= -1, uses default 10 * nparams
                "descent_type": "sgd",
                "norm_constraint": 0.001,
                "mode": "lazy",
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,  # needs to be tuned with everything else
                "learning_decay_rate": 1e-4,
            },
        },
    }
    return vmc_config


def get_default_eval_config() -> Dict:
    """Get a default evaluation configuration."""
    eval_config = {
        "nchains": 2000,
        "nburn": 5000,
        "nepochs": 20000,
        "nsteps_per_param_update": 10,
        "nmoves_per_width_update": 100,
        "record_amplitudes": False,
        "std_move": 0.25,
        # if use_data_from_training=True, nchains, nmoves_per_width_update, and
        # std_move are completely ignored, and the data output from training is
        # used as the initial positions instead
        "use_data_from_training": False,
        "record_local_energies": True,  # save local energies and compute statistics
        "nan_safe": False,
    }
    return eval_config
