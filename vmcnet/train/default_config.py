"""Create configuration of hyperparameters."""

import os
from typing import Dict

from ml_collections import ConfigDict

from vmcnet.utils.checkpoint import DEFAULT_CHECKPOINT_FILE_NAME

NO_NAME = "NONE"
NO_PATH = "NONE"
NO_RELOAD_LOG_DIR = "NONE"
DEFAULT_CONFIG_FILE_NAME = "config.json"
DEFAULT_PRESETS_DIR = "preset_configs"


def _copy_all_dicts(config: Dict) -> Dict:
    """Recursively copy the top-level Dict and all sub-Dicts.

    This ensures that command line flags used to override fields of the config will only
    apply to the intended flag, even if the code used to generate the default values is
    shared among several different configuration flags. For example, the following code
    could be problematic:

    subconfig = {"val": 2}
    config = {
        "sub1": subconfig,
        "sub2": subconfig
    }

    If this config is used directly, setting a command line flag like
    --config.sub1.val=3 will override both config.sub1.val and config.sub2.val, which is
    not the intended behavior. Calling config=copy_all_dicts(config) before turning this
    into a ConfigDict solves the problem by making separate copies of both subconfigs.

    Note: using copy.deepcopy is not a valid replacement for this method, as deepcopy
    will not generate multiple copies of the same object if encountered multiple times.
    In the example above, deepcopy will make a single copy subconfig_copy of subconfig,
    and use it for both sub1 and sub2 in the returned copy. This does not solve the
    problem!
    """
    result = {}
    for key in config:
        if isinstance(config[key], Dict):
            result[key] = _copy_all_dicts(config[key])
        else:
            result[key] = config[key]
    return result


def get_default_reload_config() -> ConfigDict:
    """Make a default reload configuration (no logdir but valid defaults otherwise)."""
    return ConfigDict(
        {
            "logdir": NO_RELOAD_LOG_DIR,
            "use_config_file": True,
            "config_relative_file_path": DEFAULT_CONFIG_FILE_NAME,
            "use_checkpoint_file": True,
            "checkpoint_relative_file_path": DEFAULT_CHECKPOINT_FILE_NAME,
            "new_optimizer_state": False,
            "reburn": False,
            "append": True,
            "same_logdir": False,
        }
    )


def get_default_config() -> ConfigDict:
    """Make a default configuration (single det FermiNet on LiH)."""
    config = ConfigDict(
        _copy_all_dicts(
            {
                "notes": "default",
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
                "subfolder_name": NO_NAME,
                "logging_level": "INFO",
                "dtype": "float32",
                "distribute": False,
                "debug_nans": False,  # If true, OVERRIDES config.distribute to be False
                "initial_seed": 0,
                "wandb": {
                    "mode": "disabled",
                    "project": "default",
                    "name": "vmc-molecule_run",
                    "group": "default",
                },
            }
        )
    )
    config.base_logdir = config.logdir
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

    input_streams = {
        "include_2e_stream": True,
        "include_ei_norm": True,
        "ei_norm_softening": 0.0,
        "include_ee_norm": True,
        "ee_norm_softening": 0.0,
    }

    base_backflow_config = {
        "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_mixed": orthogonal_init,
        "kernel_init_2e_1e_stream": orthogonal_init,
        "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
        "bias_init_1e_stream": normal_init,
        "bias_init_2e_stream": normal_init,
        "activation_fn": "tanh",
        "use_bias": True,
        "one_electron_skip": True,
        "one_electron_skip_scale": 1.0,
        "two_electron_skip": True,
        "two_electron_skip_scale": 1.0,
    }

    ferminet_backflow = {
        "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
        **base_backflow_config,
    }

    base_ferminet_config = {
        "input_streams": input_streams,
        "backflow": ferminet_backflow,
        "ndeterminants": 16,
        "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_envelope_dim": {"type": "ones"},
        "kernel_init_envelope_ion": {"type": "ones"},
        "envelope_softening": 0.0,
        "bias_init_orbital_linear": normal_init,
        "orbitals_use_bias": True,
        "isotropic_decay": True,
        "full_det": True,
        "include_cusp_jastrow": False,
    }

    config = {
        "type": "ferminet",
        "ferminet": base_ferminet_config,
        "explicit_antisym": {
            "input_streams": input_streams,
            "backflow": ferminet_backflow,
            "antisym_type": "generic",  # factorized or generic
            "rank": 1,  # Only relevant for antisym_type=factorized
            "ndense_resnet": 64,
            "nlayers_resnet": 2,
            "kernel_init_resnet": {"type": "orthogonal", "scale": 2.0},
            "bias_init_resnet": normal_init,
            "activation_fn_resnet": "tanh",
            "resnet_use_bias": True,
            "jastrow": {
                # type must be a value in models.jastrow.VALID_JASTROW_TYPES
                "type": "backflow_based",
                "one_body_decay": {"kernel_init": {"type": "ones"}},
                "two_body_decay": {"init_ee_strength": 1.0, "trainable": True},
                "backflow_based": {
                    "use_separate_jastrow_backflow": True,
                    "backflow": {
                        "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
                        "kernel_init_unmixed": orthogonal_init,
                        "kernel_init_mixed": orthogonal_init,
                        "kernel_init_2e_1e_stream": orthogonal_init,
                        "kernel_init_2e_2e_stream": orthogonal_init,
                        "bias_init_1e_stream": normal_init,
                        "bias_init_2e_stream": normal_init,
                        "activation_fn": "gelu",
                        "use_bias": True,
                        "one_electron_skip": True,
                        "one_electron_skip_scale": 1.0,
                        "two_electron_skip": True,
                        "two_electron_skip_scale": 1.0,
                    },
                },
            },
        },
    }
    return config


def get_default_molecular_config() -> Dict:
    """Get a default molecular configuration (LiH)."""
    problem_config = {
        "ion_pos": ((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621)),
        "ion_charges": (1.0, 3.0),
        "nelec": (2, 2),
        "ei_softening": 0.0,
        "ee_softening": 0.0,
    }
    return problem_config


def get_default_vmc_config() -> Dict:
    """Get a default VMC training configuration."""
    vmc_config = {
        "nchains": 1000,
        "nepochs": 200000,
        "nburn": 5000,
        "nsteps_per_param_update": 10,
        "nmoves_per_width_update": 100,
        "std_move": 0.25,
        "checkpoint_every": 5000,
        "best_checkpoint_every": 100,
        "checkpoint_dir": "checkpoints",
        "checkpoint_variance_scale": 10,
        "check_for_nans": False,
        "nhistory_max": 200,
        "record_amplitudes": False,
        "record_param_l1_norm": False,
        "clip_threshold": 5.0,
        "clip_center": "mean",  # mean or median
        "nan_safe": True,
        "optimizer_type": "spring",
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
            "spring": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # SPRING hyperparams
                "mu": 0.99,
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
            },
            "gauss_newton": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 1.0,
                "learning_decay_rate": 1e-4,
                # GN hyperparams
                "E": 0.0,  # target energy
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "clip_threshold": 1000.0,  # GN works best with cusp Jastrow and no clipping
            },
        },
    }
    return vmc_config


def get_default_eval_config() -> Dict:
    """Get a default evaluation configuration."""
    eval_config = {
        "nchains": 1000,
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
