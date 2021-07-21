"""Create configuration of hyperparameters."""
import os

from ml_collections import ConfigDict, FieldReference


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
            "initial_seed": 0,
        }
    )
    return config


def choose_model_type_in_config(model_config):
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


def get_default_model_config() -> ConfigDict:
    """Get a default model configuration from a model type."""
    orthogonal_init = ConfigDict({"type": "orthogonal", "scale": 1.0})
    normal_init = ConfigDict({"type": "normal"})

    # tie together the values of ferminet_backflow.cyclic_spins and
    # invariance.cyclic_spins
    cyclic_spins = FieldReference(False)

    ferminet_backflow = ConfigDict(
        {
            "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
            "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
            "kernel_init_mixed": orthogonal_init,
            "kernel_init_2e_1e_stream": orthogonal_init,
            "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
            "bias_init_1e_stream": normal_init,
            "bias_init_2e_stream": normal_init,
            "activation_fn": "tanh",
            "include_2e_stream": True,
            "include_ei_norm": True,
            "include_ee_norm": True,
            "use_bias": True,
            "skip_connection": True,
            "cyclic_spins": cyclic_spins,
        }
    )
    invariance = ConfigDict(
        {
            "ndense_list": ((256,), (256,), (1,)),
            "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
            "kernel_init_mixed": orthogonal_init,
            "kernel_init_2e_1e_stream": orthogonal_init,
            "kernel_init_2e_2e_stream": {
                "type": "orthogonal",
                "scale": 2.0,
            },
            "bias_init_1e_stream": normal_init,
            "bias_init_2e_stream": normal_init,
            "activation_fn": "tanh",
            "include_2e_stream": False,
            "include_ei_norm": False,
            "include_ee_norm": False,
            "use_bias": True,
            "skip_connection": True,
            "cyclic_spins": cyclic_spins,
        }
    )
    config = ConfigDict(
        {
            "type": "ferminet",
            "ferminet": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "ndeterminants": 1,
                    "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_envelope_dim": {"type": "ones"},
                    "kernel_init_envelope_ion": {"type": "ones"},
                    "bias_init_orbital_linear": normal_init,
                    "orbitals_use_bias": True,
                    "isotropic_decay": True,
                    "use_det_resnet": False,
                    "det_resnet": ConfigDict(
                        {
                            "ndense": 10,
                            "nlayers": 3,
                            "activation": "gelu",
                            "kernel_init": {"type": "orthogonal", "scale": 2.0},
                            "bias_init": normal_init,
                            "use_bias": True,
                            "register_kfac": False,
                        }
                    ),
                }
            ),
            "orbital_cofactor_net": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_envelope_dim": {"type": "ones"},
                    "kernel_init_envelope_ion": {"type": "ones"},
                    "bias_init_orbital_linear": normal_init,
                    "orbitals_use_bias": True,
                    "isotropic_decay": True,
                    "invariance": invariance,
                }
            ),
            "per_particle_dets_net": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_envelope_dim": {"type": "ones"},
                    "kernel_init_envelope_ion": {"type": "ones"},
                    "bias_init_orbital_linear": normal_init,
                    "orbitals_use_bias": True,
                    "isotropic_decay": True,
                    "invariance": invariance,
                }
            ),
            "brute_force_antisym": ConfigDict(
                {
                    "backflow": ferminet_backflow,
                    "antisym_type": "double",
                    "ndense_resnet": 64,
                    "nlayers_resnet": 2,
                    "kernel_init_resnet": {"type": "orthogonal", "scale": 2.0},
                    "kernel_init_jastrow": {"type": "ones"},
                    "bias_init_resnet": normal_init,
                    "activation_fn_resnet": "tanh",
                    "resnet_use_bias": True,
                }
            ),
        }
    )
    return config


def get_default_molecular_config() -> ConfigDict:
    """Get a default molecular configuration (LiH)."""
    problem_config = ConfigDict(
        {
            "ion_pos": ((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621)),
            "ion_charges": (1.0, 3.0),
            "nelec": (2, 2),
        }
    )
    return problem_config


def get_default_vmc_config() -> ConfigDict:
    """Get a default VMC training configuration."""
    vmc_config = ConfigDict(
        {
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
            "nhistory_max": 200,
            "record_param_l1_norm": False,
            "clip_threshold": 5.0,
            "schedule_type": "inverse_time",  # constant or inverse_time
            "learning_rate": 1e-4,
            "learning_decay_rate": 1e-4,
            "nan_safe": True,
            "optimizer_type": "kfac",
            "optimizer": {
                "kfac": ConfigDict(
                    {
                        "l2_reg": 0.0,
                        "norm_constraint": 0.001,
                        "curvature_ema": 0.95,
                        "inverse_update_period": 1,
                        "min_damping": 1e-4,
                        "register_only_generic": False,
                        "estimation_mode": "fisher_exact",
                        "damping": 0.001,
                    }
                ),
                "adam": {"b1": 0.9, "b2": 0.999, "eps": 1e-8, "eps_root": 0.0},
            },
        }
    )
    return vmc_config


def get_default_eval_config() -> ConfigDict:
    """Get a default evaluation configuration."""
    eval_config = ConfigDict(
        {
            "nchains": 2000,
            "nburn": 5000,
            "nepochs": 20000,
            "nsteps_per_param_update": 10,
            "nmoves_per_width_update": 100,
            "std_move": 0.25,
            # if use_data_from_training=True, nchains, nmoves_per_width_update, and
            # std_move are completely ignored, and the data output from training is
            # used as the initial positions instead
            "use_data_from_training": False,
            "nan_safe": False,
        }
    )
    return eval_config
