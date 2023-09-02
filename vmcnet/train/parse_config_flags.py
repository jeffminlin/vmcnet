"""Logic for parsing command line flags into a ConfigDict."""

import sys
from typing import Tuple

import os
import jax
from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

import vmcnet.train as train
import vmcnet.utils.io as io


def _get_config_from_base(path, flag_values: flags.FlagValues) -> ConfigDict:
    reloaded_config = io.load_config_dict(".", path)
    config_flags.DEFINE_config_dict(
        "config", reloaded_config, lock_config=True, flag_values=flag_values
    )
    flag_values(sys.argv)
    return flag_values.config


def _get_config_from_reload(
    reload_config: ConfigDict, flag_values: flags.FlagValues
) -> ConfigDict:
    reloaded_config = io.load_config_dict(
        reload_config.logdir, reload_config.config_relative_file_path
    )
    config_flags.DEFINE_config_dict(
        "config", reloaded_config, lock_config=True, flag_values=flag_values
    )
    flag_values(sys.argv)
    return flag_values.config


def _get_config_from_default_config(flag_values: flags.FlagValues) -> ConfigDict:
    config_flags.DEFINE_config_dict(
        "config",
        train.default_config.get_default_config(),
        lock_config=False,
        flag_values=flag_values,
    )
    flag_values(sys.argv)
    config = flag_values.config
    config.model = train.default_config.choose_model_type_in_model_config(config.model)
    config.lock()
    return config


def parse_flags(flag_values: flags.FlagValues) -> Tuple[ConfigDict, ConfigDict]:
    """Parse command line flags into ConfigDicts.

    a) with flag --base_config.path=...json

    Load a base config from a json file, then override it with any command line flags


    b) with flag --reload.logdir=...

    In the second, the default ConfigDict is loaded from a previous run by specifying
    the log directory for that run, as well as several other options. These options are
    provided using `--reload..`, for example `--reload.logdir=./logs`.
    The user can still override settings from the previous run by providing regular
    command-line flags via `--config..`, as described above. However, to override
    model-related flags, some care must be taken since the structure of the ConfigDict
    loaded from the json snapshot is not identical to the structure of the default
    ConfigDict. The difference is due to
    :func:`~vmcnet.train.choose_model_type_in_model_config`.


    c) with no --base_config or --reload.logdir

    A global default ConfigDict is made in python, and changed only where the user
    overrides it via command line flags such as `--config.vmc.nburn=100`.


    a and b are mutually exclusive. If more than one is specified, an error is raised.

    Args:
        flag_values (FlagValues): a FlagValues object used to manage the command line
            flags. Should generally use the global flags.FLAGS, but it's useful to be
            able to override this for testing, since an error will be thrown if multiple
            tests define configs for the same FlagValues object.
    Returns:
        (reload_config, config): Two ConfigDicts. The first holds settings for the
            case where configurations or checkpoints are reloaded from a previous run.
            The second holds all other settings.
    """
    config_flags.DEFINE_config_dict(
        "base_config",
        ConfigDict({"path": "NONE"}),
        lock_config=True,
        flag_values=flag_values,
    )
    config_flags.DEFINE_config_dict(
        "reload",
        train.default_config.get_default_reload_config(),
        lock_config=True,
        flag_values=flag_values,
    )
    flag_values(sys.argv, True)
    base_config_path = flag_values.base_config.path
    reload_config = flag_values.reload

    reload = (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_config_file
    )
    load_base_config = base_config_path != "NONE"

    if reload and load_base_config:
        raise ValueError("Cannot specify --base_config.path when using reloaded config")
    if reload:
        config = _get_config_from_reload(reload_config, flag_values)
    elif load_base_config:
        config = _get_config_from_base(base_config_path, flag_values)
    else:
        config = _get_config_from_default_config(flag_values)

    if config.debug_nans:
        config.distribute = False
        jax.config.update("jax_debug_nans", True)

    return reload_config, config
