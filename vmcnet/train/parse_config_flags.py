"""Logic for parsing command line flags into a ConfigDict."""

import sys
import os
from typing import Tuple

import jax
from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

import vmcnet.train as train
import vmcnet.utils.io as io
from vmcnet.train.default_config import NO_NAME, NO_PATH, DEFAULT_PRESETS_DIR


def _get_config_from_reload(
    reload_config: ConfigDict, flag_values: flags.FlagValues
) -> ConfigDict:
    reloaded_config = io.load_config_dict(
        reload_config.logdir, reload_config.config_relative_file_path
    )
    reloaded_config.logdir = reloaded_config.base_logdir
    config_flags.DEFINE_config_dict(
        "config", reloaded_config, lock_config=True, flag_values=flag_values
    )
    flag_values(sys.argv)
    return flag_values.config


def _get_config_from_default_config(
    flag_values: flags.FlagValues, presets_path=None
) -> ConfigDict:
    base_config = train.default_config.get_default_config()

    if presets_path is not None:
        presets = io.load_config_dict("", presets_path)
        base_config.update(presets)

    config_flags.DEFINE_config_dict(
        "config",
        base_config,
        lock_config=False,
        flag_values=flag_values,
    )
    flag_values(sys.argv)
    config = flag_values.config
    config.model = train.default_config.choose_model_type_in_model_config(config.model)
    return config


def parse_flags(flag_values: flags.FlagValues) -> Tuple[ConfigDict, ConfigDict]:
    """Parse command line flags into ConfigDicts.

    a) with flag --presets.path=my_preset.json
    or --presets.name=my_preset, if PRESETS_DIR/my_preset.json exists

    Update default config with a preset config from a json file, then override it with
    any command line flags.


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


    c) with no --presets or --reload.logdir

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
        "presets",
        ConfigDict({"name": NO_NAME, "path": NO_PATH}),
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

    reload_config = flag_values.reload

    reload = (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_config_file
    )

    load_presets = (
        flag_values.presets.name != NO_NAME or flag_values.presets.path != NO_PATH
    )
    if flag_values.presets.path != NO_PATH and flag_values.presets.name != NO_NAME:
        raise ValueError("Cannot specify both --presets.path and --presets.name")

    if flag_values.presets.name != NO_NAME:
        presets_path = os.path.join(
            DEFAULT_PRESETS_DIR, flag_values.presets.name + ".json"
        )
    elif flag_values.presets.path != NO_PATH:
        presets_path = flag_values.presets.path

    if reload and load_presets:
        raise ValueError("Cannot specify --presets.path when using reloaded config")
    if reload:
        config = _get_config_from_reload(reload_config, flag_values)
    elif load_presets:
        config = _get_config_from_default_config(flag_values, presets_path)
    else:
        config = _get_config_from_default_config(flag_values)

    if config.debug_nans:
        config.distribute = False
        jax.config.update("jax_debug_nans", True)

    config.base_logdir = config.logdir
    config.lock()
    return reload_config, config
