"""Logic for parsing command line flags into a ConfigDict."""

import json
import os
import sys
from typing import Tuple

from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

import vmcnet.train as train


def _get_config_from_reload(
    reload_config: ConfigDict, flag_values: flags.FlagValues
) -> ConfigDict:
    config_path = os.path.join(
        reload_config.logdir, reload_config.config_relative_file_path
    )
    with open(config_path) as json_file:
        config_flags.DEFINE_config_dict(
            "config",
            ConfigDict(json.load(json_file)),
            lock_config=True,
            flag_values=flag_values,
        )
        flag_values(sys.argv, True)
        return flag_values.config


def _get_config_from_default_config(flag_values: flags.FlagValues) -> ConfigDict:
    config_flags.DEFINE_config_dict(
        "config",
        train.default_config.get_default_config(),
        lock_config=False,
        flag_values=flag_values,
    )
    flag_values(sys.argv, True)
    config = flag_values.config
    config.model = train.default_config.choose_model_type_in_model_config(config.model)
    config.lock()
    return config


def parse_flags(flag_values: flags.FlagValues) -> Tuple[ConfigDict, ConfigDict]:
    """Parse command line flags into ConfigDicts.

    Supports two cases. In the first, a global default ConfigDict is used as a
    starting point, and changed only where the user overrides it via command line
    flags such as `--config.vmc.nburn=100`.

    In the second, the default ConfigDict is loaded from a previous run by specifying
    the log directory for that run, as well as several other options. These options are
    provided using `--reload..`, for example `--reload.logdir=./logs`.
    The user can still override settings from the previous run by providing regular
    command-line flags via `--config..`, as described above. However, to override
    model-related flags, some care must be taken since the structure of the ConfigDict
    loaded from the json snapshot is not identical to the structure of the default
    ConfigDict. The difference is due to
    :func:`~vmcnet.train.choose_model_type_in_model_config`.

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
        "reload",
        train.default_config.get_default_reload_config(),
        lock_config=True,
        flag_values=flag_values,
    )
    flag_values(sys.argv, True)
    reload_config = flag_values.reload_config

    if (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_config_file
    ):
        return reload_config, _get_config_from_reload(reload_config, flag_values)
    else:
        return reload_config, _get_config_from_default_config(flag_values)
