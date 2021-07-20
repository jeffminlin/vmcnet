"""Logic for parsing command line flags into a ConfigDict."""

import json
import os
import sys
from typing import Tuple

from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

import vmcnet.train as train

FLAGS = flags.FLAGS


def _get_config_from_reload(
    reload_config: ConfigDict, flag_values: flags.FlagValues
) -> ConfigDict:
    config_path = os.path.join(
        reload_config.log_dir, reload_config.config_relative_file_path
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
    config.model = train.default_config.choose_model_type_in_config(config.model)
    config.lock()
    return config


def parse_flags(flag_values: flags.FlagValues = FLAGS) -> Tuple[ConfigDict, ConfigDict]:
    """Parse command line flags into ConfigDicts.

    Args:
        flag_values (FlagValues): a FlagValues object used to manage the command line
            flags. Can generally be left to its default, but it's useful to be able to
            override this for testing, since an error will be thrown if multiple tests
            define configs for the same FlagValues object. Defaults to global FLAGS.

    Returns:
        (reload_config, config): Two ConfigDicts. The first holds settings for the
            case where configurations or checkpoints are reloaded from a previous run.
            The second holds all other settings.
    """
    config_flags.DEFINE_config_dict(
        "reload_config",
        train.default_config.get_default_reload_config(),
        lock_config=True,
        flag_values=flag_values,
    )
    flag_values(sys.argv, True)
    reload_config = flag_values.reload_config

    if (
        reload_config.use_config_file
        and reload_config.log_dir != train.default_config.NO_RELOAD_LOG_DIR
    ):
        return reload_config, _get_config_from_reload(reload_config, flag_values)
    else:
        return reload_config, _get_config_from_default_config(flag_values)
