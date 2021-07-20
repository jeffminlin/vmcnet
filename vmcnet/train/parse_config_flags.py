"""Logic for parsing command line flags into a ConfigDict."""

import json
import logging
import os
import sys

from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

import vmcnet.train as train

FLAGS = flags.FLAGS


def get_config_from_reload(reload_config: ConfigDict):
    config_path = os.path.join(
        reload_config.log_dir, reload_config.config_relative_file_path
    )
    with open(config_path) as json_file:
        config_flags.DEFINE_config_dict(
            "config", ConfigDict(json.load(json_file)), lock_config=True
        )
        FLAGS(sys.argv, True)
        return FLAGS.config


def get_config_from_default_config():
    config_flags.DEFINE_config_dict(
        "config", train.default_config.get_default_config(), lock_config=False
    )
    FLAGS(sys.argv, True)
    config = FLAGS.config
    config.model = train.default_config.choose_model_type_in_config(config.model)
    config.lock()
    return config


def parse_flags():
    config_flags.DEFINE_config_dict(
        "reload_config",
        train.default_config.get_default_reload_config(),
        lock_config=True,
    )
    FLAGS(sys.argv, True)
    reload_config = FLAGS.reload_config
    logging.info("Reload configuration: \n%s", reload_config)

    if (
        reload_config.use_config_file
        and reload_config.log_dir != train.default_config.NO_RELOAD_LOG_DIR
    ):
        return reload_config, get_config_from_reload(reload_config)
    else:
        return reload_config, get_config_from_default_config()
