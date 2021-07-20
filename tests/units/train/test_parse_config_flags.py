"""Test parsing config flags from command line."""
from absl import flags
import os
import pytest

from vmcnet.train.parse_config_flags import parse_flags
import vmcnet.train.default_config as default_config
import vmcnet.utils as utils


def test_parse_default_config(mocker):
    """Test that with no flags, the expected default config is returned."""
    flagValues = flags.FlagValues()

    mocker.patch("sys.argv", ["vmcnet"])
    reload_config, config = parse_flags(flagValues)

    expected_reload_config = default_config.get_default_reload_config()
    expected_config = default_config.get_default_config()
    expected_config.model = default_config.choose_model_type_in_config(
        expected_config.model
    )
    expected_config.lock()

    assert reload_config.to_dict() == expected_reload_config.to_dict()
    assert config.to_dict() == expected_config.to_dict()


def test_parse_config_with_valid_model_type_flag(mocker):
    """Test that a valid model type produces the correct config."""
    flagValues = flags.FlagValues()

    mocker.patch("sys.argv", ["vmcnet", "--config.model.type=orbital_cofactor_net"])
    reload_config, config = parse_flags(flagValues)

    expected_reload_config = default_config.get_default_reload_config()
    expected_config = default_config.get_default_config()
    expected_config.model.type = "orbital_cofactor_net"
    expected_config.model = default_config.choose_model_type_in_config(
        expected_config.model
    )
    expected_config.lock()

    assert reload_config.to_dict() == expected_reload_config.to_dict()
    assert config.to_dict() == expected_config.to_dict()


def test_parse_config_with_invalid_model_type_flag(mocker):
    """Test that invalid model type throws an error."""
    flagValues = flags.FlagValues()

    mocker.patch("sys.argv", ["vmcnet", "--config.model.type=not_a_real_model"])
    with pytest.raises(KeyError):
        parse_flags(flagValues)


def _write_fake_config_json(logdir_path, config_filename):
    fake_config = default_config.get_default_config()
    fake_config.model = default_config.choose_model_type_in_config(fake_config.model)
    fake_config.vmc.nepochs = 20
    fake_config.model.ndeterminants = 3
    fake_config.lock()

    with utils.io.open_or_create(logdir_path, config_filename, "w") as json_writer:
        json_writer.write(fake_config.to_json())

    return fake_config


def test_parse_config_with_reload_log_dir(mocker, tmp_path):
    """Test for expected config when a valid reload log dir is provided."""
    flagValues = flags.FlagValues()

    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)

    expected_config = _write_fake_config_json(logdir_path, "config.json")

    mocker.patch(
        "sys.argv", ["vmcnet", "--reload_config.log_dir={}".format(logdir_path)]
    )

    reload_config, config = parse_flags(flagValues)
    assert expected_config.to_dict() == config.to_dict()


def test_parse_config_with_invalid_reload_param(mocker, tmp_path):
    """Test for expected config when a valid reload log dir is provided."""
    flagValues = flags.FlagValues()

    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)

    _write_fake_config_json(logdir_path, "config.json")

    mocker.patch(
        "sys.argv", ["vmcnet", "--reload_config.log_dir_typo={}".format(logdir_path)]
    )

    with pytest.raises(AttributeError):
        parse_flags(flagValues)


def test_parse_config_with_reload_log_dir_and_override_params(mocker, tmp_path):
    """Test for expected config when reloading config with override flags."""
    flagValues = flags.FlagValues()

    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)

    expected_config = _write_fake_config_json(logdir_path, "config.json")
    expected_config.model.ndeterminants = 5

    mocker.patch(
        "sys.argv",
        [
            "vmcnet",
            "--reload_config.log_dir={}".format(logdir_path),
            "--config.model.ndeterminants=5",
        ],
    )

    reload_config, config = parse_flags(flagValues)
    assert expected_config.to_dict() == config.to_dict()
