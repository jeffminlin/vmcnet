"""Test parsing config flags from command line."""
from absl import flags
from ml_collections import ConfigDict
import os
import pytest

from vmcnet.train.parse_config_flags import parse_flags
import vmcnet.train.default_config as default_config
import vmcnet.utils as utils
from tests.test_utils import get_default_config_with_chosen_model


def _assert_configs_equal(config1: ConfigDict, config2: ConfigDict):
    assert config1.to_dict() == config2.to_dict()


def test_parse_default_config(mocker):
    """Test that with no flags, the expected default config is returned."""
    flag_values = flags.FlagValues()
    mocker.patch("sys.argv", ["vmcnet"])

    reload_config, config = parse_flags(flag_values)

    expected_reload_config = default_config.get_default_reload_config()
    expected_config = get_default_config_with_chosen_model("ferminet")

    _assert_configs_equal(reload_config, expected_reload_config)
    _assert_configs_equal(config, expected_config)


def test_parse_config_with_invalid_flag(mocker):
    """Test that invalid model type throws an error."""
    flag_values = flags.FlagValues()
    mocker.patch("sys.argv", ["vmcnet", "--config.model.type=not_a_real_model"])

    with pytest.raises(KeyError):
        parse_flags(flag_values)


def test_parse_config_with_invalid_top_level_flag(mocker):
    """Test that invalid top level flag throws an error."""
    flag_values = flags.FlagValues()
    mocker.patch("sys.argv", ["vmcnet", "--not_a_config.has_truth=True"])

    with pytest.raises(flags.UnrecognizedFlagError):
        parse_flags(flag_values)


def test_parse_config_with_valid_flags_including_tuple(mocker):
    """Test that valid model flags produce the correct config."""
    flag_values = flags.FlagValues()
    mocker.patch(
        "sys.argv",
        [
            "vmcnet",
            "--config.model.type=orbital_cofactor_net",
            "--config.model.orbital_cofactor_net.isotropic_decay=False",
            # On the command line setting a tuple value requires extra quotes like
            # '((0.,0.,0.),)', but in testing the extra quotes cause a type error.
            "--config.problem.ion_pos=((0.,0.,0.),)",
        ],
    )
    expected_config = get_default_config_with_chosen_model("orbital_cofactor_net")
    expected_config.model.isotropic_decay = False
    expected_config.problem.ion_pos = ((0.0, 0.0, 0.0),)

    _, config = parse_flags(flag_values)

    _assert_configs_equal(config, expected_config)


def test_setting_duplicated_config_flag_sets_only_desired_flag(mocker):
    """Test changing normal_init.type only affects the desired instance of the flag.

    This is a regression test as reuse of the same Dict object used to mean that setting
    any instance of normal_init.type would change all other instances too.
    """
    flag_values = flags.FlagValues()
    mocker.patch(
        "sys.argv",
        ["vmcnet", "--config.model.ferminet.bias_init_orbital_linear.type=CHANGED"],
    )
    _, config = parse_flags(flag_values)

    assert config.model.bias_init_orbital_linear.type == "CHANGED"
    assert config.model.backflow.bias_init_1e_stream.type == "normal"


def test_parse_config_with_invalid_reload_param(mocker):
    """Verify that error is thrown when an invalid reload flag is passed."""
    flag_values = flags.FlagValues()
    mocker.patch("sys.argv", ["vmcnet", "--reload.logdir_typo=log_dir_path"])

    with pytest.raises(AttributeError):
        parse_flags(flag_values)


def _write_fake_config_json(logdir_path: str, config_filename: str) -> ConfigDict:
    fake_config = get_default_config_with_chosen_model("ferminet")
    fake_config.vmc.nepochs = 20
    fake_config.model.ndeterminants = 3

    utils.io.save_config_dict_to_json(fake_config, logdir_path, config_filename)
    return fake_config


def test_parse_config_with_valid_reload_log_dir(mocker, tmp_path):
    """Test for expected config when a valid reload log dir is provided."""
    flag_values = flags.FlagValues()
    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)
    mocker.patch("sys.argv", ["vmcnet", "--reload.logdir={}".format(logdir_path)])
    expected_config = _write_fake_config_json(logdir_path, "config")

    _, config = parse_flags(flag_values)

    _assert_configs_equal(config, expected_config)


def test_parse_config_with_reload_log_dir_and_override_params(mocker, tmp_path):
    """Test for expected config when reloading config with override flags."""
    flag_values = flags.FlagValues()
    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)
    mocker.patch(
        "sys.argv",
        [
            "vmcnet",
            "--reload.logdir={}".format(logdir_path),
            "--config.model.ndeterminants=5",
            "--config.vmc.nburn=100000",
        ],
    )
    expected_config = _write_fake_config_json(logdir_path, "config")
    expected_config.model.ndeterminants = 5
    expected_config.vmc.nburn = 100000

    _, config = parse_flags(flag_values)

    _assert_configs_equal(config, expected_config)


def test_parse_config_with_use_config_file_false(mocker, tmp_path):
    """Test that parser does not try to load from file when use_config_file=False ."""
    flag_values = flags.FlagValues()
    logdir_name = "logs"
    logdir_path = os.path.join(tmp_path, logdir_name)
    mocker.patch(
        "sys.argv",
        [
            "vmcnet",
            "--reload.logdir={}".format(logdir_path),
            "--reload.use_config_file=False",
            "--config.vmc.nburn=100000",
        ],
    )
    expected_config = get_default_config_with_chosen_model("ferminet")
    expected_config.vmc.nburn = 100000

    _, config = parse_flags(flag_values)

    _assert_configs_equal(config, expected_config)
