"""Test parsing config flags from command line."""
from absl import flags
import pytest

from vmcnet.train.parse_config_flags import parse_flags
import vmcnet.train.default_config as default_config


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
