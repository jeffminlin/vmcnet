"""Test parsing config flags from command line."""
from vmcnet.train.parse_config_flags import parse_flags
import vmcnet.train.default_config as default_config

from absl import flags


def test_parse_default_config(mocker):
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


def test_parse_config_with_model_type_flag(mocker):
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
