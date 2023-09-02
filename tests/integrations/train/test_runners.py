"""Test of train.runners with near-default configs."""
import os

import jax
import numpy as np
import pytest

import vmcnet.train as train
from ml_collections import ConfigDict


def _get_config(vmc_nchains, eval_nchains, distribute):
    config = train.default_config.get_default_config()
    config.vmc.nchains = vmc_nchains
    config.vmc.nepochs = 5
    config.vmc.nburn = 2
    config.vmc.checkpoint_every = 2
    config.vmc.best_checkpoint_every = 4
    config.vmc.nsteps_per_param_update = 3

    config.eval.nchains = eval_nchains
    config.eval.nepochs = 3
    config.eval.nburn = 5
    config.eval.nsteps_per_param_update = 4

    config.distribute = distribute
    return config


def _check_length_and_finiteness_of_metrics(nepochs, inner_logdir, metric_files):
    for metric_file in metric_files:
        assert (inner_logdir / metric_file).exists()
        with (inner_logdir / metric_file).open() as f:
            metric = np.loadtxt(f)
        assert len(metric) == nepochs
        assert np.all(np.isfinite(metric))


def _run_and_check_output_files(mocker, tmp_path, config):
    """Mock flags, run the molecular runner, and check the resulting output files.

    Unfortunately, mocking sys.argv (which would be slightly more end-to-end) doesn't
    quite work due to some quirks with how flags are registered when not running from
    the command-line, and we'd generally like to avoid using subprocess.run(...) if it's
    not necessary. Thus here we directly mock FLAGS to have the desired config, then
    directly call the runner.
    """
    mock_flags = mocker.patch("vmcnet.train.runners.FLAGS")
    mock_flags.config = config
    mock_flags.reload = train.default_config.get_default_reload_config()
    mock_flags.base_config = ConfigDict({"path": "base_configs/quicktest.json"})

    train.runners.run_molecule()

    parent_logdir = tmp_path / "logs"
    assert parent_logdir.exists()

    datetime_dirs = list(parent_logdir.iterdir())
    assert len(datetime_dirs) == 1  # one datetime directory since we're using tmp_path

    inner_logdir = datetime_dirs[0]

    # Check that the config is saved
    config_file_name = train.default_config.DEFAULT_CONFIG_FILE_NAME
    assert (inner_logdir / config_file_name).exists()
    config.logdir = os.path.normpath(inner_logdir)
    desired_config_json_str = config.to_json(indent=4)
    assert (inner_logdir / config_file_name).read_text() == desired_config_json_str

    # Check that there are nepochs finite metrics being saved
    vmc_metric_files = [
        "accept_ratio.txt",
        "energy.txt",
        "energy_noclip.txt",
        "variance.txt",
        "variance_noclip.txt",
    ]
    _check_length_and_finiteness_of_metrics(
        config.vmc.nepochs, inner_logdir, vmc_metric_files
    )

    # Check that regular and best checkpoints are being saved
    checkpoint_dir = inner_logdir / "checkpoints"
    num_regular_checkpoints = config.vmc.nepochs // config.vmc.checkpoint_every
    assert checkpoint_dir.exists()
    assert set(checkpoint_dir.iterdir()) == set(
        [
            checkpoint_dir / (str((i + 1) * config.vmc.checkpoint_every) + ".npz")
            for i in range(num_regular_checkpoints)
        ]
    )
    assert (inner_logdir / "checkpoint.npz").exists()

    # Check that evaluation is being done and metrics are being saved
    assert (inner_logdir / "eval").exists()
    assert (inner_logdir / "eval" / "statistics.json").exists()
    eval_metric_files = [
        "accept_ratio.txt",
        "energy.txt",
        "variance.txt",
        "local_energies.txt",
    ]
    _check_length_and_finiteness_of_metrics(
        config.eval.nepochs, inner_logdir / "eval", eval_metric_files
    )
    # specially check that shape of local_energies is correct
    local_es = np.loadtxt(os.path.join(inner_logdir, "eval", "local_energies.txt"))
    assert local_es.shape == (config.eval.nepochs, config.eval.nchains)


@pytest.mark.very_slow
def test_run_molecule_pmapped(mocker, tmp_path):
    """End-to-end test of the molecular runner (with smaller nchains/nepochs).

    This test mostly exists to check that no top-level errors are raised during the call
    to the runner with default configs, and that there is some potentially reasonable
    logging occurring. It will not generally catch more subtle bugs.
    """
    vmc_nchains = 3 * jax.local_device_count()
    eval_nchains = 2 * jax.local_device_count()
    mocker.patch("os.curdir", tmp_path)
    config = _get_config(vmc_nchains, eval_nchains, True)

    _run_and_check_output_files(mocker, tmp_path, config)


@pytest.mark.very_slow
def test_run_molecule_jitted(mocker, tmp_path):
    """End-to-end test of the molecular runner, but only jitted."""
    vmc_nchains = 5  # use a prime number here to catch if pmapping is trying to happen
    eval_nchains = 3
    mocker.patch("os.curdir", tmp_path)
    config = _get_config(vmc_nchains, eval_nchains, False)

    _run_and_check_output_files(mocker, tmp_path, config)
