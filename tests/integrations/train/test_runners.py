"""Test of train.runners with near-default configs."""
import os

import jax
import numpy as np
import pytest

import vmcnet.train as train
from ml_collections import ConfigDict
from vmcnet.utils import io
from vmcnet.utils.pytree_helpers import tree_dist

from absl import flags


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
    mock_flags.presets = ConfigDict(
        {"path": train.default_config.NO_PATH, "name": train.default_config.NO_NAME}
    )

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
    assert (inner_logdir / "best_checkpoint.npz").exists()

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
    config.vmc.optimizer_type = "kfac"  # Multi-device SPRING not yet implemented

    _run_and_check_output_files(mocker, tmp_path, config)


@pytest.mark.very_slow
def test_run_molecule_jitted(mocker, tmp_path):
    """End-to-end test of the molecular runner, but only jitted."""
    vmc_nchains = 5  # use a prime number here to catch if pmapping is trying to happen
    eval_nchains = 3
    mocker.patch("os.curdir", tmp_path)
    config = _get_config(vmc_nchains, eval_nchains, False)

    _run_and_check_output_files(mocker, tmp_path, config)


@pytest.mark.very_slow
def test_run_molecule_double_precision(mocker, tmp_path):
    """End-to-end test of the molecular runner, in double precision."""
    vmc_nchains = 3 * jax.local_device_count()
    eval_nchains = 2 * jax.local_device_count()
    mocker.patch("os.curdir", tmp_path)
    config = _get_config(vmc_nchains, eval_nchains, False)
    config.dtype = "float64"

    _run_and_check_output_files(mocker, tmp_path, config)


@pytest.mark.very_slow
def test_reload_append(mocker, tmp_path):
    """Reload and continue a run from a checkpoint.

    This runs an example for 10 epochs as run_1. It then reloads
    from the checkpoint at 5 epochs and re-runs the last epochs
    until 10 epochs total is reached again.
    The weights and energy histories from the two runs are compared.
    """
    mocker.patch("os.curdir", tmp_path)
    path1 = (tmp_path / "run_1").as_posix()
    path2 = (tmp_path / "run_2").as_posix()

    start_argv = [
        "vmc-molecule",
        "--presets.name=quicktest",
        "--config.vmc.nchains=" + str(2 * jax.local_device_count()),
        "--config.vmc.nburn=10",
        "--config.vmc.nepochs=10",
        "--config.eval.nburn=0",
        "--config.eval.nepochs=0",
        "--config.vmc.checkpoint_every=1",
        "--config.save_to_current_datetime_subfolder=False",
        "--config.subfolder_name=NONE",
        "--config.vmc.optimizer_type=spring",
        "--config.distribute=False",
    ]
    reload_argv = [
        "vmc-molecule",
        "--reload.logdir=" + path1,
        "--reload.append=True",
        "--reload.checkpoint_relative_file_path=checkpoints/5.npz",
    ]
    mock_argv1 = start_argv + ["initial_seed=0", "--config.logdir=" + path1]
    mock_argv2 = reload_argv + ["--config.logdir=" + path2]

    mocker.patch("sys.argv", mock_argv1)
    train.runners.run_molecule()
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)

    mocker.patch("sys.argv", mock_argv2)
    train.runners.run_molecule()
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)

    states1 = dict()
    states2 = dict()
    for f in sorted(
        os.listdir(path1 + "/checkpoints"), key=lambda x: int(x.split(".")[0])
    ):
        states1[f] = io.reload_vmc_state(path1 + "/checkpoints", f)
    for f in sorted(
        os.listdir(path2 + "/checkpoints"), key=lambda x: int(x.split(".")[0])
    ):
        states2[f] = io.reload_vmc_state(path2 + "/checkpoints", f)

    common = [f for f in states1.keys() if f in states2.keys()]

    with open(path1 + "/energy.txt", "r") as f:
        energies1 = np.array(f.readlines(), dtype=float)
    with open(path2 + "/energy.txt", "r") as f:
        energies2 = np.array(f.readlines(), dtype=float)

    all_dists = [
        [[tree_dist(states1[i][k], states2[j][k]) for j in common] for i in common]
        for k in range(5)
    ]
    names = ("epoch", "data", "old_params", "optimizer_state", "key")

    eps = 1e-6
    for k, name in enumerate(names):
        distmatrix = all_dists[k]
        assert np.mean(np.diag(distmatrix)) < eps * np.mean(distmatrix)
        print("test passed for", name)

    np.testing.assert_allclose(energies1, energies2, rtol=1e-6)
