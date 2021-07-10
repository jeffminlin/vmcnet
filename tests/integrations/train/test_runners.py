"""Test of train.runners with near-default configs."""
import os

import numpy as np

import vmcnet.train as train


def _check_length_and_finiteness_of_metrics(nepochs, inner_logdir, metric_files):
    for metric_file in metric_files:
        assert (inner_logdir / metric_file).exists()
        with (inner_logdir / metric_file).open() as f:
            metric = np.loadtxt(f)
        assert len(metric) == nepochs
        assert np.all(np.isfinite(metric))


def test_run_molecule(mocker, tmp_path):
    """End-to-end test of the molecular runner (with smaller nchains/nepochs).

    Unfortunately, mocking sys.argv (which would be slightly more end-to-end) doesn't
    quite work due to some quirks with how flags are registered when not running from
    the command-line, and we'd generally like to avoid using subprocess.run(...) if it's
    not necessary. Thus here we directly mock FLAGS to have the desired config, then
    directly call the runner.

    This test mostly exists to check that no top-level errors are raised during the call
    to the runner with default configs, and that there is some potentially reasonable
    logging occurring. It will not generally catch more subtle bugs.
    """
    vmc_nchains = 10
    vmc_nepochs = 5
    vmc_checkpoint_every = 2
    vmc_best_checkpoint_every = 4

    eval_nchains = 20
    eval_nepochs = 3

    mocker.patch("os.curdir", tmp_path)

    config = train.default_config.get_default_config()
    config.vmc.nchains = vmc_nchains
    config.vmc.nepochs = vmc_nepochs
    config.vmc.checkpoint_every = vmc_checkpoint_every
    config.vmc.best_checkpoint_every = vmc_best_checkpoint_every
    config.eval.nchains = eval_nchains
    config.eval.nepochs = eval_nepochs

    mock_flags = mocker.patch("vmcnet.train.runners.FLAGS")
    mock_flags.config = config
    train.runners.run_molecule()

    parent_logdir = tmp_path / "logs"
    assert parent_logdir.exists()

    datetime_dirs = list(parent_logdir.iterdir())
    assert len(datetime_dirs) == 1  # one datetime directory since we're using tmp_path

    inner_logdir = datetime_dirs[0]

    # Check that the config is saved
    assert (inner_logdir / "config.json").exists()
    config.logdir = os.path.normpath(inner_logdir)
    desired_config_json_str = config.to_json(indent=4)
    assert (inner_logdir / "config.json").read_text() == desired_config_json_str

    # Check that there are nepochs finite metrics being saved
    vmc_metric_files = [
        "accept_ratio.txt",
        "energy.txt",
        "energy_noclip.txt",
        "variance.txt",
        "variance_noclip.txt",
    ]
    _check_length_and_finiteness_of_metrics(vmc_nepochs, inner_logdir, vmc_metric_files)

    # Check that regular and best checkpoints are being saved
    checkpoint_dir = inner_logdir / "checkpoints"
    num_regular_checkpoints = vmc_nepochs // vmc_checkpoint_every
    assert checkpoint_dir.exists()
    assert set(checkpoint_dir.iterdir()) == set(
        [
            checkpoint_dir / (str((i + 1) * vmc_checkpoint_every) + ".npz")
            for i in range(num_regular_checkpoints)
        ]
    )
    assert (inner_logdir / "checkpoint.npz").exists()

    # Check that evaluation is being done and metrics are being saved
    assert (inner_logdir / "eval").exists()
    eval_metric_files = ["accept_ratio.txt", "energy.txt", "variance.txt"]
    _check_length_and_finiteness_of_metrics(
        eval_nepochs, inner_logdir / "eval", eval_metric_files
    )
