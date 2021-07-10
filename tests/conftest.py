"""Initial pytest configuration.

On cpu, tests can run in an N-device regime, via the flag --chex_n_cpu_devices=N.

Following the example in https://docs.pytest.org/en/latest/example/simple.html,
tests marked with pytest.mark.slow will be skipped unless the --runslow option is
provided.
"""
import chex
import pytest


def pytest_addoption(parser):
    """Provide the --chex_n_cpu_devices arg to pytest."""
    parser.addoption("--chex_n_cpu_devices", type=int, default=4)
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """If --chex_n_cpu_devices=N is passed to pytest, run tests on N CPU threads."""
    chex.set_n_cpu_devices(config.getoption("chex_n_cpu_devices"))
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Modify pytest collection to respect the --runslow flag."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
