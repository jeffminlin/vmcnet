"""Initial pytest configuration.

On cpu, tests can run in an N-device regime, via the flag --chex_n_cpu_devices=N.

Following the example in https://docs.pytest.org/en/latest/example/simple.html,
tests marked with pytest.mark.slow will be skipped unless the --runslow option is
provided.
"""
import chex
import pytest


# Markers to apply to tests to skip them in ascending level of slow-ness (how slow and
# which ones are the slowest will probably be a bit machine dependent)
SKIP_LEVELS = (
    (
        "slow",
        {"help": "run slow tests", "description": "mark test as slow to run"},
    ),
    (
        "very_slow",
        {"help": "run very slow tests", "description": "mark test as very slow to run"},
    ),
)


def _make_flag(marker):
    return "--run_{}".format(marker)


def pytest_addoption(parser):
    """Provide the --chex_n_cpu_devices arg to pytest."""
    parser.addoption("--chex_n_cpu_devices", type=int, default=4)
    for marker, info in SKIP_LEVELS:
        parser.addoption(
            _make_flag(marker),
            action="store_true",
            default=False,
            help=info["help"],
        )


def pytest_configure(config):
    """If --chex_n_cpu_devices=N is passed to pytest, run tests on N CPU threads."""
    chex.set_n_cpu_devices(config.getoption("chex_n_cpu_devices"))
    for marker, info in SKIP_LEVELS:
        config.addinivalue_line("markers", "{}: {}".format(marker, info["description"]))


def pytest_collection_modifyitems(config, items):
    """Modify pytest collection to respect the --runslow flag."""
    for marker, _ in reversed(SKIP_LEVELS):
        # iterate through the levels in reverse, mark tests to skip until we hit a flag
        if config.getoption(_make_flag(marker)):
            return
        skip_marker = pytest.mark.skip(
            reason="need {} option to run".format(_make_flag(marker))
        )
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip_marker)
