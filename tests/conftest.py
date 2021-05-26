"""Initial pytest configuration.

Added flags allow for tests to run in an N-device regime, via the flag
--chex_n_cpu_devices=N.
"""
import chex


def pytest_addoption(parser):
    """Provide the --chex_n_cpu_devices arg to pytest."""
    parser.addoption("--chex_n_cpu_devices", type=int, default=4)


def pytest_configure(config):
    """If --chex_n_cpu_devices=N is passed to pytest, run tests on N CPU threads."""
    chex.set_n_cpu_devices(config.getoption("chex_n_cpu_devices"))
