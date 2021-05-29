"""Test kinetic energy functions."""
import vmcnet.physics as physics

from .test_core import _make_dummy_log_f, _make_dummy_x


def test_kinetic_energy_shape():
    """Check that the vmapped shape is correct."""
    _, log_f = _make_dummy_log_f()
    x = _make_dummy_x()

    kinetic_energy_fn = physics.kinetic.create_continuous_kinetic_energy(log_f)
    kinetic_energies = kinetic_energy_fn(None, x)

    assert kinetic_energies.shape == (x.shape[0],)
