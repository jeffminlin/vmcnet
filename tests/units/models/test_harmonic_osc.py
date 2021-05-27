"""Test quantum harmonic osccillator orbitals."""
import jax
import jax.numpy as jnp

import vmcnet.models as models


# TODO(Jeffmin): put in harmonic oscillator integration tests after local energy calcs
# are put in


def _make_input_tree():
    x1 = jnp.array(
        [
            [[1], [2], [3]],
            [[4], [5], [6]],
        ]
    )
    x2 = jnp.array(
        [
            [[7], [8]],
            [[9], [10]],
        ]
    )

    xs = {0: x1, 1: (x2, x1)}

    return xs


def test_harmonic_osc_orbital_shape():
    """Test that putting in a pytree of inputs gives a pytree of orbitals."""
    orbital_model = models.harmonic_osc.HarmonicOscillatorOrbitals(4.0)
    xs = _make_input_tree()

    key = jax.random.PRNGKey(0)

    params = orbital_model.init(key, xs)
    orbitals = orbital_model.apply(params, xs)

    assert orbitals[0].shape == (2, 3, 3)
    assert orbitals[1][0].shape == (2, 2, 2)
    assert orbitals[1][1].shape == orbitals[0].shape
