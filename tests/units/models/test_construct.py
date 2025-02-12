"""Test model construction."""

import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmcnet.models as models

from tests.test_utils import get_default_config_with_chosen_model


def test_compose_negative_with_three_body_antisymmetry():
    """Test a simple three-body antisymmetry composed with the *(-1) equivariance."""

    def equivariance(xs):
        return jax.tree_map(lambda a: -a, xs)

    def three_body_antisymmetry(x):
        return (
            (x[..., 0] - x[..., 1]) * (x[..., 0] - x[..., 2]) * (x[..., 1] - x[..., 2])
        )

    def product_antisymmetry(xs):
        antisymmetry_tree = jax.tree_map(three_body_antisymmetry, xs)
        return functools.reduce(
            lambda a, b: a * b, jax.tree_util.tree_leaves(antisymmetry_tree)
        )

    model = models.core.ComposedModel([equivariance, product_antisymmetry])

    x = [
        jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),  # antisymmetry gives [-2, -2]
        jnp.array([[1, 3, 5], [7, 10, 13]]),  # antisymmetry gives [-16, -54]
        jnp.array([[1, 0, -1, 6], [-1, 0, 4, 2]]),  # antisymmetry gives [2, -20]
    ]

    output = model.apply({}, x)

    np.testing.assert_allclose(output, jnp.array([2 * 16 * -2, 2 * 54 * 20]))


def _get_initial_pos_and_hyperparams():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    ion_pos = jax.random.normal(subkey, (5, 3))
    ion_charges = jnp.array([1.0, 2.0, 1.0, 3.0, 5.0])

    key, subkey = jax.random.split(key)
    init_pos = jax.random.normal(subkey, shape=(7, 6, 3))

    spin_split = (2,)  # 2 up, 4 down
    ndense_list = ((6, 3), (6, 3), (3, 3), (3, 4), (12,), (11, 3), (9,))

    return key, ion_pos, ion_charges, init_pos, spin_split, ndense_list


def _get_compute_input_streams(ion_pos):
    return functools.partial(models.equivariance.compute_input_streams, ion_pos=ion_pos)


def _get_backflow(spin_split, ndense_list, cyclic_spins):
    residual_blocks = models.construct.get_residual_blocks_for_ferminet_backflow(
        spin_split,
        ndense_list,
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_kernel_initializer("xavier_normal"),
        models.weights.get_kernel_initializer("glorot_uniform"),
        models.weights.get_bias_initializer("zeros"),
        models.weights.get_bias_initializer("zeros"),
        jnp.tanh,
        cyclic_spins=cyclic_spins,
    )
    return models.construct.FermiNetBackflow(residual_blocks)


def _make_ferminets():
    (
        key,
        ion_pos,
        _,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    slog_psis = []
    # No need for combinatorial testing over these flags; just make sure Ferminet is
    # tested with and without cyclic spins and with and without full_det
    for (
        cyclic_spins,
        full_det,
    ) in [(False, False), (True, True)]:
        compute_input_streams = _get_compute_input_streams(ion_pos)
        backflow = _get_backflow(spin_split, ndense_list, cyclic_spins)

        slog_psi = models.construct.FermiNet(
            spin_split,
            compute_input_streams,
            backflow,
            3,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            0.0,
            models.weights.get_bias_initializer("uniform"),
            orbitals_use_bias=True,
            isotropic_decay=True,
            full_det=full_det,
        )
        slog_psis.append(slog_psi)

    return key, init_pos, slog_psis


def _make_factorized_antisymmetries():
    (
        key,
        ion_pos,
        ion_charges,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    compute_input_streams = _get_compute_input_streams(ion_pos)
    backflow = _get_backflow(
        spin_split,
        ndense_list,
        cyclic_spins=False,
    )
    jastrow = models.jastrow.get_two_body_decay_scaled_for_chargeless_molecules(
        ion_pos, ion_charges
    )

    slog_psis = [
        models.construct.FactorizedAntisymmetry(
            spin_split,
            compute_input_streams,
            backflow,
            jastrow,
            rank,
            32,
            3,
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_bias_initializer("uniform"),
            jnp.tanh,
        )
        for rank in (1, 3)
    ]

    return key, init_pos, slog_psis


def _make_generic_antisymmetry():
    (
        key,
        ion_pos,
        ion_charges,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()
    compute_input_streams = _get_compute_input_streams(ion_pos)
    backflow = _get_backflow(
        spin_split,
        ndense_list,
        cyclic_spins=True,
    )
    jastrow = models.jastrow.get_two_body_decay_scaled_for_chargeless_molecules(
        ion_pos, ion_charges
    )
    slog_psi = models.construct.GenericAntisymmetry(
        spin_split,
        compute_input_streams,
        backflow,
        jastrow,
        32,
        3,
        models.weights.get_kernel_initializer("lecun_normal"),
        models.weights.get_bias_initializer("uniform"),
        jnp.tanh,
    )

    return key, init_pos, slog_psi


def _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi):
    key, subkey = jax.random.split(key)
    params = slog_psi.init(subkey, init_pos)
    results = jax.jit(slog_psi.apply)(params, init_pos)
    chex.assert_shape(results, init_pos.shape[:-2])


def test_ferminet_can_be_constructed():
    """Check construction of FermiNet does not fail."""
    _make_ferminets()


@pytest.mark.slow
def test_ferminet_can_be_evaluated():
    """Check evaluation of FermiNet does not fail."""
    key, init_pos, slog_psis = _make_ferminets()
    [
        _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)
        for slog_psi in slog_psis
    ]


def test_factorized_antisymmetry_can_be_constructed():
    """Check construction of FactorizedAntisymmetry does not fail."""
    _make_factorized_antisymmetries()


@pytest.mark.slow
def test_factorized_antisymmetry_can_be_evaluated():
    """Check evaluation of FactorizedAntisymmetry does not fail."""
    key, init_pos, slog_psis = _make_factorized_antisymmetries()
    [
        _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)
        for slog_psi in slog_psis
    ]


def test_generic_antisymmetry_can_be_constructed():
    """Check construction of GenericAntisymmetry does not fail."""
    _make_generic_antisymmetry()


@pytest.mark.slow
def test_generic_antisymmetry_can_be_evaluated():
    """Check evaluation of GenericAntisymmetry does not fail."""
    key, init_pos, slog_psi = _make_generic_antisymmetry()
    _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)


def test_get_model_from_default_config():
    """Test that construction using the default model config does not raise an error."""
    ion_pos = jnp.array([[1.0, 2.0, 3.0], [-2.0, 3.0, -4.0], [-0.5, 0.0, 0.0]])
    ion_charges = jnp.array([1.0, 3.0, 2.0])
    nelec = jnp.array([4, 3])

    def _construct_model(
        model_type,
        explicit_antisym_subtype=None,
    ):
        model_config = get_default_config_with_chosen_model(
            model_type,
            explicit_antisym_subtype=explicit_antisym_subtype,
        ).model
        models.construct.get_model_from_config(
            model_config, nelec, ion_pos, ion_charges
        )

    for model_type in ["explicit_antisym"]:
        for subtype in ["factorized", "generic"]:
            _construct_model(model_type, explicit_antisym_subtype=subtype)

    for model_type in [
        "ferminet",
        "embedded_particle_ferminet",
        "extended_orbital_matrix_ferminet",
    ]:
        _construct_model("ferminet")
