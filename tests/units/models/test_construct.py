"""Test model construction."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmcnet.models as models
from vmcnet.utils.typing import ArrayList

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
        return functools.reduce(lambda a, b: a * b, jax.tree_leaves(antisymmetry_tree))

    model = models.construct.ComposedModel([equivariance, product_antisymmetry])

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

    key, subkey = jax.random.split(key)
    init_pos = jax.random.normal(subkey, shape=(7, 6, 3))

    spin_split = (2,)  # 2 up, 4 down
    ndense_list = ((6, 3), (6, 3), (3, 3), (3, 4), (12,), (11, 3), (9,))

    return key, ion_pos, init_pos, spin_split, ndense_list


def _get_backflow(spin_split, ndense_list, cyclic_spins, ion_pos):
    residual_blocks = models.construct.get_residual_blocks_for_ferminet_backflow(
        spin_split,
        ndense_list,
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_kernel_initializer("xavier_normal"),
        models.weights.get_kernel_initializer("glorot_uniform"),
        models.weights.get_kernel_initializer("kaiming_uniform"),
        models.weights.get_bias_initializer("zeros"),
        models.weights.get_bias_initializer("normal"),
        jnp.tanh,
        cyclic_spins=cyclic_spins,
    )
    return models.construct.FermiNetBackflow(residual_blocks, ion_pos=ion_pos)


def _get_det_resnet():
    return models.construct.get_resnet_determinant_fn_for_ferminet(
        6,
        3,
        jax.nn.gelu,
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_bias_initializer("uniform"),
    )


def _make_ferminets():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    log_psis = []
    # No need for combinatorial testing over these flags; just test with both
    # false and both true to cover our bases without making the test too slow.
    for (cyclic_spins, use_det_resnet) in [(False, False), (True, True)]:
        backflow = _get_backflow(spin_split, ndense_list, cyclic_spins, ion_pos)
        resnet_det_fn = _get_det_resnet() if use_det_resnet else None
        log_psi = models.construct.FermiNet(
            spin_split,
            backflow,
            3,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
            determinant_fn=resnet_det_fn,
        )
        log_psis.append(log_psi)

    return key, init_pos, log_psis


def _make_embedded_particle_ferminets():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    log_psis = []
    cyclic_spins = False
    backflow = _get_backflow(spin_split, ndense_list, cyclic_spins, ion_pos)
    invariance_backflow = _get_backflow(spin_split, ndense_list, cyclic_spins, ion_pos)
    nhidden_fermions_per_spin_vals = [(2, 3), 2]

    for nhidden_fermions_per_spin in nhidden_fermions_per_spin_vals:
        log_psi = models.construct.EmbeddedParticleFermiNet(
            spin_split,
            nhidden_fermions_per_spin,
            backflow,
            3,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
            invariance_backflow=invariance_backflow,
            invariance_kernel_initializer=models.weights.get_kernel_initializer(
                "he_normal"
            ),
            invariance_bias_initializer=models.weights.get_bias_initializer("uniform"),
        )
        log_psis.append(log_psi)

    return key, init_pos, log_psis


def _make_antiequivariance_net(
    spin_split, ndense_list, antiequivariance, cyclic_spins, ion_pos
):
    backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins=cyclic_spins, ion_pos=ion_pos
    )

    def array_list_equivariance(x: ArrayList) -> jnp.ndarray:
        concat_x = jnp.concatenate(x, axis=-2)
        return _get_backflow(
            spin_split, ((9,), (2,), (1,)), cyclic_spins=True, ion_pos=None
        )(concat_x)[0]

    log_psi = models.construct.AntiequivarianceNet(
        backflow, antiequivariance, array_list_equivariance
    )

    return log_psi


def _make_orbital_cofactor_net():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    antiequivariance = models.antiequivariance.OrbitalCofactorAntiequivarianceLayer(
        spin_split,
        models.weights.get_kernel_initializer("he_normal"),
        models.weights.get_kernel_initializer("lecun_normal"),
        models.weights.get_kernel_initializer("ones"),
        models.weights.get_bias_initializer("uniform"),
    )

    log_psi = _make_antiequivariance_net(
        spin_split, ndense_list, antiequivariance, cyclic_spins=True, ion_pos=ion_pos
    )

    return key, init_pos, log_psi


def _make_per_particle_dets_net():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    antiequivariance = (
        models.antiequivariance.PerParticleDeterminantAntiequivarianceLayer(
            spin_split,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
        )
    )

    log_psi = _make_antiequivariance_net(
        spin_split, ndense_list, antiequivariance, cyclic_spins=False, ion_pos=ion_pos
    )

    return key, init_pos, log_psi


def _make_split_antisymmetry():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins=False, ion_pos=ion_pos
    )
    log_psi = models.construct.SplitBruteForceAntisymmetryWithDecay(
        spin_split,
        backflow,
        32,
        3,
        models.weights.get_kernel_initializer("lecun_normal"),
        models.weights.get_kernel_initializer("ones"),
        models.weights.get_bias_initializer("uniform"),
        jnp.tanh,
    )

    return key, init_pos, log_psi


def _make_double_antisymmetry():
    key, ion_pos, init_pos, spin_split, ndense_list = _get_initial_pos_and_hyperparams()

    backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins=True, ion_pos=ion_pos
    )
    log_psi = models.construct.ComposedBruteForceAntisymmetryWithDecay(
        spin_split,
        backflow,
        32,
        3,
        models.weights.get_kernel_initializer("lecun_normal"),
        models.weights.get_kernel_initializer("ones"),
        models.weights.get_bias_initializer("uniform"),
        jnp.tanh,
    )

    return key, init_pos, log_psi


def _jit_eval_model(key, init_pos, log_psi):
    key, subkey = jax.random.split(key)
    params = log_psi.init(subkey, init_pos)
    jax.jit(log_psi.apply)(params, init_pos)


def test_ferminet_can_be_constructed():
    """Check construction of FermiNet does not fail."""
    _make_ferminets()


@pytest.mark.slow
def test_ferminet_can_be_evaluated():
    """Check evaluation of FermiNet does not fail."""
    key, init_pos, log_psis = _make_ferminets()
    (_jit_eval_model(key, init_pos, log_psi) for log_psi in log_psis)


def test_embedded_particle_ferminet_can_be_constructed():
    """Check construction of EmbeddedParticleFerminet does not fail."""
    _make_embedded_particle_ferminets()


@pytest.mark.slow
def test_embedded_particle_ferminet_can_be_evaluated():
    """Check evaluation of EmbeddedParticleFerminet does not fail."""
    key, init_pos, log_psis = _make_embedded_particle_ferminets()
    (_jit_eval_model(key, init_pos, log_psi) for log_psi in log_psis)


def test_orbital_cofactor_net_can_be_constructed():
    """Check construction of the orbital cofactor AntiequivarianceNet does not fail."""
    _make_orbital_cofactor_net()


@pytest.mark.slow
def test_orbital_cofactor_net_can_be_evaluated():
    """Check evaluation of the orbital cofactor AntiequivarianceNet."""
    key, init_pos, log_psi = _make_orbital_cofactor_net()
    _jit_eval_model(key, init_pos, log_psi)


def test_per_particle_dets_net_can_be_constructed():
    """Check construction of the per-particle dets AntiequivarianceNet does not fail."""
    _make_per_particle_dets_net()


@pytest.mark.slow
def test_per_particle_dets_net_can_be_evaluated():
    """Check evaluation of the per-particle dets AntiequivarianceNet."""
    key, init_pos, log_psi = _make_per_particle_dets_net()
    _jit_eval_model(key, init_pos, log_psi)


def test_split_antisymmetry_can_be_constructed():
    """Check construction of SplitBruteForceAntisymmetryWithDecay does not fail."""
    _make_split_antisymmetry()


@pytest.mark.slow
def test_split_antisymmetry_can_be_evaluated():
    """Check evaluation of SplitBruteForceAntisymmetryWithDecay does not fail."""
    key, init_pos, log_psi = _make_split_antisymmetry()
    _jit_eval_model(key, init_pos, log_psi)


def test_composed_antisymmetry_can_be_constructed():
    """Check construction of ComposedBruteForceAntisymmetryWithDecay does not fail."""
    _make_double_antisymmetry()


@pytest.mark.slow
def test_ferminet_composed_antisymmetry_can_be_evaluated():
    """Check evaluation of ComposedBruteForceAntisymmetryWithDecay does not fail."""
    key, init_pos, log_psi = _make_double_antisymmetry()
    _jit_eval_model(key, init_pos, log_psi)


def test_get_model_from_default_config():
    """Test that construction using the default model config does not raise an error."""
    ion_pos = jnp.array([[1.0, 2.0, 3.0], [-2.0, 3.0, -4.0], [-0.5, 0.0, 0.0]])
    nelec = jnp.array([4, 3])

    for model_type in [
        "ferminet",
        "embedded_particle_ferminet",
        "orbital_cofactor_net",
        "per_particle_dets_net",
        "brute_force_antisym",
    ]:
        if model_type == "brute_force_antisym":
            for subtype in ["rank_one", "double"]:
                model_config = get_default_config_with_chosen_model(
                    model_type, brute_force_subtype=subtype
                ).model
                models.construct.get_model_from_config(model_config, nelec, ion_pos)
        elif model_type == "ferminet":
            for use_det_resnet in [False, True]:
                model_config = get_default_config_with_chosen_model(model_type).model
                model_config.use_det_resnet = use_det_resnet
                models.construct.get_model_from_config(model_config, nelec, ion_pos)
        elif model_type == "embedded_particle_ferminet":
            model_config = get_default_config_with_chosen_model(model_type).model
            models.construct.get_model_from_config(model_config, nelec, ion_pos)
        elif model_type in ["orbital_cofactor_net", "per_particle_dets_net"]:
            model_config = get_default_config_with_chosen_model(model_type).model
            models.construct.get_model_from_config(model_config, nelec, ion_pos)
