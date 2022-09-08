"""Test model construction."""
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmcnet.models as models
import vmcnet.models.sign_symmetry as sign_sym
from vmcnet.models.construct import DeterminantFnMode
from vmcnet.utils.typing import Array, ArrayList

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


def _get_backflow(spin_split, ndense_list, cyclic_spins, use_transformer, num_heads):
    residual_blocks = models.construct.get_residual_blocks_for_ferminet_backflow(
        spin_split,
        ndense_list,
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_kernel_initializer("xavier_normal"),
        models.weights.get_kernel_initializer("glorot_uniform"),
        models.weights.get_kernel_initializer("kaiming_uniform"),
        models.weights.get_bias_initializer("zeros"),
        models.weights.get_bias_initializer("zeros"),
        models.weights.get_bias_initializer("normal"),
        jnp.tanh,
        cyclic_spins=cyclic_spins,
        use_transformer=use_transformer,
        num_heads=num_heads,
    )
    return models.construct.FermiNetBackflow(residual_blocks)


def _get_det_resnet_fn():
    return models.construct.get_resnet_determinant_fn_for_ferminet(
        6,
        3,
        jax.nn.gelu,
        models.weights.get_kernel_initializer("orthogonal"),
        models.weights.get_bias_initializer("uniform"),
    )


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
    # tested with and without cyclic spins, and with each different determinant_fn_mode.
    for (
        cyclic_spins,
        use_det_resnet,
        determinant_fn_mode,
        full_det,
        num_heads,
        use_transformer,
    ) in [
        (
            False,
            False,
            models.construct.DeterminantFnMode.SIGN_COVARIANCE,
            False,
            1,
            False,
        ),
        (
            False,
            False,
            models.construct.DeterminantFnMode.SIGN_COVARIANCE,
            True,
            1,
            False,
        ),
        (
            True,
            True,
            models.construct.DeterminantFnMode.SIGN_COVARIANCE,
            False,
            1,
            False,
        ),
        (
            False,
            True,
            models.construct.DeterminantFnMode.PARALLEL_EVEN,
            False,
            1,
            False,
        ),
        (True, True, models.construct.DeterminantFnMode.PAIRWISE_EVEN, False, 1, False),
    ]:
        compute_input_streams = _get_compute_input_streams(ion_pos)
        backflow = _get_backflow(
            spin_split, ndense_list, cyclic_spins, use_transformer, num_heads
        )
        resnet_det_fn = _get_det_resnet_fn() if use_det_resnet else None

        slog_psi = models.construct.FermiNet(
            spin_split,
            compute_input_streams,
            backflow,
            3,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
            orbitals_use_bias=True,
            isotropic_decay=True,
            determinant_fn=resnet_det_fn,
            determinant_fn_mode=determinant_fn_mode,
            full_det=full_det,
        )
        slog_psis.append(slog_psi)

    return key, init_pos, slog_psis


def _make_embedded_particle_ferminets():
    (
        key,
        ion_pos,
        _,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    log_psis = []
    cyclic_spins = False
    num_heads = 1
    use_transformer = False
    compute_input_streams = _get_compute_input_streams(ion_pos)
    invariance_compute_input_streams = _get_compute_input_streams(ion_pos)
    invariance_backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins, use_transformer, num_heads
    )

    for nhidden_fermions_per_spin, full_det in [((2, 3), False), ((4, 0), True)]:
        total_spin_split = (spin_split[0] + nhidden_fermions_per_spin[0],)
        backflow = _get_backflow(
            total_spin_split, ndense_list, cyclic_spins, use_transformer, num_heads
        )
        slog_psi = models.construct.EmbeddedParticleFermiNet(
            spin_split,
            compute_input_streams,
            backflow,
            3,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
            orbitals_use_bias=True,
            isotropic_decay=True,
            determinant_fn=None,
            determinant_fn_mode=DeterminantFnMode.PARALLEL_EVEN,
            full_det=full_det,
            nhidden_fermions_per_spin=nhidden_fermions_per_spin,
            invariance_compute_input_streams=invariance_compute_input_streams,
            invariance_backflow=invariance_backflow,
            invariance_kernel_initializer=models.weights.get_kernel_initializer(
                "he_normal"
            ),
            invariance_bias_initializer=models.weights.get_bias_initializer("uniform"),
        )
        log_psis.append(slog_psi)

    return key, init_pos, log_psis


def _make_extended_orbital_matrix_ferminets():
    (
        key,
        ion_pos,
        _,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    slog_psis = []
    cyclic_spins = False
    num_heads = 1
    use_transformer = False
    backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins, use_transformer, num_heads
    )
    extra_backflow = _get_backflow(
        spin_split, ndense_list, cyclic_spins, use_transformer, num_heads
    )

    for nhidden_fermions_per_spin, use_separate_invariance_backflow, full_det in [
        ((2, 3), True, False),
        ((4, 0), False, True),
    ]:
        if use_separate_invariance_backflow:
            invariance_backflow = extra_backflow
        else:
            invariance_backflow = None

        slog_psi = models.construct.ExtendedOrbitalMatrixFermiNet(
            spin_split=spin_split,
            compute_input_streams=_get_compute_input_streams(ion_pos),
            backflow=backflow,
            ndeterminants=3,
            kernel_initializer_orbital_linear=models.weights.get_kernel_initializer(
                "he_normal"
            ),
            kernel_initializer_envelope_dim=models.weights.get_kernel_initializer(
                "lecun_normal"
            ),
            kernel_initializer_envelope_ion=models.weights.get_kernel_initializer(
                "ones"
            ),
            bias_initializer_orbital_linear=models.weights.get_bias_initializer(
                "uniform"
            ),
            orbitals_use_bias=True,
            isotropic_decay=True,
            determinant_fn=None,
            determinant_fn_mode=DeterminantFnMode.PARALLEL_EVEN,
            full_det=full_det,
            nhidden_fermions_per_spin=nhidden_fermions_per_spin,
            invariance_backflow=invariance_backflow,
            invariance_kernel_initializer=models.weights.get_kernel_initializer(
                "he_normal"
            ),
            invariance_bias_initializer=models.weights.get_bias_initializer("uniform"),
        )
        slog_psis.append(slog_psi)

    return key, init_pos, slog_psis


def _make_antiequivariance_net_with_resnet_sign_covariance(
    spin_split,
    ndense_list,
    antiequivariance,
    cyclic_spins,
    ion_pos,
    multiply_by_eq_features=False,
):
    compute_input_streams = _get_compute_input_streams(ion_pos)
    num_heads = 1
    use_transformer = False
    backflow = _get_backflow(
        spin_split,
        ndense_list,
        cyclic_spins=cyclic_spins,
        use_transformer=use_transformer,
        num_heads=num_heads,
    )

    def backflow_based_equivariance(x: ArrayList) -> Array:
        concat_x = jnp.concatenate(x, axis=-2)
        return _get_backflow(
            spin_split,
            ((9,), (2,), (1,)),
            cyclic_spins=True,
            use_transformer=False,
            num_heads=1,
        )(concat_x)

    odd_equivariance = sign_sym.make_array_list_fn_sign_covariant(
        backflow_based_equivariance, axis=-3
    )

    def array_list_sign_covariance(x: ArrayList) -> Array:
        return jnp.sum(odd_equivariance(x), axis=-2)

    slog_psi = models.construct.AntiequivarianceNet(
        spin_split,
        compute_input_streams,
        backflow,
        antiequivariance,
        array_list_sign_covariance,
        multiply_by_eq_features=multiply_by_eq_features,
    )

    return slog_psi


def _make_antiequivariance_net_with_products_sign_covariance(
    spin_split, ndense_list, antiequivariance, cyclic_spins, ion_pos
):
    compute_input_streams = _get_compute_input_streams(ion_pos)
    backflow = _get_backflow(
        spin_split,
        ndense_list,
        cyclic_spins=cyclic_spins,
        use_transformer=False,
        num_heads=1,
    )

    array_list_sign_covariance = sign_sym.ProductsSignCovariance(
        1, models.weights.get_kernel_initializer("orthogonal")
    )

    slog_psi = models.construct.AntiequivarianceNet(
        spin_split,
        compute_input_streams,
        backflow,
        antiequivariance,
        array_list_sign_covariance,
    )

    return slog_psi


def _make_orbital_cofactor_nets():
    (
        key,
        ion_pos,
        _,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    antiequivariance = models.antiequivariance.OrbitalCofactorAntiequivarianceLayer(
        spin_split,
        models.weights.get_kernel_initializer("he_normal"),
        models.weights.get_kernel_initializer("lecun_normal"),
        models.weights.get_kernel_initializer("ones"),
        models.weights.get_bias_initializer("uniform"),
    )

    slog_psi_no_features = _make_antiequivariance_net_with_resnet_sign_covariance(
        spin_split,
        ndense_list,
        antiequivariance,
        cyclic_spins=True,
        ion_pos=ion_pos,
        multiply_by_eq_features=False,
    )
    slog_psi_with_features = _make_antiequivariance_net_with_resnet_sign_covariance(
        spin_split,
        ndense_list,
        antiequivariance,
        cyclic_spins=True,
        ion_pos=ion_pos,
        multiply_by_eq_features=True,
    )

    return key, init_pos, [slog_psi_no_features, slog_psi_with_features]


def _make_per_particle_dets_nets():
    (
        key,
        ion_pos,
        _,
        init_pos,
        spin_split,
        ndense_list,
    ) = _get_initial_pos_and_hyperparams()

    antiequivariance = (
        models.antiequivariance.PerParticleDeterminantAntiequivarianceLayer(
            spin_split,
            models.weights.get_kernel_initializer("he_normal"),
            models.weights.get_kernel_initializer("lecun_normal"),
            models.weights.get_kernel_initializer("ones"),
            models.weights.get_bias_initializer("uniform"),
        )
    )

    slog_psi_eq = _make_antiequivariance_net_with_resnet_sign_covariance(
        spin_split, ndense_list, antiequivariance, cyclic_spins=False, ion_pos=ion_pos
    )
    slog_psi_sign_cov = _make_antiequivariance_net_with_products_sign_covariance(
        spin_split, ndense_list, antiequivariance, cyclic_spins=True, ion_pos=ion_pos
    )

    return key, init_pos, [slog_psi_eq, slog_psi_sign_cov]


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
        use_transformer=False,
        num_heads=1,
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
        use_transformer=False,
        num_heads=1,
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


def test_embedded_particle_ferminet_can_be_constructed():
    """Check construction of EmbeddedParticleFerminet does not fail."""
    _make_embedded_particle_ferminets()


@pytest.mark.slow
def test_embedded_particle_ferminet_can_be_evaluated():
    """Check evaluation of EmbeddedParticleFerminet does not fail."""
    key, init_pos, slog_psis = _make_embedded_particle_ferminets()
    [
        _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)
        for slog_psi in slog_psis
    ]


def test_extended_orbital_matrix_ferminet_can_be_constructed():
    """Check construction of ExtendedOrbitalMatrixFermiNet does not fail."""
    _make_extended_orbital_matrix_ferminets()


@pytest.mark.slow
def test_extended_orbital_matrix_ferminet_can_be_evaluated():
    """Check evaluation of ExtendedOrbitalMatrixFermiNet does not fail."""
    key, init_pos, slog_psis = _make_extended_orbital_matrix_ferminets()
    [
        _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)
        for slog_psi in slog_psis
    ]


def test_orbital_cofactor_net_can_be_constructed():
    """Check construction of the orbital cofactor AntiequivarianceNet does not fail."""
    _make_orbital_cofactor_nets()


@pytest.mark.slow
def test_orbital_cofactor_net_can_be_evaluated():
    """Check evaluation of the orbital cofactor AntiequivarianceNet."""
    key, init_pos, slog_psis = _make_orbital_cofactor_nets()
    [
        _jit_eval_model_and_verify_output_shape(key, init_pos, slog_psi)
        for slog_psi in slog_psis
    ]


def test_per_particle_dets_net_can_be_constructed():
    """Check construction of the per-particle dets AntiequivarianceNet does not fail."""
    _make_per_particle_dets_nets()


@pytest.mark.slow
def test_per_particle_dets_net_can_be_evaluated():
    """Check evaluation of the per-particle dets AntiequivarianceNet."""
    key, init_pos, slog_psis = _make_per_particle_dets_nets()
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
        use_det_resnet=True,
        determinant_fn_mode=None,
        explicit_antisym_subtype=None,
        use_products_covariance=False,
    ):
        model_config = get_default_config_with_chosen_model(
            model_type,
            use_det_resnet=use_det_resnet,
            determinant_fn_mode=determinant_fn_mode,
            explicit_antisym_subtype=explicit_antisym_subtype,
            use_products_covariance=use_products_covariance,
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
        _construct_model(model_type, use_det_resnet=False)
        for mode in ["sign_covariance", "parallel_even", "pairwise_even"]:
            _construct_model(model_type, use_det_resnet=True, determinant_fn_mode=mode)
    for model_type in ["orbital_cofactor_net", "per_particle_dets_net"]:
        for use_products_covariance in [False, True]:
            _construct_model(
                model_type, use_products_covariance=use_products_covariance
            )
