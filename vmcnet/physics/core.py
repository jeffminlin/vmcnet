"""Core local energy and gradient construction routines."""

from typing import Callable, Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import kfac_jax

import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    ArrayLike,
    P,
    ClippingFn,
    PRNGKey,
    LocalEnergyApply,
    ModelApply,
    Dict,
    Any,
)

EnergyAuxData = Dict[str, Any]
ValueGradEnergyFn = Callable[[P, Array], Tuple[Array, EnergyAuxData, P]]


def initialize_molecular_pos(
    key: PRNGKey,
    nchains: int,
    ion_pos: Array,
    ion_charges: Array,
    nelec_total: int,
    init_width: float = 1.0,
    dtype=chex.Numeric,
) -> Tuple[PRNGKey, Array]:
    """Initialize a set of plausible initial electron positions.

    For each chain, each electron is assigned to a random ion and then its position is
    sampled from a normal distribution centered at that ion with diagonal covariance
    with diagonal entries all equal to init_width.

    If there are no more electrons than there are ions, the assignment is done without
    replacement. If there are more electrons than ions, the assignment is done with
    replacement, and the probability of choosing ion i is its relative charge (as a
    fraction of the sum of the ion charges).
    """
    nion = len(ion_charges)
    replace = True

    if nelec_total <= nion:
        replace = False

    assignments = []
    for _ in range(nchains):
        key, subkey = jax.random.split(key)
        choices = jax.random.choice(
            subkey,
            nion,
            shape=(nelec_total,),
            replace=replace,
            p=ion_charges / jnp.sum(ion_charges),
        )
        assignments.append(ion_pos[choices])
    elecs_at_ions = jnp.stack(assignments, axis=0)
    key, subkey = jax.random.split(key)
    return key, elecs_at_ions + init_width * jax.random.normal(
        subkey, elecs_at_ions.shape, dtype=dtype
    )


def combine_local_energy_terms(
    local_energy_terms: Sequence[ModelApply[P]],
) -> LocalEnergyApply[P]:
    """Combine a sequence of local energy terms by adding them.

    Args:
        local_energy_terms (Sequence): sequence of local energy terms, each with the
            signature (params, x) -> array of terms of shape (x.shape[0],)

    Returns:
        Callable: local energy function which computes the sum of the local energy
        terms. Has the signature
        (params, x) -> local energy array of shape (x.shape[0],)
    """

    def local_energy_fn(params: P, x: Array, key: Optional[PRNGKey]) -> Array:
        del key
        local_energy_sum = local_energy_terms[0](params, x)
        for term in local_energy_terms[1:]:
            local_energy_sum = cast(Array, local_energy_sum + term(params, x))
        return local_energy_sum

    return local_energy_fn


def get_statistics_from_local_energy(
    local_energies: Array, nchains: int, nan_safe: bool = True
) -> Tuple[Array, Array]:
    """Collectively reduce local energies to an average energy and variance.

    Args:
        local_energies (Array): local energies of shape (nchains,), possibly
            distributed across multiple devices via utils.distribute.pmap.
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        nan_safe (bool, optional): flag which controls if jnp.nanmean is used instead of
            jnp.mean. Can be set to False when debugging if trying to find the source of
            unexpected nans. Defaults to True.

    Returns:
        (chex.Numeric, chex.Numeric): local energy average, local energy (sample)
        variance
    """
    # TODO(Jeffmin) might be worth investigating the numerical stability of the XLA
    # compiled version of these two computations, since the quality of the gradients
    # is fairly crucial to the success of the algorithm
    if nan_safe:
        allreduce_mean = utils.distribute.nanmean_all_local_devices
    else:
        allreduce_mean = utils.distribute.mean_all_local_devices
    energy = allreduce_mean(local_energies)
    variance = (
        allreduce_mean(jnp.square(local_energies - energy)) * nchains / (nchains - 1)
    )  # adjust by n / (n - 1) to get an unbiased estimator
    return energy, variance


def get_clipped_energies_and_stats(
    local_energies_noclip: Array,
    nchains: int,
    clipping_fn: Optional[ClippingFn],
    nan_safe: bool,
) -> Tuple[Array, Array, EnergyAuxData]:
    """Clip local energies if requested and return auxiliary data."""
    energy_noclip, variance_noclip = get_statistics_from_local_energy(
        local_energies_noclip, nchains, nan_safe=False
    )

    if clipping_fn is not None:
        local_energies = clipping_fn(local_energies_noclip, energy_noclip)
    else:
        local_energies = local_energies_noclip

    energy, variance = get_statistics_from_local_energy(
        local_energies, nchains, nan_safe=nan_safe
    )

    energy_stats = dict(
        variance=variance,
        energy_noclip=energy_noclip,
        variance_noclip=variance_noclip,
    )

    return energy, local_energies, energy_stats


def create_value_and_grad_energy_fn(
    log_psi_apply: ModelApply[P],
    local_energy_fn: LocalEnergyApply[P],
    nchains: int,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
) -> ValueGradEnergyFn[P]:
    """Create a function which computes unbiased energy gradients.

    Due to the Hermiticity of the Hamiltonian, we can get an unbiased lower variance
    estimate of the gradient of the expected energy than the naive gradient of the
    mean of sampled local energies. Specifically, the gradient of the expected energy
    expect[E_L] takes the form

        2 * expect[(E_L - expect[E_L]) * (grad_psi / psi)(x)],

    where E_L is the local energy and expect[] denotes the expectation with respect to
    the distribution |psi|^2.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        clipping_fn (Callable, optional): post-processing function on the local energy,
            e.g. a function which clips the values to be within some multiple of the
            total variation from the median. The post-processed values are used for
            the gradient calculation, if available. Defaults to None.
        nan_safe (bool, optional): flag which controls if jnp.nanmean and jnp.nansum are
            used instead of jnp.mean and jnp.sum for the terms in the gradient
            calculation. Can be set to False when debugging if trying to find the source
            of unexpected nans. Defaults to True.

    Returns:
        Callable: function which computes the clipped energy value and gradient. Has the
        signature
            (params, x)
            -> ((expected_energy, auxiliary_energy_data), grad_energy),
        where auxiliary_energy_data is the tuple
        (expected_variance, local_energies, unclipped_energy, unclipped_variance, centered_local_energies)
    """
    mean_grad_fn = utils.distribute.get_mean_over_first_axis_fn(nan_safe=nan_safe)

    def standard_estimator_forward(
        params: P,
        positions: Array,
        centered_local_energies: Array,
    ) -> ArrayLike:
        log_psi = log_psi_apply(params, positions)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        # NOTE: for the generic gradient estimator case it may be important to include
        # the (nchains / nchains -1) factor here to make sure the standard and generic
        # gradient terms aren't mismatched by a slight scale factor.
        return (
            2.0
            * nchains
            / (nchains - 1)
            * mean_grad_fn(centered_local_energies * log_psi)
        )

    def get_standard_contribution(local_energies_noclip, params, positions):
        energy, local_energies, stats = get_clipped_energies_and_stats(
            local_energies_noclip, nchains, clipping_fn, nan_safe
        )
        centered_local_energies = local_energies - energy
        grad_E = jax.grad(standard_estimator_forward, argnums=0)(
            params, positions, centered_local_energies
        )
        return energy, stats, grad_E

    def energy_val_and_grad(params, positions):
        local_energies_noclip = jax.vmap(
            local_energy_fn, in_axes=(None, 0, None), out_axes=0
        )(params, positions, None)

        energy, stats, grad_E = get_standard_contribution(
            local_energies_noclip, params, positions
        )

        return energy, stats, grad_E

    return energy_val_and_grad


def create_energy_and_statistics_fn(
    local_energy_fn: LocalEnergyApply[P],
    nchains: int,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
) -> ValueGradEnergyFn[P]:
    """Create a function which computes energies and associated statistics.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        clipping_fn (Callable, optional): post-processing function on the local energy,
            e.g. a function which clips the values to be within some multiple of the
            total variation from the median. The post-processed values are used for
            the gradient calculation, if available. Defaults to None.
        nan_safe (bool, optional): flag which controls if jnp.nanmean and jnp.nansum are
            used instead of jnp.mean and jnp.sum for the terms in the gradient
            calculation. Can be set to False when debugging if trying to find the source
            of unexpected nans. Defaults to True.

    Returns:
        Callable: function which computes the clipped energy and associated statistics.
        Has the signature
            (params, positions)
            -> (expected_energy, auxiliary_energy_data)
        where auxiliary_energy_data is the tuple
        (expected_variance, local_energies, unclipped_energy, unclipped_variance, centered_local_energies)
    """

    def energy_and_statistics(params, positions):
        local_energies_noclip = jax.vmap(
            local_energy_fn, in_axes=(None, 0, None), out_axes=0
        )(params, positions, None)

        energy, local_energies, aux_data = get_clipped_energies_and_stats(
            local_energies_noclip, nchains, clipping_fn, nan_safe
        )

        return energy, local_energies, aux_data

    return energy_and_statistics
