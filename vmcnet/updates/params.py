"""Routines which handle model parameter updating."""
from typing import Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import kfac_ferminet_alpha
from kfac_ferminet_alpha import optimizer as kfac_opt

import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    ModelApply,
    OptimizerState,
    P,
    PRNGKey,
    PyTree,
    S,
    UpdateDataFn,
)

UpdateParamFn = Callable[[P, D, S, PRNGKey], Tuple[P, D, S, Dict, PRNGKey]]


def _update_metrics_with_noclip(
    energy_noclip: float, variance_noclip: float, metrics: Dict
) -> Dict:
    if energy_noclip is not None:
        metrics.update({"energy_noclip": energy_noclip})
    if variance_noclip is not None:
        metrics.update({"variance_noclip": variance_noclip})
    return metrics


def _make_traced_fn_with_single_metrics(
    update_param_fn: UpdateParamFn[P, D, S],
    apply_pmap: bool,
    metrics_to_get_first: Optional[Iterable[str]] = None,
) -> UpdateParamFn[P, D, S]:
    if not apply_pmap:
        return jax.jit(update_param_fn)

    pmapped_update_param_fn = utils.distribute.pmap(update_param_fn)

    def pmapped_update_param_fn_with_single_metrics(params, data, optimizer_state, key):
        params, optimizer_state, metrics, key = pmapped_update_param_fn(
            params, data, optimizer_state, key
        )
        if metrics_to_get_first is None:
            metrics = utils.distribute.get_first(metrics)
        else:
            for metric in metrics_to_get_first:
                distributed_metric = metrics.get(metric)
                if distributed_metric is not None:
                    metrics[metric] = utils.distribute.get_first(distributed_metric)

        return params, optimizer_state, metrics, key

    return pmapped_update_param_fn_with_single_metrics


def create_grad_energy_update_param_fn(
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    optimizer_apply: Callable[[P, P, S, D], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy.

    See :func:`~vmcnet.train.vmc.vmc_loop` for its usage.

    Args:
        energy_data_val_and_grad (Callable): function which computes the clipped energy
            value and gradient. Has the signature
                (params, x)
                -> ((expected_energy, auxiliary_energy_data), grad_energy),
            where auxiliary_energy_data is the tuple
            (expected_variance, local_energies, unclipped_energy, unclipped_variance)
        optimizer_apply (Callable): applies an update to the parameters. Has signature
            (grad_energy, params, optimizer_state) -> (new_params, new_optimizer_state).
        get_position_fn (GetPositionFromData): gets the walker positions from the MCMC
            data.
        apply_pmap (bool, optional): whether to apply jax.pmap to the walker function.
            If False, applies jax.jit. Defaults to True.

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
        The function is pmapped if apply_pmap is True, and jitted if apply_pmap is
        False.
    """

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy_data, grad_energy = energy_data_val_and_grad(params, position)
        energy, aux_energy_data = energy_data

        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(
            grad_energy, params, optimizer_state, data
        )
        metrics = {"energy": energy, "variance": aux_energy_data[0]}
        metrics = _update_metrics_with_noclip(
            aux_energy_data[2], aux_energy_data[3], metrics
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, optimizer_state, metrics, key

    traced_fn = _make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def _get_traced_compute_param_norm(
    apply_pmap: bool = True,
) -> Callable[[PyTree], Array]:
    if not apply_pmap:
        return jax.jit(tree_reduce_l1)

    return utils.distribute.pmap(tree_reduce_l1)


def create_kfac_update_param_fn(
    optimizer: kfac_ferminet_alpha.Optimizer,
    damping: jnp.float32,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[kfac_opt.Parameters, D, kfac_opt.State]:
    """Create momentum-less KFAC update step function.

    Args:
        optimizer (kfac_ferminet_alpha.Optimizer): instance of the Optimizer class from
            kfac_ferminet_alpha
        damping (jnp.float32): damping coefficient
        get_position_fn (GetPositionFromData): function which gets the walker positions
            from the data. Has signature data -> Array

    Returns:
        Callable: function which updates the parameters given the current data, params,
        and optimizer state. The signature of this function is
            (data, params, optimizer_state, key)
            -> (new_params, new_optimizer_state, metrics, key)
    """
    momentum = jnp.asarray(0.0)
    damping = jnp.asarray(damping)
    if optimizer.multi_device:
        momentum = utils.distribute.replicate_all_local_devices(momentum)
        damping = utils.distribute.replicate_all_local_devices(damping)

    traced_compute_param_norm = _get_traced_compute_param_norm(optimizer.multi_device)

    def update_param_fn(params, data, optimizer_state, key):
        key, subkey = utils.distribute.split_or_psplit_key(key, optimizer.multi_device)
        params, optimizer_state, stats = optimizer.step(
            params=params,
            state=optimizer_state,
            rng=subkey,
            data_iterator=iter([get_position_fn(data)]),
            momentum=momentum,
            damping=damping,
        )
        data = update_data_fn(data, params)

        energy = stats["loss"]
        variance = stats["aux"][0]
        energy_noclip = stats["aux"][2]
        variance_noclip = stats["aux"][3]
        picked_stats = (energy, variance, energy_noclip, variance_noclip)

        if record_param_l1_norm:
            param_l1_norm = traced_compute_param_norm(params)
            picked_stats = picked_stats + (param_l1_norm,)

        stats_to_save = picked_stats
        if optimizer.multi_device:
            stats_to_save = [utils.distribute.get_first(stat) for stat in picked_stats]

        metrics = {"energy": stats_to_save[0], "variance": stats_to_save[1]}
        metrics = _update_metrics_with_noclip(
            stats_to_save[2], stats_to_save[3], metrics
        )

        if record_param_l1_norm:
            metrics.update({"param_l1_norm": stats_to_save[4]})

        return params, data, optimizer_state, metrics, key

    return update_param_fn


def create_eval_update_param_fn(
    local_energy_fn: ModelApply[P],
    nchains: int,
    get_position_fn: GetPositionFromData[D],
    apply_pmap: bool = True,
    record_local_energies: bool = True,
    nan_safe: bool = False,
) -> UpdateParamFn[P, D, OptimizerState]:
    """No update/clipping/grad function which simply evaluates the local energies.

    Can be used to do simple unclipped MCMC with :func:`~vmcnet.train.vmc.vmc_loop`.

    Arguments:
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        get_position_fn (GetPositionFromData): gets the walker positions from the MCMC
            data.
        nan_safe (bool): whether or not to mask local energy nans in the evaluation
            process. This option should not be used under normal circumstances, as the
            energy estimates are of unclear validity if nans are masked. However,
            it can be used to get a coarse estimate of the energy of a wavefunction even
            if a few walkers are returning nans for their local energies.

    Returns:
        Callable: function which evaluates the local energies and averages them, without
        updating the parameters
    """

    def eval_update_param_fn(params, data, optimizer_state, key):
        local_energies = local_energy_fn(params, get_position_fn(data))
        energy, variance = physics.core.get_statistics_from_local_energy(
            local_energies, nchains, nan_safe=nan_safe
        )
        metrics = {"energy": energy, "variance": variance}
        if record_local_energies:
            metrics.update({"local_energies": local_energies})
        return params, optimizer_state, metrics, key

    traced_fn = _make_traced_fn_with_single_metrics(
        eval_update_param_fn, apply_pmap, {"energy", "variance"}
    )

    return traced_fn


def constrain_norm(
    grads: P,
    preconditioned_grads: P,
    learning_rate: jnp.float32,
    norm_constraint: jnp.float32 = 0.001,
) -> P:
    """Constrains the preconditioned norm of the update, adapted from KFAC."""
    sq_norm_grads = tree_inner_product(preconditioned_grads, grads)
    sq_norm_scaled_grads = sq_norm_grads * learning_rate**2

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = utils.distribute.pmean_if_pmap(sq_norm_scaled_grads)

    max_coefficient = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(max_coefficient, 1)
    constrained_grads = multiply_tree_by_scalar(preconditioned_grads, coefficient)

    return constrained_grads
