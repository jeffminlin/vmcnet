"""Routines which handle model parameter updating."""

from typing import Callable, Dict, Iterable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import kfac_jax

import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.pytree_helpers import (
    tree_reduce_l1,
)
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LocalEnergyApply,
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
        params, data, optimizer_state, metrics, key = pmapped_update_param_fn(
            params, data, optimizer_state, key
        )
        if metrics_to_get_first is None:
            metrics = utils.distribute.get_first(metrics)
        else:
            for metric in metrics_to_get_first:
                distributed_metric = metrics.get(metric)
                if distributed_metric is not None:
                    metrics[metric] = utils.distribute.get_first(distributed_metric)

        return params, data, optimizer_state, metrics, key

    return pmapped_update_param_fn_with_single_metrics


def create_grad_energy_update_param_fn(
    energy_data_val_and_grad: physics.core.ValueGradEnergyFn[P],
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
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
        update_data_fn (Callable): function which updates data for new params
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
        key, subkey = jax.random.split(key)

        energy_data, grad_energy = energy_data_val_and_grad(params, subkey, position)
        energy, aux_energy_data = energy_data

        grad_energy = utils.distribute.pmean_if_pmap(grad_energy)
        params, optimizer_state = optimizer_apply(
            grad_energy,
            params,
            optimizer_state,
            data,
            dict(centered_local_energies=aux_energy_data["centered_local_energies"]),
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": aux_energy_data["variance"]}
        metrics = _update_metrics_with_noclip(
            aux_energy_data["energy_noclip"],
            aux_energy_data["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = _make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def _get_traced_compute_param_norm(
    apply_pmap: bool = True,
) -> Callable[[PyTree], Array]:
    if not apply_pmap:
        return jax.jit(tree_reduce_l1)

    return utils.distribute.pmap(tree_reduce_l1)


def create_kfac_update_param_fn(
    optimizer: kfac_jax.Optimizer,
    damping: chex.Numeric,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, OptimizerState]:
    """Create momentum-less KFAC update step function.

    Args:
        optimizer (kfac_jax.Optimizer): instance of the Optimizer class from
            kfac_jax
        damping (chex.Numeric): damping coefficient
        get_position_fn (GetPositionFromData): function which gets the walker positions
            from the data. Has signature data -> Array
        update_data_fn (Callable): function which updates data for new params

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
        update_data_fn = utils.distribute.pmap(update_data_fn)

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
        variance = stats["aux"]["variance"]
        energy_noclip = stats["aux"]["energy_noclip"]
        variance_noclip = stats["aux"]["variance_noclip"]
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
    local_energy_fn: LocalEnergyApply[P],
    nchains: int,
    get_position_fn: GetPositionFromData[D],
    apply_pmap: bool = True,
    record_local_energies: bool = True,
    nan_safe: bool = False,
    use_PRNGKey: bool = False,
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
        positions = get_position_fn(data)

        if use_PRNGKey:
            nbatch = positions.shape[0]
            key = jax.random.split(key, nbatch)
            local_energies = jax.vmap(
                local_energy_fn, in_axes=(None, 0, 0), out_axes=0
            )(params, positions, key)
        else:
            local_energies = jax.vmap(
                local_energy_fn, in_axes=(None, 0, None), out_axes=0
            )(params, get_position_fn(data), None)

        energy, variance = physics.core.get_statistics_from_local_energy(
            local_energies, nchains, nan_safe=nan_safe
        )
        metrics = {"energy": energy, "variance": variance}
        if record_local_energies:
            metrics.update({"local_energies": local_energies})
        return params, data, optimizer_state, metrics, key

    traced_fn = _make_traced_fn_with_single_metrics(
        eval_update_param_fn, apply_pmap, {"energy", "variance"}
    )

    return traced_fn
