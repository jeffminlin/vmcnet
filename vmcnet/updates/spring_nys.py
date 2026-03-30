"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, PRNGKey, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer


def construct_spring_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)

        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state = optimizer_apply(
            energy,
            local_energies,
            params,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring_nys(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    key: PRNGKey,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    rank = optimizer_config.rank
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))

    key, subkey = jax.random.split(key)
    omega = jax.random.normal(subkey, (nparams, rank))

    spring_step = get_spring_step(
        log_psi_apply,
        omega,   
        optimizer_config.damping,
        optimizer_config.mu,
        optimizer_config.beta,
        optimizer_config.nu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        Y = optimizer_state[2]
        grad, Y = spring_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            positions,
            Y,
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state[:2], params
        )
        optimizer_state = [*optimizer_state[:2], Y]

        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = construct_spring_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    optimizer_state = [*optimizer_state, omega]

    return update_param_fn, optimizer_state, key


"""Get the SPRING update function."""
def get_spring_step(
    log_psi_apply: ModelApply[P],
    omega: Array,
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    beta = 0.995,
    nu = 1e-6,
):

    def raveled_log_psi_grad(params: P, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(raveled_log_psi_grad, in_axes=(None, 0))

    def nys_solve(Y, b):
        Ynu = Y + nu * omega

        C = jnp.linalg.cholesky(omega.T @ Ynu).T
        B = jax.scipy.linalg.solve_triangular(C.T, Ynu.T, lower=True).T
        R = B.T @ B

        eigs, _ = jnp.linalg.eigh(R)
        min_eig = jnp.min(eigs)
        R = R + min_eig * jnp.eye(R.shape[0])

        Rsolve = jax.scipy.linalg.solve(R, B.T @ b)
        BinvOhatT = 1/min_eig * b - 1/min_eig * B @ Rsolve
        return BinvOhatT
    
    def spring_step(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
        Y: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]

        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad

        log_psi_grads = batch_raveled_log_psi_grad(params, positions) / jnp.sqrt(
            nchains
        )
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)
        
        Y = beta * Y + (1 - beta) * Ohat.T @ (Ohat @ omega)
        BinvOhatT = nys_solve(Y, Ohat.T) 

        T = Ohat @ BinvOhatT
        ones = jnp.ones((nchains, 1))
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed

        dtheta_residual = BinvOhatT @ jax.scipy.linalg.solve(
            T_reg, epsion_tilde, assume_a="pos"
        )

        SR_G = dtheta_residual + prev_grad_decayed

        return unravel_fn(SR_G), Y

    return spring_step


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
