"""Streaming Nyström-preconditioned SPRING prototype."""

from typing import NamedTuple, Tuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from ml_collections import ConfigDict

from vmcnet.utils.pytree_helpers import tree_reduce_l1
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    P,
    PRNGKey,
    UpdateDataFn,
)

from .kfac_spring import get_centered_jacobian
from .optax_utils import initialize_optax_optimizer
from .spring import constrain_norm
from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)


class NystromPreconditionerState(NamedTuple):
    """State for the streaming low-rank Fisher approximation."""

    omega: Array
    y: Array
    step: Array


class SpringNystromState(NamedTuple):
    """Optimizer state for the streaming Nyström-SPRING prototype."""

    nystrom_state: NystromPreconditionerState
    descent_state: optax.OptState


class NystromDiagnostics(NamedTuple):
    """Scalar diagnostics for Nyström-SPRING logging."""

    phasein: Array
    metric_shift: Array
    metric_scale: Array
    trace: Array
    max_eigenvalue: Array
    min_eigenvalue: Array
    effective_rank: Array
    clipped_eigenvalues: Array
    kernel_trace: Array


class NystromEigendecomposition(NamedTuple):
    """Low-rank eigendecomposition recovered from a Nyström sketch."""

    eigenvectors: Array
    eigenvalues: Array
    c_eigenvalues: Array
    clipped_count: Array


class SpringLinearSolveResult(NamedTuple):
    """Intermediate quantities from the preconditioned SPRING solve."""

    flat_direction: Array
    preconditioned_kernel: Array
    regularized_kernel: Array
    residual: Array
    zeta: Array
    centered_zeta: Array


def initialize_nystrom_state(
    params: P,
    key: PRNGKey,
    rank: int,
) -> NystromPreconditionerState:
    """Initialize a fixed random Nyström sketch matrix."""
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    parameter_count = flat_params.shape[0]
    sketch_rank = min(max(rank, 1), parameter_count)
    omega = jax.random.normal(key, (parameter_count, sketch_rank))
    return NystromPreconditionerState(
        omega=omega,
        y=jnp.zeros_like(omega),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def get_phasein_coefficient(
    step: Array,
    warmup_steps: int,
    phasein_steps: int,
) -> Array:
    """Return the identity-to-Nyström phase-in coefficient."""
    step = step.astype(jnp.float32)
    warmup_steps = jnp.asarray(warmup_steps, dtype=jnp.float32)
    if phasein_steps <= 0:
        return jnp.where(step >= warmup_steps, 1.0, 0.0)

    phasein_steps = jnp.asarray(phasein_steps, dtype=jnp.float32)
    return jnp.clip((step - warmup_steps) / phasein_steps, 0.0, 1.0)


def update_nystrom_state(
    state: NystromPreconditionerState,
    scaled_jacobian: Array,
    ema_decay: chex.Numeric,
    warmup_steps: int,
    collect_during_warmup: bool,
) -> NystromPreconditionerState:
    """Update the streaming Nyström statistics from one centered Jacobian."""
    sketch_product = scaled_jacobian.T @ (scaled_jacobian @ state.omega)
    updated_y = ema_decay * state.y + (1.0 - ema_decay) * sketch_product
    if collect_during_warmup:
        y = updated_y
    else:
        y = jnp.where(state.step >= warmup_steps, updated_y, state.y)
    return NystromPreconditionerState(state.omega, y, state.step + 1)


def get_nystrom_eigendecomposition(
    state: NystromPreconditionerState,
    eigenvalue_floor: chex.Numeric,
) -> NystromEigendecomposition:
    """Return eigenvectors/eigenvalues of the low-rank Nyström approximation."""
    return get_nystrom_eigendecomposition_from_sketch(
        state.omega, state.y, eigenvalue_floor
    )


def get_nystrom_eigendecomposition_from_sketch(
    omega: Array,
    y_matrix: Array,
    eigenvalue_floor: chex.Numeric,
) -> NystromEigendecomposition:
    """Return eigenpairs for the Nyström matrix ``Y (Omega^T Y)^-1 Y^T``."""
    c_matrix = omega.T @ y_matrix
    c_matrix = c_matrix + eigenvalue_floor * jnp.eye(c_matrix.shape[0])
    chol = jnp.linalg.cholesky(c_matrix)
    z_matrix = jnp.linalg.solve(chol, y_matrix.T).T
    eigenvectors, singular_values, _ = jnp.linalg.svd(
        z_matrix, full_matrices=False
    )
    eigenvalues = jnp.square(singular_values)
    c_eigenvalues = jnp.linalg.eigvalsh((c_matrix + c_matrix.T) / 2)
    clipped_count = jnp.asarray(0)
    return NystromEigendecomposition(
        eigenvectors, eigenvalues, c_eigenvalues, clipped_count
    )


def reconstruct_nystrom_matrix(
    eigenvectors: Array,
    eigenvalues: Array,
) -> Array:
    """Materialize a low-rank Nyström approximation for small tests."""
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def get_effective_rank(eigenvalues: Array) -> Array:
    """Return the participation-ratio effective rank of a PSD spectrum."""
    trace = jnp.sum(eigenvalues)
    squared_trace = jnp.sum(jnp.square(eigenvalues))
    return jnp.where(
        squared_trace > 0.0,
        trace * trace / squared_trace,
        0.0,
    )


def get_nystrom_metric_scale(
    eigenvalues: Array,
    normalization: str,
    eigenvalue_floor: chex.Numeric,
) -> Array:
    """Return the scalar used to normalize the Nyström metric eigenvalues."""
    if normalization == "none":
        return jnp.asarray(1.0)
    if normalization == "trace_effective_rank":
        trace = jnp.sum(eigenvalues)
        squared_trace = jnp.sum(jnp.square(eigenvalues))
        scale = squared_trace / jnp.maximum(trace, eigenvalue_floor)
        scale = jnp.maximum(scale, eigenvalue_floor)
        return jnp.where(trace > eigenvalue_floor, scale, 1.0)
    raise ValueError(f"Unsupported Nyström metric normalization: {normalization}")


def normalize_nystrom_eigenvalues(
    eigenvalues: Array,
    normalization: str,
    eigenvalue_floor: chex.Numeric,
) -> Tuple[Array, Array]:
    """Normalize Nyström metric eigenvalues and return ``(values, scale)``."""
    metric_scale = get_nystrom_metric_scale(
        eigenvalues, normalization, eigenvalue_floor
    )
    return eigenvalues / metric_scale, metric_scale


def apply_nystrom_inverse_to_flat_vector(
    flat_vector: Array,
    eigenvectors: Array,
    eigenvalues: Array,
    phasein: chex.Numeric,
    metric_shift: chex.Numeric,
) -> Array:
    """Apply the inverse of the phased low-rank-plus-identity metric."""
    diagonal_weight = (1.0 - phasein) + phasein * metric_shift
    projected = eigenvectors.T @ flat_vector
    inverse_correction = (1.0 / (diagonal_weight + phasein * eigenvalues)) - (
        1.0 / diagonal_weight
    )
    return flat_vector / diagonal_weight + eigenvectors @ (
        inverse_correction * projected
    )


def get_metric_shift(
    eigenvalues: Array,
    constant_shift: chex.Numeric,
    shift_strategy: str,
    eigenvalue_floor: chex.Numeric,
) -> Array:
    """Return the identity shift used for the unrepresented eigenspace."""
    smallest_sketch_eigenvalue = jnp.maximum(jnp.min(eigenvalues), eigenvalue_floor)
    if shift_strategy == "constant":
        return jnp.asarray(constant_shift)
    if shift_strategy == "min_eigenvalue":
        return smallest_sketch_eigenvalue
    if shift_strategy == "min_eigenvalue_plus_constant":
        return smallest_sketch_eigenvalue + constant_shift
    raise ValueError(f"Unsupported Nyström shift strategy: {shift_strategy}")


def get_nystrom_metric_parameters(
    eigenvalues: Array,
    metric_identity_shift: chex.Numeric,
    metric_shift_strategy: str,
    sketch_damping: chex.Numeric,
    eigenvalue_floor: chex.Numeric,
) -> Tuple[Array, Array, Array]:
    """Return low-rank eigenvalues, identity shift, and reported metric scale.

    ``regularization_coupled`` constructs the metric

    ``B = I + F_hat / sketch_damping``

    so the same regularization scale is used in the parameter-space metric and
    the walker-space minSR solve. Other strategies retain the original
    ``B = delta I + F_hat`` behavior.
    """
    if metric_shift_strategy == "regularization_coupled":
        damping = jnp.maximum(jnp.asarray(sketch_damping), eigenvalue_floor)
        return eigenvalues / damping, jnp.asarray(1.0), damping

    metric_shift = get_metric_shift(
        eigenvalues,
        metric_identity_shift,
        metric_shift_strategy,
        eigenvalue_floor,
    )
    return eigenvalues, metric_shift, jnp.asarray(1.0)


def solve_preconditioned_spring_system(
    scaled_jacobian: Array,
    centered_energies: Array,
    flat_mu_previous: Array,
    apply_inverse,
    sketch_damping: chex.Numeric,
) -> SpringLinearSolveResult:
    """Solve the walker-space SPRING system for a generic inverse metric."""
    nchains = scaled_jacobian.shape[0]
    preconditioned_jacobian_t = jax.vmap(apply_inverse, in_axes=1, out_axes=1)(
        scaled_jacobian.T
    )
    preconditioned_kernel = scaled_jacobian @ preconditioned_jacobian_t
    preconditioned_kernel = (preconditioned_kernel + preconditioned_kernel.T) / 2

    kernel_eigenvalues, kernel_eigenvectors = jnp.linalg.eigh(preconditioned_kernel)
    kernel_eigenvalues = jnp.maximum(kernel_eigenvalues, 0) + sketch_damping
    regularized_kernel = (kernel_eigenvectors * kernel_eigenvalues) @ (
        kernel_eigenvectors.T
    )

    residual = centered_energies / jnp.sqrt(nchains)
    residual = residual - scaled_jacobian @ flat_mu_previous

    zeta = kernel_eigenvectors @ (
        (kernel_eigenvectors.T @ residual) / kernel_eigenvalues
    )
    centered_zeta = zeta - jnp.mean(zeta)
    flat_direction = preconditioned_jacobian_t @ centered_zeta
    return SpringLinearSolveResult(
        flat_direction,
        preconditioned_kernel,
        regularized_kernel,
        residual,
        zeta,
        centered_zeta,
    )


def get_spring_nystrom_step(
    log_psi_apply: ModelApply[P],
    sketch_damping: chex.Scalar,
    metric_identity_shift: chex.Scalar,
    metric_shift_strategy: str,
    nystrom_ema_decay: chex.Scalar,
    nystrom_warmup_steps: int,
    nystrom_phasein_steps: int,
    eigenvalue_floor: chex.Scalar,
    collect_during_warmup: bool,
    metric_normalization: str = "none",
    mu: chex.Scalar = 0.99,
):
    """Get a streaming Nyström-preconditioned SPRING step function."""

    def spring_nystrom_step(
        centered_energies: Array,
        params: P,
        previous_direction: P,
        positions: Array,
        nystrom_state: NystromPreconditionerState,
    ) -> Tuple[P, NystromPreconditionerState, NystromDiagnostics]:
        nchains = positions.shape[0]
        jacobian, unravel_fn = get_centered_jacobian(log_psi_apply, params, positions)
        scaled_jacobian = jacobian / jnp.sqrt(nchains)

        nystrom_state = update_nystrom_state(
            nystrom_state,
            scaled_jacobian,
            nystrom_ema_decay,
            nystrom_warmup_steps,
            collect_during_warmup,
        )
        phasein = get_phasein_coefficient(
            nystrom_state.step - 1,
            nystrom_warmup_steps,
            nystrom_phasein_steps,
        )
        nystrom_eigendecomposition = get_nystrom_eigendecomposition(
            nystrom_state, eigenvalue_floor
        )
        eigenvectors = nystrom_eigendecomposition.eigenvectors
        raw_eigenvalues = nystrom_eigendecomposition.eigenvalues
        eigenvalues, metric_scale = normalize_nystrom_eigenvalues(
            raw_eigenvalues,
            metric_normalization,
            eigenvalue_floor,
        )
        metric_eigenvalues, metric_shift, coupled_metric_scale = (
            get_nystrom_metric_parameters(
                eigenvalues,
                metric_identity_shift,
                metric_shift_strategy,
                sketch_damping,
                eigenvalue_floor,
            )
        )
        metric_scale = metric_scale * coupled_metric_scale

        apply_inverse = lambda flat_vector: apply_nystrom_inverse_to_flat_vector(
            flat_vector,
            eigenvectors,
            metric_eigenvalues,
            phasein,
            metric_shift,
        )

        mu_previous = jax.tree_map(lambda x: mu * x, previous_direction)
        flat_mu_previous, _ = jax.flatten_util.ravel_pytree(mu_previous)
        solve_result = solve_preconditioned_spring_system(
            scaled_jacobian,
            centered_energies,
            flat_mu_previous,
            apply_inverse,
            sketch_damping,
        )

        direction = unravel_fn(solve_result.flat_direction)
        direction = jax.tree_map(
            lambda dtheta, previous: dtheta + previous, direction, mu_previous
        )

        trace = jnp.sum(eigenvalues)
        effective_rank = get_effective_rank(eigenvalues)
        diagnostics = NystromDiagnostics(
            phasein=phasein,
            metric_shift=metric_shift,
            metric_scale=metric_scale,
            trace=trace,
            max_eigenvalue=jnp.max(eigenvalues),
            min_eigenvalue=jnp.min(eigenvalues),
            effective_rank=effective_rank,
            clipped_eigenvalues=(
                nystrom_eigendecomposition.clipped_count
                + jnp.sum(nystrom_eigendecomposition.c_eigenvalues < eigenvalue_floor)
            ),
            kernel_trace=jnp.trace(solve_result.preconditioned_kernel),
        )
        return direction, nystrom_state, diagnostics

    return spring_nystrom_step


def construct_spring_nystrom_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = False,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, SpringNystromState]:
    """Create the Nyström-SPRING update function."""
    if apply_pmap:
        raise NotImplementedError("Nyström-SPRING prototype only supports single-device.")

    def update_param_fn(params, data, optimizer_state, key):
        positions = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, positions)
        centered_local_energies = local_energies - energy

        params, optimizer_state, diagnostics = optimizer_apply(
            centered_local_energies,
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
        metrics["nystrom_spring_phasein"] = diagnostics.phasein
        metrics["nystrom_spring_metric_shift"] = diagnostics.metric_shift
        metrics["nystrom_spring_metric_scale"] = diagnostics.metric_scale
        metrics["nystrom_spring_trace"] = diagnostics.trace
        metrics["nystrom_spring_max_eigenvalue"] = diagnostics.max_eigenvalue
        metrics["nystrom_spring_min_eigenvalue"] = diagnostics.min_eigenvalue
        metrics["nystrom_spring_effective_rank"] = diagnostics.effective_rank
        metrics["nystrom_spring_clipped_eigenvalues"] = diagnostics.clipped_eigenvalues
        metrics["nystrom_spring_kernel_trace"] = diagnostics.kernel_trace
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    return make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)


def initialize_spring_nystrom(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    key: PRNGKey,
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = False,
) -> Tuple[UpdateParamFn[P, D, SpringNystromState], SpringNystromState, PRNGKey]:
    """Initialize the streaming Nyström-preconditioned SPRING prototype."""
    if apply_pmap:
        raise NotImplementedError("Nyström-SPRING prototype only supports single-device.")
    if (
        optimizer_config.metric_shift_strategy == "regularization_coupled"
        and optimizer_config.metric_normalization != "none"
    ):
        raise ValueError(
            "regularization_coupled requires metric_normalization='none' so "
            "B = I + F_hat / sketch_damping"
        )

    key, subkey = jax.random.split(key)
    nystrom_state = initialize_nystrom_state(
        params,
        subkey,
        int(optimizer_config.nystrom_rank),
    )
    spring_nystrom_step = get_spring_nystrom_step(
        log_psi_apply,
        optimizer_config.sketch_damping,
        optimizer_config.metric_identity_shift,
        optimizer_config.metric_shift_strategy,
        optimizer_config.nystrom_ema_decay,
        optimizer_config.nystrom_warmup_steps,
        optimizer_config.nystrom_phasein_steps,
        optimizer_config.eigenvalue_floor,
        optimizer_config.collect_during_warmup,
        optimizer_config.metric_normalization,
        optimizer_config.mu,
    )
    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def previous_direction(optimizer_state):
        return optimizer_state.descent_state[0].trace

    def optimizer_apply(centered_local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)
        direction, nystrom_state, diagnostics = spring_nystrom_step(
            centered_local_energies,
            params,
            previous_direction(optimizer_state),
            positions,
            optimizer_state.nystrom_state,
        )
        updates, descent_state = descent_optimizer.update(
            direction, optimizer_state.descent_state, params
        )

        if optimizer_config.constrain_norm:
            updates = constrain_norm(updates, optimizer_config.norm_constraint)

        params = optax.apply_updates(params, updates)
        optimizer_state = SpringNystromState(nystrom_state, descent_state)
        return params, optimizer_state, diagnostics

    update_param_fn = construct_spring_nystrom_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    descent_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
    optimizer_state = SpringNystromState(nystrom_state, descent_state)

    return update_param_fn, optimizer_state, key
