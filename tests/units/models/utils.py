"""Helper functions for model tests."""
from typing import Optional, List, Tuple

import jax
import jax.numpy as jnp

import vmcnet.models as models
from vmcnet.utils.typing import Array, PRNGKey


def get_elec_hyperparams() -> (
    Tuple[
        int,
        int,
        int,
        int,
        Tuple[int, ...],
        Tuple[int, ...],
        List[Tuple[int, ...]],
    ]
):
    """Get hyperparameters for electron data."""
    nchains = 25
    nelec_total = 7
    nion = 3
    d = 3
    permutation = (1, 0, 2, 5, 6, 3, 4)
    spin_split = (3,)
    split_perm = [(1, 0, 2), (2, 3, 0, 1)]

    return nchains, nelec_total, nion, d, permutation, spin_split, split_perm


def get_elec_and_ion_pos_from_hyperparams(
    nchains: int, nelec_total: int, nion: int, d: int, permutation: Tuple[int, ...]
) -> Tuple[PRNGKey, Array, Array, Optional[Array]]:
    """Get electron, permuted electron, and ion positions from hyperparameters."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    elec_pos = jax.random.normal(subkey, (nchains, nelec_total, d))
    permuted_elec_pos = elec_pos[:, permutation, :]
    key, subkey = jax.random.split(key)
    if nion > 0:
        ion_pos: Optional[Array] = jax.random.normal(subkey, (nion, d))
    else:
        ion_pos = None
    return key, elec_pos, permuted_elec_pos, ion_pos


def get_input_streams_from_hyperparams(
    nchains: int, nelec_total: int, nion: int, d: int, permutation: Tuple[int, ...]
) -> Tuple[
    Array,
    Optional[Array],
    Optional[Array],
    Array,
    Optional[Array],
    Optional[Array],
    PRNGKey,
]:
    """Get electron and permuted electron input streams given hyperparameters."""
    key, elec_pos, permuted_elec_pos, ion_pos = get_elec_and_ion_pos_from_hyperparams(
        nchains, nelec_total, nion, d, permutation
    )

    input_1e, input_2e, input_ei, _ = models.equivariance.compute_input_streams(
        elec_pos, ion_pos
    )
    (
        perm_input_1e,
        perm_input_2e,
        perm_input_ei,
        _,
    ) = models.equivariance.compute_input_streams(permuted_elec_pos, ion_pos)
    return (
        input_1e,
        input_2e,
        input_ei,
        perm_input_1e,
        perm_input_2e,
        perm_input_ei,
        key,
    )


def simple_backflow(stream_1e, _stream_2e):
    """Simple equivariance with linear and quadratic features."""
    return jnp.concatenate([2.0 * stream_1e, jnp.square(stream_1e)], axis=-1)
