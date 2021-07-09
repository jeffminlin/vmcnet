"""Helper functions for model tests."""
from typing import Sequence, Optional, Tuple

import jax
import jax.numpy as jnp

import vmcnet.models as models


def get_elec_hyperparams() -> Tuple[
    int,
    int,
    int,
    int,
    Sequence[int],
    Sequence[int],
    Sequence[Sequence[int]],
]:
    """Get hyperparameters for electron data."""
    nchains = 25
    nelec_total = 7
    nion = 3
    d = 3
    permutation = (1, 0, 2, 5, 6, 3, 4)
    spin_split = (3,)
    split_perm = [(1, 0, 2), (2, 3, 0, 1)]

    return nchains, nelec_total, nion, d, permutation, spin_split, split_perm


def get_input_streams_from_hyperparams(
    nchains: int, nelec_total: int, nion: int, d: int, permutation: Sequence[int]
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    jnp.ndarray,
]:
    """Get electron input data given several hyperparameters."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    elec_pos = jax.random.normal(subkey, (nchains, nelec_total, d))
    permuted_elec_pos = elec_pos[:, permutation, :]
    key, subkey = jax.random.split(key)
    ion_pos = jax.random.normal(subkey, (nion, d))

    input_1e, input_2e, input_ei = models.equivariance.compute_input_streams(
        elec_pos, ion_pos
    )
    (
        perm_input_1e,
        perm_input_2e,
        perm_input_ei,
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
