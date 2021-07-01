"""Routines for dealing with odd symmetrization of a function."""
import functools
from typing import Callable

import jax
import jax.numpy as jnp

from .core import is_tuple_of_arrays
from vmcnet.utils.typing import SLArray, SpinSplitSLArray
from vmcnet.utils.slog_helpers import slog_multiply, slog_sum


# x leaves are shape (..., nelec[i], d),
# output leaves are shape (..., 2**nspins, nelec[i], d)
def get_odd_symmetries_one_spin(
    x: SLArray, i_spin: int, nspins: int, axis: int = 0
) -> SLArray:
    """Generates odd symmetries of single spin part of slog array."""
    nsyms = 2 ** nspins
    sym_ints = jnp.arange(nsyms)
    sym_signs = (-1) ** jnp.sign((2 ** i_spin) & sym_ints)

    naxes_out = len(x[0].shape) + 1
    sign_shape = tuple(
        jax.ops.index_update(jnp.ones(naxes_out, dtype=int), axis, nsyms)
    )

    sym_signs = jnp.reshape(sym_signs, sign_shape)
    sym_logs = jnp.zeros(sign_shape)

    x = jax.tree_map(lambda a: jnp.expand_dims(a, axis), x)

    return slog_multiply(x, (sym_signs, sym_logs))


# x is shape [((..., nelec[i], d),(..., nelec[i], d))]
# output is shape [((..., 2**nspins, nelec[i], d),(..., 2**nspins, nelec[i], d))]
def get_all_odd_symmetries(x: SpinSplitSLArray) -> SpinSplitSLArray:
    """Generates odd symmetries of spin-split slog array."""
    nspins = len(x)
    i_spin_list = [i for i in range(nspins)]
    get_odd_symmetries = functools.partial(
        get_odd_symmetries_one_spin, nspins=nspins, axis=-3
    )
    return jax.tree_map(get_odd_symmetries, x, i_spin_list, is_leaf=is_tuple_of_arrays)


def get_odd_output_signs(nspins: int) -> jnp.ndarray:
    """Gets odd signs."""
    nsyms = 2 ** nspins
    sym_ints = jnp.expand_dims(jnp.arange(nsyms), axis=-1)
    i_spins = jnp.expand_dims(jnp.arange(nspins), axis=0)
    signs_per_spin = (-1) ** jnp.sign((2 ** i_spins) & sym_ints)
    return jnp.product(signs_per_spin, axis=-1)


# f is SpinSplitArray
def make_fn_odd(
    fn: Callable[[SpinSplitSLArray], SLArray]
) -> Callable[[SpinSplitSLArray], SLArray]:
    """Make a fn odd."""
    # x leaves are shape (..., nelec[i], d)
    def odd_fn(x: SpinSplitSLArray):
        nspins = len(x)
        # symmetries leaves are shape (..., 2**nspins, nelec[i], d)
        symmetries = get_all_odd_symmetries(x)
        # All results are shape (..., 2**nspins, d')
        all_results = fn(symmetries)
        odd_signs = jnp.expand_dims(get_odd_output_signs(nspins), axis=-1)
        signed_results = (all_results[0] * odd_signs, all_results[1])

        # Collapsed results are shape (..., d')
        collapsed_results = slog_sum(signed_results, axis=-2)
        return collapsed_results

    return odd_fn
