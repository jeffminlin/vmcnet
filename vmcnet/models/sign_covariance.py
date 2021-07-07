"""Routines for symmetrizing a function to be sign covariant."""
import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .core import is_tuple_of_arrays
from vmcnet.utils.typing import SLArray, SLArrayList
from vmcnet.utils.slog_helpers import slog_multiply, slog_linear_comb


def get_sign_orbit_one_sl_array(
    x: SLArray, i: int, n_total: int, axis: int = 0
) -> Tuple[SLArray, jnp.ndarray]:
    """Generates the sign orbit of a single SLArray within a larger SLArrayList.

    Args:
        x (SLArray): input data to be symmetrized
        i (int): the index of this data in the SLArrayList. This is used to
            decide how often the sign should flip as the orbit is generated. For
            i = 0, the sign will flip every time; for i = 1, it will flip every other
            time; and so forth, flipping every 2**i times in general. This ensures that
            when this operation is applied separately to each element of an SLArrayList,
            the overall result is to generate the full orbit of the SLArrayList with
            respect to the sign of each SLArray entry.
        n_total (int): the total number of SLArrays involved. There will always be
            2**n_total symmetries generated, regardless of i.
        axis (int): the axis along which to insert the symmetries. For example, if axis
            is -2, n_total is 5, and the shape of x is (3,2,4), the output shape will be
            (3,2,32,4).

    Returns:
        (Tuple[SLArray, jnp.ndarray]): First entry is the resulting orbit of x as an
            SLogArray. This will look something like [x, x, -x, -x, x, x, ...]
            (if i=2), but stacked along the specified axis. Second entry is the
            associated signs as a single array, i.e. [1, 1, -1, -1, 1, 1, ...]. This is
            returned so that downstream functions don't have to recompute it.
    """
    nsyms = 2 ** n_total
    sym_ints = jnp.arange(nsyms)

    # The exponent below is simply a clever way to get the value of the i_spin-th bit
    # of each integer in the sym_int array, to give us signs which alternate every
    # 2**i_spin values.
    sym_signs_1d = (-1) ** jnp.sign((2 ** i) & sym_ints)

    # Reshape signs to be of shape (1, ..., 1, nsyms, 1, ..., 1) where the nsyms value
    # is at the specified axis and the total number of axes is one more than the input.
    sign_shape = [1 for _ in range(len(x[0].shape) + 1)]
    sign_shape[axis] = nsyms
    sym_signs = jnp.reshape(sym_signs_1d, sign_shape)
    sym_logs = jnp.zeros(sign_shape)

    x = jax.tree_map(lambda a: jnp.expand_dims(a, axis), x)

    return slog_multiply(x, (sym_signs, sym_logs)), sym_signs_1d


def get_sign_orbit_slog_array_list(
    x: SLArrayList, axis: int = 0
) -> Tuple[SLArrayList, jnp.ndarray]:
    """Generates the orbit of an SLArrayList w.r.t the sign group on each array.

    For example, if the input is (s_1, s_2, s_3), where each s_i is the SLArray for
    index i, the generated symmetries will be
    (s_1, s_2, s_3), (-s_1, s_2, s_3), (s_1, -s_2, s_3), (-s_1, -s_2, s_3),
    (s_1, s_2, -s_3), (-s_1, s_2, -s_3), (s_1, -s_2, -s_3), (-s_1, -s_2, -s_3).

    Also returns the associated overall sign of each symmetry, as a single 1D array. For
    this example, that would be [1, -1, -1, 1, -1, 1, 1, -1]. This is calculated by
    multiplying together the individual sign arrays for each SLArray.

    Args:
        x (SLArrayList): the original data, as a list of slog arrays.
        axis (int): the location of the new axis which will index the generated
            symmetries.

    Returns:
        (Tuple[SpinSplitSLArray, jnp.ndarray]): First entry is a new list of slog
        arrays that contains the orbit of the input values with respect to the sign
        group, applied separately to each SLArray entry in the list. Second entry is the
        associated signs of each symmetry in the orbit, as a single 1D array.

    """
    n_total = len(x)
    i_list = [i for i in range(n_total)]
    get_sign_symmetries = functools.partial(
        get_sign_orbit_one_sl_array, n_total=n_total, axis=axis
    )
    syms_and_signs = jax.tree_map(
        get_sign_symmetries, x, i_list, is_leaf=is_tuple_of_arrays
    )
    syms = [x[0] for x in syms_and_signs]
    stacked_signs = jnp.stack([x[1] for x in syms_and_signs])
    combined_signs = jnp.prod(stacked_signs, axis=0)
    return syms, combined_signs


def make_slog_fn_sign_covariant(
    fn: Callable[[SLArrayList], SLArray],
) -> Callable[[SLArrayList], SLArray]:
    """Make a function of an SLArrayList covariant in the sign of each input array.

    That is, output a function g(s_1, s_2, ..., s_n) with is odd with respect to each
    input, such that g(s_1, ...) = -g(-s_1, ...), and likewise for every other input.

    This is done by taking the orbit of the inputs with respect to the sign group
    applied separately to each one, and adding up the results with appropriate
    covariant signs. For example, for two spins this calculates
    g(U,D) = f(U,D) - f(-U,D) - f(U, -D) + f(-U, -D).

    As currently implemented, this method assumes the output of the function is a single
    SLArray, and that the function only applies to the last two axes of its inputs, so
    that the previous axes can all be treated as batch dimensions. Essentially, the
    function must be a map from inputs of shape (..., n[i], d) to outputs of shape
    (..., d'). This allows us to insert the orbit symmetries along axis -3 and then
    remove the extra dimension by summing over the inserted axis at the end, which
    considerably simplifies the implementation relative to handling a more general case.

    Args:
        fn (Callable): the function to symmetrize, which takes an input an SLArrayList
            with leaves of shape (..., n[i], d), and outputs a single SLArray of
            shape (..., d').

    Returns:
        fn (Callable): a function with the same signature as the input function, but
            which has been symmetrized so that its output will be covariant with respect
            to the sign of each input, or in other words, will be odd.
    """
    # x is a SLArrayList of length n_total, with leaves of shape (..., n[i], d)
    def sign_covariant_fn(x: SLArrayList):
        # Symmetries leaves are of shape (..., 2**n_total, n[i], d)
        # Signs is single array of shape (2**n_total)
        symmetries, signs = get_sign_orbit_slog_array_list(x, axis=-3)
        # all_results is of shape (..., 2**n_total, d')
        all_results = fn(symmetries)
        shaped_signs = jnp.expand_dims(signs, axis=-1)
        signed_results = (all_results[0] * shaped_signs, all_results[1])

        # Return final results collapsed to shape (..., d')
        return slog_linear_comb(signed_results, axis=-2)

    return sign_covariant_fn
