"""Routines for dealing with odd symmetrization of a function."""
import functools
from typing import Callable

import jax
import jax.numpy as jnp

from .core import is_tuple_of_arrays
from vmcnet.utils.typing import SLArray, SpinSplitSLArray
from vmcnet.utils.slog_helpers import slog_multiply, slog_sum

## TODO (ggoldsh): should axis be a parameter for these so we don't generate the lists
# of symmetries over and over again? How to even test that?

# TODO (ggoldsh): what was actually wrong with my example before that was making it always
# output variations on zero?? Is that not concerning? Is it somehow saying odd(permutation invariant) = 0??


def get_odd_symmetries_one_spin(
    x: SLArray, i_spin: int, nspins: int, axis: int = 0
) -> SLArray:
    """Generates odd symmetries of a single SLArray within a larger SpinSplitSLArray.

    Args:
        x (SLArray): input data to be symmetrized
        i_spin (int): the index of this data in the SpinSplitSLArray. This is used to
            decide how often the sign should flip as the symmetries are generated. For
            i_spin = 1, the sign will flip every time; or i_spin = 2, it will flip every
            other time; and so forth. This ensures that when this symmetry is applied
            separately to each element of a SpinSplitSLArray, the overall result is to
            generate all of the symmetries of the full SpinSplitSLArray.
        nspins (int): the total number of spins involved. There will always be 2**nspins
            symmetries generated, regardless of i_spin.
        axis (int): the axis along which to insert the symmetries. For example, if axis
            is -2, nspins is 5, and the shape of x is (3,2,4), the output shape will be
            (3,2,32,4).

    Returns:
        (SLArray): all odd symmetries of x as an SLogArray. This will look something
            like [x, x, -x, -x, x, x, ...] (if ispin =2), but stacked along the
            specified axis.
    """
    nsyms = 2 ** nspins
    sym_ints = jnp.arange(nsyms)

    # The exponent below is simply a clever way to get the value of the i_spin-th bit
    # of each integer in the sym_int array, to give us signs which alternate every
    # 2**i_spin values.
    sym_signs = (-1) ** jnp.sign((2 ** i_spin) & sym_ints)

    # Reshape signs to be of shape (1, ..., 1, nsyms, 1, ..., 1) where the nsyms value
    # is at the specified axis and the total number of axes is one more than the input.
    sign_shape = [1 for _ in range(len(x[0].shape) + 1)]
    sign_shape[axis] = nsyms
    sym_signs = jnp.reshape(sym_signs, sign_shape)
    sym_logs = jnp.zeros(sign_shape)

    x = jax.tree_map(lambda a: jnp.expand_dims(a, axis), x)

    return slog_multiply(x, (sym_signs, sym_logs))


def get_all_odd_symmetries(x: SpinSplitSLArray, axis: int = 0) -> SpinSplitSLArray:
    """Generates all odd symmetries of a spin-split slog array.

    For example, if the input is (s_1, s_2, s_3), where each s_i is the SLArray for spin
    i, the generated symmetries will be
    (s_1, s_2, s_3), (-s_1, s_2, s_3), (s_1, -s_2, s_3), (-s_1, -s_2, s_3),
    (s_1, s_2, -s_3), (-s_1, s_2, -s_3), (s_1, -s_2, -s_3), (-s_1, -s_2, -s_3).

    Args:
        x (SpinSplitSLArray): the original data, as a spin-split slog array.
        axis (int): the location of the new axis which will index the generated
            symmetries.

    Returns:
        (SpinSplitSLArray): a new spin-split slog array that contains all the odd
            symmetries of the input values, inserted at the specified axis.
    """
    nspins = len(x)
    i_spin_list = [i for i in range(nspins)]
    get_odd_symmetries = functools.partial(
        get_odd_symmetries_one_spin, nspins=nspins, axis=axis
    )
    return jax.tree_map(get_odd_symmetries, x, i_spin_list, is_leaf=is_tuple_of_arrays)


def get_odd_output_signs(nspins: int) -> jnp.ndarray:
    """Gets signs to apply to the outputs of a fn that is being oddly symmetrized.

    For example for a function of two spins, f(U,D), the odd symmetrization is
    f(U,D) - f(-U,D) - f(U, -D) + f(-U, -D), so the returned signs should be
    [1, -1, -1, 1].

    Args:
        nspins (int): the number of spins over which the symmetry is being applied.
    """
    nsyms = 2 ** nspins

    # Generate integer indices for each symmetry and each spin. Expand them along
    # different axes so that we can use broadcasting to combine them in all combinations.
    sym_ints = jnp.expand_dims(jnp.arange(nsyms), axis=-1)
    i_spins = jnp.expand_dims(jnp.arange(nspins), axis=0)

    # The exponent below is a clever way to get, for each value of i_spin and each int
    # in the sym_ints array, the value of the i_spin-th bit the sym_int integer. This
    # gives us the per-spin signs we need, which we can multiply together along the
    # i_spin axis to get the final signs.
    signs_per_spin = (-1) ** jnp.sign((2 ** i_spins) & sym_ints)
    return jnp.product(signs_per_spin, axis=-1)


def make_fn_odd(
    fn: Callable[[SpinSplitSLArray], SLArray],
) -> Callable[[SpinSplitSLArray], SLArray]:
    """Make a function of a SpinSplitSLArray odd with respect to each spin.

    That is, output a function g(s_1, s_2, ..., s_n) such that
    g(s_1, ...) = -g(-s_1, ...), and likewise for every other spin.

    This is done by taking all signed symmetries of the inputs, applying the function
    to each one, and adding up the results. For example, for two spins this calculates
    g(U,D) = f(U,D) - f(-U,D) - f(U, -D) + f(-U, -D).

    As currently implemented, this method assumes the output of the function is a single
    SLArray, and that the function only applies to the last two axes of its inputs, so
    that the previous axes can all be treated as batch dimensions. Essentially, the
    function must be a map from inputs of shape (..., nelec[i], d) to outputs of shape
    (..., d'). This allows us to insert the symmetries along axis -3 and then remove
    the extra dimension by summing over the inserted axis at the end, which considerably
    simplifies the implementation relative to handling a more general case.

    Args:
        fn (Callable): the function to symmetrize, which takes an input SpinSplitSLArray
            with leaves of shape (..., nelec[i], d), and outputs a single SLArray of
            shape (..., d').

    Returns:
        fn (Callable): a function with the same signature as the input function, but
            which has been symmetrized so that its output will be odd with respect to
            each spin taken as a whole.
    """

    def odd_fn(x: SpinSplitSLArray):
        # x is a SpinSplitSLArray with leaves of shape (..., nelec[i], d)
        nspins = len(x)
        # Symmetries leaves are of shape (..., 2**nspins, nelec[i], d)
        symmetries = get_all_odd_symmetries(x, axis=-3)
        # all_results is of shape (..., 2**nspins, d')
        all_results = fn(symmetries)
        odd_signs = jnp.expand_dims(get_odd_output_signs(nspins), axis=-1)
        signed_results = (all_results[0] * odd_signs, all_results[1])

        # Return final results collapsed to shape (..., d')
        return slog_sum(signed_results, axis=-2)

    return odd_fn
