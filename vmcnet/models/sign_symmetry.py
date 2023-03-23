"""Routines for symmetrizing a function to be sign covariant."""
import functools
from typing import Callable, List, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp

from vmcnet.models.core import Dense, Module
from vmcnet.models.weights import WeightInitializer
from vmcnet.utils.slog_helpers import slog_multiply, slog_sum_over_axis
from vmcnet.utils.typing import Array, ArrayList, SLArray, SLArrayList

# TypeVar used for representing either an Array or a SLArray
A = TypeVar("A", Array, SLArray)


def _get_sign_array_1d(i: int, nsyms: int) -> Array:
    """Calculate array of nsyms signs that alternate every 2**i entries."""
    sym_ints = jnp.arange(nsyms)

    # The exponent below is simply a clever way to get the value of the i_spin-th bit
    # of each integer in the sym_int array, to give us signs which alternate every
    # 2**i_spin values.
    sym_signs_1d = (-1) ** jnp.sign((2**i) & sym_ints)

    return sym_signs_1d


def _reshape_sign_array_for_orbit(
    signs: Array, x_shape: Tuple[int, ...], axis: int, nsyms: int
) -> Array:
    """Reshape 1D sign array to shape (1, 1, ..., nsyms, 1, ..., 1).

    Resulting shape has one more axis than x and has dimension 1 on every axis except
    for the specified one, where the dimension is nsyms.
    """
    sign_shape = [1 for _ in range(len(x_shape) + 1)]
    sign_shape[axis] = nsyms
    return jnp.reshape(signs, sign_shape)


def _get_sign_orbit_single_array(
    x: Array, i: int, n_total: int, axis: int
) -> Tuple[Array, Array]:
    """Generates the sign orbit of a single array within a larger ArrayList.

    Args:
        x (Array): input data to be symmetrized.
        i (int): the index of this data in the ArrayList. This is used to
            decide how often the sign should flip as the orbit is generated. For
            i = 0, the sign will flip every time; for i = 1, it will flip every other
            time; and so forth, flipping every 2**i times in general. This ensures that
            when this operation is applied separately to each element of an ArrayList,
            the overall result is to generate the full orbit of the ArrayList with
            respect to the sign of each array entry.
        n_total (int): the total number of arrays involved. There will always be
            2**n_total symmetries generated, regardless of i.
        axis (int): the axis along which to insert the symmetries. For example, if axis
            is -2, n_total is 5, and the shape of x is (3,2,4), the output shape will be
            (3,2,32,4).

    Returns:
        (Array, Array): First entry is the resulting orbit of x as a
        jnp array. This will look something like [x, x, -x, -x, x, x, ...]
        (if i=2), but stacked along the specified axis. Second entry is the
        associated signs as a single array, i.e. [1, 1, -1, -1, 1, 1, ...]. This is
        returned so that downstream functions don't have to recompute it.
    """
    nsyms = 2**n_total
    sym_signs_1d = _get_sign_array_1d(i, nsyms)
    sym_signs_shaped = _reshape_sign_array_for_orbit(sym_signs_1d, x.shape, axis, nsyms)
    return sym_signs_shaped * jnp.expand_dims(x, axis), sym_signs_1d


def _get_sign_orbit_single_sl_array(
    x: SLArray, i: int, n_total: int, axis: int
) -> Tuple[SLArray, Array]:
    """Generates the sign orbit of a single SLArray within a larger SLArrayList.

    Args:
        x (SLArray): input data to be symmetrized.
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
        (SLArray, Array): First entry is the resulting orbit of x as an
        SLArray. This will look something like [x, x, -x, -x, x, x, ...]
        (if i=2), but stacked along the specified axis. Second entry is the
        associated signs as a single array, i.e. [1, 1, -1, -1, 1, 1, ...]. This is
        returned so that downstream functions don't have to recompute it.
    """
    nsyms = 2**n_total
    sym_signs_1d = _get_sign_array_1d(i, nsyms)
    sym_signs_shaped = _reshape_sign_array_for_orbit(
        sym_signs_1d, x[0].shape, axis, nsyms
    )

    sym_logs = jnp.zeros_like(sym_signs_shaped)
    x = jax.tree_map(lambda a: jnp.expand_dims(a, axis), x)
    return slog_multiply(x, (sym_signs_shaped, sym_logs)), sym_signs_1d


def _get_sign_orbit_for_list(
    x: List[A],
    get_one_orbit_fn: Callable[[A, int, int, int], Tuple[A, Array]],
    axis: int,
) -> Tuple[List[A], Array]:
    """Generates the orbit of a list of inputs w.r.t the sign group on each input.

    Inputs are assumed to be either Arrays or SLArrays, so that in either case the
    resulting symmetries can be stacked along a new axis of the underlying array values.

    For example, if the input is (s_1, s_2, s_3), the generated symmetries will be
    (s_1, s_2, s_3), (-s_1, s_2, s_3), (s_1, -s_2, s_3), (-s_1, -s_2, s_3),
    (s_1, s_2, -s_3), (-s_1, s_2, -s_3), (s_1, -s_2, -s_3), (-s_1, -s_2, -s_3).

    Also returns the associated overall sign of each symmetry, as a single 1D array. For
    this example, that would be [1, -1, -1, 1, -1, 1, 1, -1]. This is calculated by
    multiplying together the individual sign arrays for each input.

    Args:
        x (List[Array or SLArray]): the original data, as a list of either arrays
            or SLArrays.
        get_one_orbit_fn (Callable): function for getting the orbit of a
            single input in the input list. Signature should be
            (x_in, i, n_total, axis) -> (x_syms, signs).
        axis (int): the location of the new axis which will index the generated
            symmetries.

    Returns:
        (List[Array or SLArray], Array): First entry is a new list of arrays
        or SLArrays that contains the orbit of the input values with respect to the sign
        group, applied separately to each input entry. Second entry is the associated
        signs of each symmetry in the orbit, as a single 1D array.
    """
    n_total = len(x)
    syms_and_signs = [get_one_orbit_fn(x[i], i, n_total, axis) for i in range(n_total)]
    syms = [x[0] for x in syms_and_signs]
    stacked_signs = jnp.stack([x[1] for x in syms_and_signs])
    combined_signs = jnp.prod(stacked_signs, axis=0)
    return syms, combined_signs


def _get_sign_orbit_array_list(x: ArrayList, axis: int = 0) -> Tuple[ArrayList, Array]:
    """Get sign orbit for an ArrayList."""
    return _get_sign_orbit_for_list(x, _get_sign_orbit_single_array, axis)


def _get_sign_orbit_sl_array_list(
    x: SLArrayList, axis: int = 0
) -> Tuple[SLArrayList, Array]:
    """Get sign orbit for an SLArrayList."""
    return _get_sign_orbit_for_list(x, _get_sign_orbit_single_sl_array, axis)


def apply_sign_symmetry_to_fn(
    fn: Callable[[List[A]], A],
    get_signs_and_syms: Callable[[List[A]], Tuple[List[A], Array]],
    apply_output_signs: Callable[[A, Array], A],
    add_up_results: Callable[[A], A],
) -> Callable[[List[A]], A]:
    """Make a function of a list of inputs covariant in the sign of each input.

    That is, output a function g(s_1, s_2, ..., s_n) with is odd with respect to each
    input, such that g(s_1, ...) = -g(-s_1, ...), and likewise for every other input.

    This is done by taking the orbit of the inputs with respect to the sign group
    applied separately to each one, and adding up the results with appropriate
    covariant signs. For example, for two spins this calculates
    g(U,D) = f(U,D) - f(-U,D) - f(U, -D) + f(-U, -D).

    Inputs are assumed to be either Arrays or SLArrays, so that in either case the
    required symmetries can be stacked along a new axis of the underlying array values.
    The function `fn` is assumed to support the injection of a batch dimension, done in
    `get_signs_and_syms`, and pass it through to the output (e.g., a function which
    flattens the input would not be supported). The extra dimension is removed at the
    end via `add_up_results`.

    Args:
        fn (Callable): the function to symmetrize. The given axis is injected into the
            inputs and the sign orbit is computed, so this function should be able to
            treat the given sign orbit axis as a batch dimension, and the overall tensor
            rank should not change (len(input.shape) == len(output.shape))
        get_signs_and_syms (Callable): a function which gets the signs and symmetries
            for the input array. Returns a tuple of the symmetries plus the associated
            signs as a 1D array.
        apply_output_signs (Callable): function for applying signs to the outputs of
            the symmetrized function. For example, if the outputs are Arrays, this
            would simply multiply the arrays by the signs along the appropriate axis.
        add_up_results (Callable): function for combining the signed outputs into a
            single, sign-covariant output. For example, simple addition for Arrays
            or the slog_sum function for SLArrays.

    Returns:
        Callable: a function with the same signature as the input function, but
        which has been symmetrized so that its output will be covariant with respect
        to the sign of each input, or in other words, will be odd.
    """

    def sign_covariant_fn(x: List[A]) -> A:
        symmetries, signs = get_signs_and_syms(x)
        outputs = fn(symmetries)
        signed_results = apply_output_signs(outputs, signs)
        return add_up_results(signed_results)

    return sign_covariant_fn


def _multiply_sign_along_axis(x, s, axis):
    return jnp.swapaxes(s * jnp.swapaxes(x, axis, -1), axis, -1)


def make_sl_array_list_fn_sign_covariant(
    fn: Callable[[SLArrayList], SLArray], axis: int = -2
) -> Callable[[SLArrayList], SLArray]:
    """Make a function of an SLArrayList sign-covariant in the sign of each SLArray.

    Shallow wrapper around the generic apply_sign_symmetry_to_fn.

    Args:
        fn (Callable): the function to symmetrize. The given axis is injected into the
            inputs and the sign orbit is computed, so this function should be able to
            treat the given sign orbit axis as a batch dimension, and the overall tensor
            rank should not change (len(input.shape) == len(output.shape))

    Returns:
        Callable: a function with the same signature as the input function, but
        which has been symmetrized so that its output will be covariant with respect
        to the sign of each input, or in other words, will be odd.
    """
    return apply_sign_symmetry_to_fn(
        fn,
        functools.partial(_get_sign_orbit_sl_array_list, axis=axis),
        lambda x, s: (_multiply_sign_along_axis(x[0], s, axis), x[1]),
        functools.partial(slog_sum_over_axis, axis=axis),
    )


def make_array_list_fn_sign_covariant(
    fn: Callable[[ArrayList], Array], axis: int = -2
) -> Callable[[ArrayList], Array]:
    """Make a function of an ArrayList sign-covariant in the sign of each array.

    Shallow wrapper around the generic apply_sign_symmetry_to_fn.

    Args:
        fn (Callable): the function to symmetrize. The given axis is injected into the
            inputs and the sign orbit is computed, so this function should be able to
            treat the given sign orbit axis as a batch dimension, and the overall tensor
            rank should not change (len(input.shape) == len(output.shape))

    Returns:
        Callable: a function with the same signature as the input function, but
        which has been symmetrized so that its output will be covariant with respect
        to the sign of each input, or in other words, will be odd.
    """
    return apply_sign_symmetry_to_fn(
        fn,
        functools.partial(_get_sign_orbit_array_list, axis=axis),
        functools.partial(_multiply_sign_along_axis, axis=axis),
        functools.partial(jnp.sum, axis=axis),
    )


def make_array_list_fn_sign_invariant(
    fn: Callable[[ArrayList], Array], axis: int = -2
) -> Callable[[ArrayList], Array]:
    """Make a function of an ArrayList sign-invariant (even) in the sign of each array.

    Shallow wrapper around the generic apply_sign_symmetry_to_fn.

    Args:
        fn (Callable): the function to symmetrize. The given axis is injected into the
            inputs and the sign orbit is computed, so this function should be able to
            treat the given sign orbit axis as a batch dimension, and the overall tensor
            rank should not change (len(input.shape) == len(output.shape))

    Returns:
        Callable: a function with the same signature as the input function, but
        which has been symmetrized so that its output will be invariant with respect
        to the sign of each input, or in other words, will be even.
    """
    return apply_sign_symmetry_to_fn(
        fn,
        functools.partial(_get_sign_orbit_array_list, axis=axis),
        lambda x, _: x,  # Ignore the signs to get an invariance
        functools.partial(jnp.sum, axis=axis),
    )


class ProductsSignCovariance(Module):
    """Sign covariance from a weighted sum of products of per-particle values.

    Only supports two spins at the moment. Given per-spin antiequivariant vectors
    a_1, a_2, ..., and b_1, b_2, ..., computes an antisymmetry of
    sum_{i,j} (w_{i,j} sum_{k} a_ik b_jk), or multiple such antisymmetries if
    features>1. If use_weights=False, then no weights are used, so that effectively
    w_{i,j} = 1 for all i,j.

    Attributes:
        features (int): the number of antisymmetric output features to generate. If
            use_weights is False, must be equal to 1.
        kernel_init (WeightInitializer): initializer for the weights of the dense layer.
        use_weights (bool, optional): whether to use a weighted sum of products.
            Defaults to False.
    """

    features: int
    kernel_init: WeightInitializer
    use_weights: bool = False

    @flax.linen.compact
    def __call__(self, x: ArrayList) -> Array:  # type: ignore[override]
        """Calculate weighted sum of products of up- and down-spin antiequivariances.

        Arguments:
            x (ArrayList): input antiequivariant arrays of shape
                [(..., nelec_up, d), (..., nelec_down, d)]

        Returns:
            Array: array of length features of antisymmetric values calculated
                by taking a weighted sum of the pairwise dot-products of the up- and
                down-spin antiequivariant inputs.
        """
        # TODO (ggoldsh): update this to support nspins != 2 as well
        if len(x) != 2:
            raise ValueError(
                "Products covariance only supported for nspins=2, got {}".format(len(x))
            )

        naxes = len(x[0].shape)
        batch_dims = range(naxes - 2)
        contraction_dim = (naxes - 1,)
        # Since the second last axis is not specified as either a batch or contraction
        # dim, jax.lax.dot_general will automatically compute over all pairs of up and
        # down spins. pairwise_dots thus has shape (..., nelec_up, nelec_down).
        pairwise_dots = jax.lax.dot_general(
            x[0], x[1], ((contraction_dim, contraction_dim), (batch_dims, batch_dims))
        )

        if not self.use_weights:
            if self.features != 1:
                raise ValueError(
                    "Can only return one output feature when use_weights is False. "
                    "Received {} for features.".format(self.features)
                )
            return jnp.expand_dims(jnp.sum(pairwise_dots, axis=(-1, -2)), -1)

        shape = pairwise_dots.shape
        # flattened_dots has shape (..., nelec_up * nelec_down)
        flattened_dots = jnp.reshape(
            pairwise_dots, (*shape[:-2], shape[-1] * shape[-2])
        )

        return Dense(
            self.features,
            kernel_init=self.kernel_init,
            use_bias=False,
        )(flattened_dots)
