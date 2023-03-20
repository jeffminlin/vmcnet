"""Antisymmetry parts to compose into a model."""
import functools
import itertools
from typing import Callable, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, SLArray, PyTree
from vmcnet.utils.slog_helpers import array_list_to_slog, array_to_slog, slog_multiply
from .core import Module, get_alternating_signs, is_tuple_of_arrays


def _reduce_sum_over_leaves(xs: PyTree) -> Array:
    return functools.reduce(lambda a, b: a + b, jax.tree_util.tree_leaves(xs))


def _reduce_prod_over_leaves(xs: PyTree) -> Array:
    return functools.reduce(lambda a, b: a * b, jax.tree_util.tree_leaves(xs))


def slogdet_product(xs: PyTree) -> SLArray:
    """Compute the (sign, log) of the product of determinants of the leaves of a pytree.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        (Array, Array): the product of the sign_dets and the sum of the
        log_dets over all leaves of the pytree xs
    """
    slogdets = jax.tree_map(jnp.linalg.slogdet, xs)

    slogdet_leaves, _ = jax.tree_util.tree_flatten(slogdets, is_tuple_of_arrays)
    sign_prod, log_prod = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdet_leaves
    )
    return sign_prod, log_prod


def logdet_product(xs: PyTree) -> Array:
    """Compute the log|prod_x det(x)| of the leaves x of a pytree (throwing away sign).

    Because we don't need to carry sign, the logic can be made slightly simpler and
    we can avoid a few computations.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        Array: the sum of the log_dets over all leaves of the pytree xs, which is
        equal to the log of the product of the dets over all leaves of xs
    """
    logdets = jax.tree_map(lambda x: jnp.linalg.slogdet(x)[1], xs)
    log_prod = _reduce_sum_over_leaves(logdets)
    return log_prod


def _get_lexicographic_signs(n: int) -> Array:
    """Get the signs of the n! permutations of (1,...,n) in lexicographic order."""
    signs = jnp.ones(1)

    for i in range(2, n + 1):
        alternating_signs = get_alternating_signs(i)
        signs = jnp.concatenate([sign * signs for sign in alternating_signs])

    return signs


class ParallelPermutations(Module):
    """Get all perms along the 2nd-to-last axis, w/ perms stored as a constant.

    If inputs are shape (..., n, d), then the outputs are shape (..., n!, n, d).
    The signs of the permutations are also returned. This layer is here so that the
    permutations and their signs are stored as a constant in the computational graph,
    instead of being recomputed at each iteration. This makes sense to do if there is
    enough memory to store all permutations of the input and any downstream
    computations, so that downstream computations can be done in parallel on all
    permutations. If there is not enough memory for this, then it is better to compute
    the permutations on the fly.

    Attributes:
        n (int): size of the second-to-last axis of the inputs. Should be >= 1.
    """

    n: int

    def setup(self):
        """Store the list of permutations and signs for the symmetric group."""
        self.permutation_list = jnp.array(list(itertools.permutations(range(self.n))))
        self.signs = _get_lexicographic_signs(self.n)

    def __call__(self, x: Array) -> Tuple[Array, Array]:  # type: ignore[override]
        """Collect all permutations of x and the signs of these permutations.

        Args:
            x (Array): an array of particles with shape (..., n, d)

        Returns:
            (Array, Array): all permutations of x along the second axis,
            with shape (..., n!, n, d), and the signs of the permutations in the same
            order as the third-to-last axis (lexicographic order)
        """
        return jnp.take(x, self.permutation_list, axis=-2), self.signs


class FactorizedAntisymmetrize(Module):
    """Separately antisymmetrize fns over leaves of a pytree and return the product.

    See https://arxiv.org/abs/2112.03491 for a description of the factorized
    antisymmetric layer.

    Attributes:
        fns_to_antisymmetrize (pytree): pytree of functions with the same tree structure
            as the input pytree, each of which is a Callable with signature
            Array of shape (..., ninput_dim) -> Array of shape (..., dout).
            On the ith leaf, ninput_dim = n[i] * din[i], where n[i] is the size of the
            second-to-last axis and din[i] is the size of the last axis of the input xs.
        logabs (bool, optional): whether to compute sum_i log(abs(psi_i)) if logabs is
            True, or prod_i psi_i if logabs is False, where psi_i is the output from
            antisymmetrizing the ith function on the ith input. Defaults to True.
    """

    fns_to_antisymmetrize: PyTree  # pytree of Callables with the same treedef as input
    logabs: bool = True

    def _single_leaf_call(
        self, fn_to_antisymmetrize: Callable[[Array], Array], x: Array
    ) -> Array:
        n = x.shape[-2]
        din = x.shape[-1]

        x_perm, signs = ParallelPermutations(n)(x)
        x_perm = jnp.reshape(
            x_perm, x_perm.shape[:-2] + (n * din,)
        )  # (..., n!, n * din)
        signs = jnp.expand_dims(signs, axis=-1)  # (n!, 1)

        # perms_out has shape (..., n!, dout)
        perms_out = fn_to_antisymmetrize(x_perm)
        signed_perms_out = signs * perms_out

        return jnp.sum(signed_perms_out, axis=-2)

    @flax.linen.compact
    def __call__(self, xs: PyTree) -> Union[Array, SLArray]:  # type: ignore[override]
        """Antisymmetrize the leaves of self.fns_to_antisymmetrize on the leaves of xs.

        Args:
            xs (pytree): pytree of inputs with the same tree structure as that of
                self.fns_to_antisymmetrize. The ith leaf has shape (..., n[i], d[i]),
                and the ith antisymmetrization happens with respect to n[i].

        Returns:
            Array or SLArray:
                prod_i psi_i if self.logabs is False, or
                prod_i sign(psi_i), sum_i log(abs(psi_i)) if self.logabs is True,
            where psi_i is the output from antisymmetrizing the ith function on the ith
            input.
        """
        # Flatten the trees for fns_to_antisymmetrize and xs, because Module
        # freezes all instances of lists to tuples, so this can cause treedef
        # compatibility problems
        antisyms = jax.tree_map(
            self._single_leaf_call,
            jax.tree_util.tree_leaves(self.fns_to_antisymmetrize),
            jax.tree_util.tree_leaves(xs),
        )
        if not self.logabs:
            return _reduce_prod_over_leaves(antisyms)

        slog_antisyms = array_list_to_slog(jax.tree_util.tree_leaves(antisyms))
        return functools.reduce(slog_multiply, slog_antisyms)


class GenericAntisymmetrize(Module):
    """Antisymmetrize a single function over the leaves of a pytree.

    See https://arxiv.org/abs/2112.03491 for a description of the generic antisymmetric
    layer.

    For each leaf of a pytree, a given function of all the leaves is antisymmetrized
    over the second-to-last axis of each leaf. These explicit antisymmetrization
    operations are composed with each other (they commute, so the order does not
    matter), giving an output which is antisymmetric with respect to particle exchange
    within each leaf but not with respect to particle exchange between leaves.

    Attributes:
        fn_to_antisymmetrize (Callable): Callable with signature
            Array with shape (..., ninput_dim) -> (..., 1). This is the function
            to be antisymmetrized. ninput_dim is equal to
                n[1] * d[1] + ... + n[k] * d[k],
            where n[i] is the size of the second-to-last axis and d[i] is the size of
            the last axis of the ith leaf of the input xs.
        logabs (bool, optional): whether to compute log(abs(psi)) if logabs is True, or
            psi if logabs is False, where psi is the output from antisymmetrizing
            self.fn_to_antisymmetrize. Defaults to True.
    """

    fn_to_antisymmetrize: Callable[[Array], Array]
    logabs: bool = True

    def setup(self):
        """Setup the function to antisymmetrize."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._fn_to_antisymmetrize = self.fn_to_antisymmetrize

    def _get_single_leaf_perm(self, x: Array) -> Tuple[Array, Array]:
        n = x.shape[-2]
        return ParallelPermutations(n)(x)

    @flax.linen.compact
    def __call__(self, xs: PyTree) -> Union[Array, SLArray]:  # type: ignore[override]
        """Antisymmetrize self.fn_to_antisymmetrize over the leaves of xs.

        Args:
            xs (pytree): a pytree of inputs, each which corresponds to a different set
                of particles to antisymmetrize with respect to. The ith leaf has shape
                (..., n[i], d[i]), and the antisymmetrization happens with respect to
                n[i].

        Returns:
            Array:
                psi if logabs is False, or
                sign(psi), log(abs(psi)) if self.logabs is True,
            where psi is the output from antisymmetrizing
            self.fn_to_antisymmetrize on all leaves of xs.
        """
        perms_and_signs = jax.tree_map(self._get_single_leaf_perm, xs)
        perms_and_signs_leaves, _ = jax.tree_util.tree_flatten(
            perms_and_signs, is_tuple_of_arrays
        )
        nleaves = len(perms_and_signs_leaves)
        nperms_per_leaf = [leaf[0].shape[-3] for leaf in perms_and_signs_leaves]

        broadcasted_perms = []
        reshaped_signs = []
        for i, (leaf_perms, leaf_signs) in enumerate(perms_and_signs_leaves):
            ith_factorial = (1,) * i + leaf_signs.shape[0:1] + (1,) * (nleaves - i - 1)

            # desired sign[i] shape is (1, ..., n_i!, ..., 1, 1), with nspins + 1 dims
            sign_shape = ith_factorial + (1,)
            leaf_signs = jnp.reshape(leaf_signs, sign_shape)
            reshaped_signs.append(leaf_signs)

            # desired broadcasted x_i shape is [i: (..., n_1!, ..., n_k!, n_i * d_i)],
            # where k = nleaves, and x_i = (..., n_i, d_i). This is achieved by:
            # 1) reshape to (..., 1, ..., n_i!,... 1, n_i, d_i), then
            # 2) broadcast to (..., n_1!, ..., n_k!, n_i, d_i)
            # 3) flatten last axis to (..., n_1!, ..., n_k!, n_i * d_i)
            reshape_x_shape = (
                leaf_perms.shape[:-3] + ith_factorial + leaf_perms.shape[-2:]
            )
            broadcast_x_shape = (
                leaf_perms.shape[:-3] + tuple(nperms_per_leaf) + leaf_perms.shape[-2:]
            )
            leaf_perms = jnp.reshape(leaf_perms, reshape_x_shape)
            leaf_perms = jnp.broadcast_to(leaf_perms, broadcast_x_shape)
            flat_leaf_perms = jnp.reshape(leaf_perms, leaf_perms.shape[:-2] + (-1,))
            broadcasted_perms.append(flat_leaf_perms)

        # make input shape (..., n_1!, ..., n_k!, n_1 * d_1 + ... + n_k * d_k)
        concat_perms = jnp.concatenate(broadcasted_perms, axis=-1)

        all_perms_out = self._fn_to_antisymmetrize(concat_perms)

        # all_perms_out has shape (..., n_1!, ..., n_k!, 1)
        # Each leaf of reshaped_signs has k+1 axes, but all except the ith axis has size
        # 1. The ith axis has size n_i!. Thus when the leaves of reshaped_signs are
        # multiplied with all_perms_out, the product will broadcast each leaf and apply
        # the signs along the correct (ith) axis of the output.
        signed_perms_out = _reduce_prod_over_leaves([all_perms_out, reshaped_signs])

        antisymmetrized_out = jnp.sum(
            signed_perms_out, axis=tuple(-i for i in range(1, nleaves + 2))
        )
        if not self.logabs:
            return antisymmetrized_out

        return array_to_slog(antisymmetrized_out)
