"""Antisymmetry parts to compose into a model."""
import functools
import itertools
from typing import Callable, Tuple
from vmcnet.models.weights import WeightInitializer

import flax
import jax
import jax.numpy as jnp

from vmcnet.models.core import SimpleResNet

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _istupleofarrays(x) -> bool:
    return isinstance(x, tuple) and all(isinstance(x_i, jnp.ndarray) for x_i in x)


def _reduce_sum_over_leaves(xs) -> jnp.ndarray:
    return functools.reduce(lambda a, b: a + b, jax.tree_leaves(xs))


def _reduce_prod_over_leaves(xs) -> jnp.ndarray:
    return functools.reduce(lambda a, b: a * b, jax.tree_leaves(xs))


def slogdet_product(xs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the (sign, log) of the product of determinants of the leaves of a pytree.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        (jnp.ndarray, jnp.ndarray): the product of the sign_dets and the sum of the
        log_dets over all leaves of the pytree xs
    """
    slogdets = jax.tree_map(jnp.linalg.slogdet, xs)

    slogdet_leaves, _ = jax.tree_flatten(slogdets, _istupleofarrays)
    sign_prod, log_prod = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdet_leaves
    )
    return sign_prod, log_prod


def logdet_product(xs) -> jnp.ndarray:
    """Compute the log|prod_x det(x)| of the leaves x of a pytree (throwing away sign).

    Because we don't need to carry sign, the logic can be made slightly simpler and
    we can avoid a few computations.

    Args:
        xs (pytree): pytree of tensors which are square in the last two dimensions, i.e.
            all leaves must have shape (..., n_leaf, n_leaf); the last two dims can
            be different from leaf to leaf, but the batch dimensions must be the same
            for all leaves.

    Returns:
        jnp.ndarray: the sum of the log_dets over all leaves of the pytree xs, which is
        equal to the log of the product of the dets over all leaves of xs
    """
    logdets = jax.tree_map(lambda x: jnp.linalg.slogdet(x)[1], xs)
    log_prod = _reduce_sum_over_leaves(logdets)
    return log_prod


def _get_lexicographic_signs(n: int) -> jnp.ndarray:
    """Get the signs of the n! permutations of (1,...,n) in lexicographic order."""
    signs = jnp.ones(1)

    for i in range(2, n + 1):
        alternating_signs = jax.ops.index_update(jnp.ones(i), jax.ops.index[1::2], -1.0)
        signs = jnp.concatenate([sign * signs for sign in alternating_signs])

    return signs


class ParallelPermutations(flax.linen.Module):
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
        self.permutation_list = list(itertools.permutations(range(self.n)))
        self.signs = _get_lexicographic_signs(self.n)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.take(x, self.permutation_list, axis=-2), self.signs


class SplitBruteForceAntisymmetrizedResNet(flax.linen.Module):
    ndense: int
    nlayers: int
    activation_fn: Activation
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    logabs: bool = True

    def _single_leaf_call(self, x: jnp.ndarray) -> jnp.ndarray:
        n = x.shape[-2]
        d = x.shape[-1]

        x_perm, signs = ParallelPermutations(n)(x)
        x_perm = jnp.reshape(x_perm, x_perm.shape[:-2] + (n * d,))  # (..., n!, n * d)
        signs = jnp.expand_dims(signs, axis=-1)  # (n!, 1)

        # perms_out has shape (..., n!, 1)
        perms_out = SimpleResNet(
            self.ndense,
            1,
            self.nlayers,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )(x_perm)
        signed_perms_out = signs * perms_out

        return jnp.sum(signed_perms_out, axis=(-1, -2))

    @flax.linen.compact
    def __call__(self, xs) -> jnp.ndarray:
        antisyms = jax.tree_map(self._single_leaf_call, xs)
        if not self.logabs:
            return _reduce_prod_over_leaves(antisyms)

        log_antisyms = jax.tree_map(lambda x: jnp.log(jnp.abs(x)), antisyms)
        return _reduce_sum_over_leaves(log_antisyms)


class ComposedBruteForceAntisymmetrizedResNet(flax.linen.Module):
    ndense: int
    nlayers: int
    activation_fn: Activation
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    logabs: bool = True

    def _get_single_leaf_perm(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n = x.shape[-2]
        return ParallelPermutations(n)(x)

    @flax.linen.compact
    def __call__(self, xs) -> jnp.ndarray:
        perms_and_signs = jax.tree_map(self._get_single_leaf_perm, xs)
        perms_and_signs_leaves, _ = jax.tree_flatten(perms_and_signs, _istupleofarrays)
        nspins = len(perms_and_signs_leaves)
        nfactorials = [leaf[0].shape[-3] for leaf in perms_and_signs_leaves]

        broadcasted_perms = []
        reshaped_signs = []
        for i, (perm, sign) in enumerate(perms_and_signs_leaves):
            # desired sign shapes are [i: (1, ..., n_i!, ..., 1, 1,)], with k + 1 dims,
            # where k = nspins
            first_k_shape = (1,) * i + sign.shape[0:1] + (1,) * (nspins - i - 1)
            reshape_sign_shape = first_k_shape + (1,)
            sign = jnp.reshape(sign, reshape_sign_shape)
            reshaped_signs.append(sign)

            # desired broadcasted x shapes are [i: (..., n_1!, ..., n_k!, n_i, d)],
            # where k = nspins, and x_i = (..., n_i, d). This is achieved by:
            # 1) reshape to (..., 1, ..., n_i!,... 1, n_i, d), then
            # 2) broadcast to (..., n_1!, ..., n_k!, n_i, d)
            reshape_x_shape = perm.shape[:-3] + first_k_shape + perm.shape[-2:]
            broadcast_x_shape = perm.shape[:-3] + tuple(nfactorials) + perm.shape[-2:]
            perm = jnp.reshape(perm, reshape_x_shape)
            perm = jnp.broadcast_to(perm, broadcast_x_shape)
            broadcasted_perms.append(perm)

        # make input shape (..., n_1!, ..., n_k!, (n_1 + ... + n_k) * d)
        concat_perms = jnp.concatenate(broadcasted_perms, axis=-2)
        concat_perms = jnp.reshape(concat_perms, concat_perms.shape[:-2] + (-1,))

        # all_perms_out has shape (..., n_1!, ..., n_k!, 1)
        all_perms_out = SimpleResNet(
            self.ndense,
            1,
            self.nlayers,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )(concat_perms)
        signed_perms_out = _reduce_prod_over_leaves([all_perms_out, reshaped_signs])

        antisymmetrized_out = jnp.sum(
            signed_perms_out, axis=tuple(-i for i in range(1, nspins + 2))
        )
        if self.logabs:
            return jnp.log(jnp.abs(antisymmetrized_out))

        return antisymmetrized_out
