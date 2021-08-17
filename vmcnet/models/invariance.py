"""Permutation invariant models."""
import math
from typing import Iterable, Optional, Sequence

import flax
import jax.numpy as jnp

from vmcnet.utils.typing import ArrayList, Backflow, ParticleSplit
from .core import Dense, _split_mean, get_nspins
from .weights import WeightInitializer


class SplitMeanDense(flax.linen.Module):
    """Split mean of input on 2nd-to-last axis, apply unique Dense layers to each split.

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_per_spin (Sequence[int]): sequence of integers specifying the number of
            dense nodes in the unique dense layer applied to each split of the input.
            This determines the output shapes for each split, i.e. the outputs are
            shaped (..., split_size[i], ndense[i])
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        register_kfac (bool, optional): whether to register the dense computations with
            KFAC. Defaults to True.
    """

    spin_split: ParticleSplit
    ndense_per_spin: Sequence[int]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    register_kfac: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        nspins = get_nspins(self.spin_split)

        if len(self.ndense_per_spin) != nspins:
            raise ValueError(
                "Incorrect number of dense output shapes specified for number of "
                "spins, should be one shape per spin: shapes {} specified for the "
                "given spin_split {}".format(self.ndense_per_spin, self.spin_split)
            )

        self._dense_layers = [
            Dense(
                self.ndense_per_spin[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
                register_kfac=self.register_kfac,
            )
            for i in range(nspins)
        ]

    def __call__(self, x: jnp.ndarray) -> ArrayList:
        """Split the input and apply a dense layer to each split.

        Args:
            x (jnp.ndarray): array of shape (..., n, d)

        Returns:
            [(..., self.ndense_per_spin[i])]: list of length nspins, where nspins
            is the number of splits created by jnp.split(x, self.spin_split, axis=-2),
            and the ith entry of the output is the ith split mean (an array of shape
            (..., d)) transformed by a dense layer with self.ndense_per_spin[i] nodes.
        """
        x_split = _split_mean(x, self.spin_split, keepdims=False)
        return [
            self._dense_layers[i](x_spin)
            if self.ndense_per_spin[i] != 0
            else jnp.empty(x_spin.shape[:-1] + (0,))
            for i, x_spin in enumerate(x_split)
        ]


class InvariantTensor(flax.linen.Module):
    """Spinful invariance via averaged backflow, with desired shape via a dense layer.

    Attributes:
        spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        output_shape_per_spin (Sequence[Iterable[int]]): sequence of iterables which
            correspond to the desired non-batch output shapes of for each split of the
            input. This determines the output shapes for each split, i.e. the outputs
            are shaped [(batch_dims, output_shape_per_spin[i])].
        backflow (Callable or None): function which computes position features from the
            electron positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d').
            Can pass None here to only apply SplitMeanDense followed by reshaping.
        kernel_initializer (WeightInitializer): kernel initializer for the dense
            layer(s). Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer for the dense layer(s).
            Has signature (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        register_kfac (bool, optional): whether to register the dense computations with
            KFAC. Defaults to True.
    """

    spin_split: ParticleSplit
    output_shape_per_spin: Sequence[Iterable[int]]
    backflow: Optional[Backflow]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    register_kfac: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow

        nspins = get_nspins(self.spin_split)

        if len(self.output_shape_per_spin) != nspins:
            raise ValueError(
                "Incorrect number of output shapes specified for number of spins, "
                "should be one shape per spin: shapes {} specified for the given "
                "spin_split {}".format(self.output_shape_per_spin, self.spin_split)
            )

        self._ndense_per_spin = [
            math.prod(shape) for shape in self.output_shape_per_spin
        ]

    def _reshape_dense_outputs(self, dense_out: jnp.ndarray, i: int):
        output_shape = dense_out.shape[:-1] + tuple(self.output_shape_per_spin[i])
        return jnp.reshape(dense_out, output_shape)

    @flax.linen.compact
    def __call__(
        self, stream_1e: jnp.ndarray, stream_2e: Optional[jnp.ndarray] = None
    ) -> ArrayList:
        """Backflow -> split mean dense to get invariance -> reshape.

        Args:
            stream_1e (jnp.ndarray): one-electron input stream of shape
                (..., nelec, d1).
            stream_2e (jnp.ndarray, optional): two-electron input of shape
                (..., nelec, nelec, d2).

        Returns:
            ArrayList: list of invariant arrays which are invariant with respect to
            permutations within each split, where the last axes of these arrays are
            specified by self.output_shape_per_spin, and the other axes are shared batch
            axes
        """
        if self._backflow is not None:
            stream_1e = self._backflow(stream_1e, stream_2e)

        flattened_invariant_out = SplitMeanDense(
            self.spin_split,
            self._ndense_per_spin,
            self.kernel_initializer,
            self.bias_initializer,
            self.use_bias,
            self.register_kfac,
        )(stream_1e)

        return [
            self._reshape_dense_outputs(invariant_in, i)
            for i, invariant_in in enumerate(flattened_invariant_out)
        ]
