"""Permutation invariant models."""
import math
from typing import Callable, Iterable, Sequence, Tuple

import flax
import jax.numpy as jnp

from vmcnet.utils.typing import ArrayList, SpinSplit
from .core import Dense, _split_mean
from .weights import WeightInitializer


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
            are shaped (batch_dims, output_shape_per_spin[i])
        backflow (Callable): function which computes position features from the electron
            positions. Has the signature
            (elec pos of shape (..., n, d))
                -> (stream_1e of shape (..., n, d'), r_ei of shape (..., n, nion, d))
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        register_kfac (bool, optional): whether to register the dense computations with
            KFAC. Defaults to True.
    """

    spin_split: SpinSplit
    output_shape_per_spin: Sequence[Iterable[int]]
    backflow: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    register_kfac: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow

        if isinstance(self.spin_split, int):
            nspins = self.spin_split
        else:
            nspins = len(self.spin_split) + 1

        if len(self.output_shape_per_spin) != nspins:
            raise ValueError(
                "Incorrect number of output shapes specified for number of spins, "
                "should be one shape per spin: shapes {} specified for the given "
                "spin_split {}".format(self.output_shape_per_spin, self.spin_split)
            )

        ndense_per_spin = [math.prod(shape) for shape in self.output_shape_per_spin]

        self._dense_layers = [
            Dense(
                ndense_per_spin[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
                register_kfac=self.register_kfac,
            )
            for i in range(nspins)
        ]

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> ArrayList:
        """Apply backflow -> split mean -> dense to get invariance with desired shape.

        Args:
            elec_pos (jnp.ndarray): electron positions of shape (..., nelec, d)

        Returns:
            ArrayList: list of invariant arrays which are invariant with respect to
            permutations within each split, where the last two axes of these arrays are
            specified by self.output_shape_per_spin, and the other axes are shared batch
            axes
        """
        stream_1e = self._backflow(elec_pos)[0]
        invariant_split = _split_mean(stream_1e, self.spin_split, keepdims=False)

        invariant_dense_split_out = [
            self._dense_layers[i](invariant_in)
            for i, invariant_in in enumerate(invariant_split)
        ]
        return [
            jnp.reshape(x, x.shape[:-1] + tuple(self.output_shape_per_spin[i]))
            for i, x in enumerate(invariant_dense_split_out)
        ]
