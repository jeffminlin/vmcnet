"""Permutation invariant models."""
import math
from typing import Callable, Iterable, Sequence, Tuple

import flax
import jax.numpy as jnp

from vmcnet.utils.typing import ArrayList, SpinSplit
from .core import Dense, _split_mean
from .weights import WeightInitializer


class InvariantTensor(flax.linen.Module):
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
        stream_1e = self._backflow(elec_pos)[0]
        invariant_split = _split_mean(stream_1e, self.spin_split, keepdims=False)

        invariant_dense_split_out = [
            self._dense_layers[i](invariant_in)
            for i, invariant_in in enumerate(invariant_split)
        ]
        return [
            jnp.reshape(x, x.shape[:-2] + tuple(self.output_shape_per_spin[i]))
            for i, x in enumerate(invariant_dense_split_out)
        ]
