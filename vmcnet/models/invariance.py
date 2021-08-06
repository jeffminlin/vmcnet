"""Permutation invariant models."""
import math
from typing import Iterable, Sequence

import flax
import jax.numpy as jnp

from vmcnet.utils.typing import ArrayList, Backflow, ComputeInputStreams, SpinSplit
from .core import Dense, _split_mean, get_nspins
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
            are shaped [(batch_dims, output_shape_per_spin[i])].
        compute_input_streams (ComputeInputStreams): function to compute input
            streams from electron positions. Has the signature
            (elec_pos of shape (..., n, d)) -> (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
                optional r_ei of shape (..., n, nion, d),
                optional r_ee of shape (..., n, n, d),
            )
        backflow (Callable): function which computes position features from the electron
                positions. Has the signature
                (
                    stream_1e of shape (..., n, d'),
                    optional stream_2e of shape (..., nelec, nelec, d2),
                ) -> stream_1e of shape (..., n, d')
        kernel_initializer (WeightInitializer): kernel initializer for the dense
            layer(s). Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer for the dense layer(s).
            Has signature (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        register_kfac (bool, optional): whether to register the dense computations with
            KFAC. Defaults to True.
    """

    spin_split: SpinSplit
    output_shape_per_spin: Sequence[Iterable[int]]
    compute_input_streams: ComputeInputStreams
    backflow: Backflow
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True
    register_kfac: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow
        self._compute_input_streams = self.compute_input_streams

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

        self._dense_layers = [
            Dense(
                self._ndense_per_spin[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
                register_kfac=self.register_kfac,
            )
            for i in range(nspins)
        ]

    def _get_shaped_outputs(self, invariant_in: jnp.ndarray, i: int):
        output_shape = invariant_in.shape[:-1] + tuple(self.output_shape_per_spin[i])
        if self._ndense_per_spin[i] == 0:
            # Return dummy array of expected shape even when no elements are required.
            return jnp.zeros(output_shape)

        dense_out = self._dense_layers[i](invariant_in)
        return jnp.reshape(dense_out, output_shape)

    @flax.linen.compact
    def __call__(self, elec_pos: jnp.ndarray) -> ArrayList:
        """Compute input streams -> backflow -> split mean -> dense to get invariance.

        Args:
            elec_pos (jnp.ndarray): array of particle positions (..., nelec, d)

        Returns:
            ArrayList: list of invariant arrays which are invariant with respect to
            permutations within each split, where the last axes of these arrays are
            specified by self.output_shape_per_spin, and the other axes are shared batch
            axes
        """
        stream_1e, stream_2e, _, _ = self._compute_input_streams(elec_pos)
        stream_1e = self._backflow(stream_1e, stream_2e)

        invariant_split = _split_mean(stream_1e, self.spin_split, keepdims=False)

        return [
            self._get_shaped_outputs(invariant_in, i)
            for i, invariant_in in enumerate(invariant_split)
        ]
