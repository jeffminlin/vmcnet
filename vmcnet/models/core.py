"""Core model building parts."""
from typing import Any

import flax
from jax import lax
import jax.numpy as jnp

import vmcnet.utils as utils
from vmcnet.models.weights import (
    WeightInitializer,
    get_kernel_initializer,
    get_bias_initializer,
)


class Dense(flax.linen.Module):
    """A linear transformation applied over the last dimension of the input.

    This is a copy of the flax Dense layer, but with registration of the weights for use
    with KFAC.

    Attributes:
        features (int): the number of output features.
        use_bias (bool, optional): whether to add a bias to the output. Defaults to
            True.
        dtype (dtype, optional): the dtype of the computation. Defaults to jnp.float32.
        precision (lax.Precision, optional): numerical precision of the computation. See
            `jax.lax.Precision` for details. Defaults to None.
        kernel_init (WeightInitializer, optional): initializer function for the weight
            matrix. Defaults to orthogonal initialization.
        bias_init (WeightInitializer, optional): initializer function for the bias.
            Defaults to random normal initialization.
        register_kfac (bool, optional): whether to register the computation with KFAC.
            Defaults to True.
    """

    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: WeightInitializer = get_kernel_initializer("orthogonal")
    bias_init: WeightInitializer = get_bias_initializer("normal")
    register_kfac: bool = True

    @flax.linen.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        bias = None
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias

        if self.register_kfac:
            return utils.kfac.register_repeated_dense(y, inputs, kernel, bias)
        else:
            return y
