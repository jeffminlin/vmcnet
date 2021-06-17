"""Core model building parts."""
from typing import Callable
import flax
import jax.numpy as jnp

import vmcnet.utils as utils
from vmcnet.models.weights import (
    WeightInitializer,
    get_kernel_initializer,
    get_bias_initializer,
)

Activation = Callable[[jnp.ndarray], jnp.ndarray]


class Dense(flax.linen.Module):
    """A linear transformation applied over the last dimension of the input.

    This is a copy of the flax Dense layer, but with registration of the weights for use
    with KFAC.

    Attributes:
        features (int): the number of output features.
        use_bias (bool, optional): whether to add a bias to the output. Defaults to
            True.
        kernel_init (WeightInitializer, optional): initializer function for the weight
            matrix. Defaults to orthogonal initialization.
        bias_init (WeightInitializer, optional): initializer function for the bias.
            Defaults to random normal initialization.
        register_kfac (bool, optional): whether to register the computation with KFAC.
            Defaults to True.
    """

    features: int
    kernel_init: WeightInitializer = get_kernel_initializer("orthogonal")
    bias_init: WeightInitializer = get_bias_initializer("normal")
    use_bias: bool = True
    register_kfac: bool = True

    @flax.linen.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, kernel)
        bias = None
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias

        if self.register_kfac:
            return utils.kfac.register_batch_dense(y, inputs, kernel, bias)
        else:
            return y
