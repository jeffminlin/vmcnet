"""Core model building parts."""
from typing import Callable, Optional, cast

import flax
import jax
import jax.numpy as jnp

import vmcnet.utils as utils
from vmcnet.models.weights import (
    WeightInitializer,
    get_bias_initializer,
    get_kernel_initializer,
)

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _log_linear_exp(
    signs: jnp.ndarray,
    vals: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: int = 0,
) -> jnp.ndarray:
    """Stably compute log(abs(sum_i(sign_i * exp(vals_i)))) along an axis.

    Args:
        signs (jnp.ndarray): array of signs of the input x with shape (..., d, ...),
            where d is the size of the given axis
        vals (jnp.ndarray): array of log|abs(x)| with shape (..., d, ...), where d is
            the size of the given axis
        weights (jnp.ndarray, optional): weights of a linear transformation to apply to
            the given axis, with shape (d, d'). If not provided, a simple sum is taken
            along this axis, equivalent to (d, 1) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum. Defaults to 0.

    Returns:
        jnp.ndarray: outputs with shape (..., d', ...), where d' = 1 if weights is None,
        and d' = weights.shape[1] otherwise.
    """
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(vals - max_val)
    if weights is not None:
        batch_dims = terms_divided_by_max.shape[:axis]
        weights = jnp.reshape(weights, batch_dims + weights.shape)
        if axis < 0:
            axis = terms_divided_by_max.ndim + axis
        transformed_divided_by_max = jax.lax.dot_general(
            weights, terms_divided_by_max, (((0,), (axis,)), (batch_dims, batch_dims))
        )
    else:
        transformed_divided_by_max = jnp.sum(
            terms_divided_by_max, axis=axis, keepdims=True
        )
    return jnp.log(jnp.abs(transformed_divided_by_max)) + max_val


def _valid_skip(x: jnp.ndarray, y: jnp.ndarray):
    return x.shape[-1] == y.shape[-1]


flax.linen.DenseGeneral


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
        """Applies a linear transformation with optional bias along the last dimension.

        Args:
            inputs (jnp.ndarray): The nd-array to be transformed.

        Returns:
            jnp.ndarray: The transformed input.
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


class SimpleResNet(flax.linen.Module):
    """Simplest fully-connected ResNet.

    Attributes:
        ndense_inner (int): number of dense nodes in layers before the final layer.
        ndense_outer (int): number of output features, i.e. the number of dense nodes in
            the final Dense call.
        nlayers (int): number of dense layers applied to the input, including the final
            layer. If this is 0, the final dense layer will still be applied.
        kernel_init (WeightInitializer, optional): initializer function for the weight
            matrices of each layer. Defaults to orthogonal initialization.
        bias_init (WeightInitializer, optional): initializer function for the bias.
            Defaults to random normal initialization.
        activation_fn (Activation): activation function between intermediate layers (is
            not applied after the final dense layer). Has the signature
            jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether the dense layers should all have bias terms
            or not. Defaults to True.
    """

    ndense_inner: int
    ndense_final: int
    nlayers: int
    activation_fn: Activation
    kernel_init: WeightInitializer = get_kernel_initializer("orthogonal")
    bias_init: WeightInitializer = get_bias_initializer("normal")
    use_bias: bool = True

    def setup(self):
        """Setup dense layers."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._activation_fn = self.activation_fn

        self.inner_dense = [
            Dense(
                self.ndense_inner,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
            )
            for _ in range(self.nlayers - 1)
        ]
        self.final_dense = Dense(
            self.ndense_final,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=False,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Repeated application of (dense layer -> activation -> optional skip) block.

        Args:
            x (jnp.ndarray): an input array of shape (..., d)

        Returns:
            jnp.ndarray: array of shape (..., self.ndense_final)
        """
        for dense_layer in self.inner_dense:
            prev_x = x
            x = dense_layer(prev_x)
            x = self._activation_fn(x)
            if _valid_skip(prev_x, x):
                x = cast(jnp.ndarray, x + prev_x)

        return self.final_dense(x)
