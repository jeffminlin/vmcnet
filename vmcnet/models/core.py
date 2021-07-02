"""Core model building parts."""
from typing import Callable, Optional, Sequence, Tuple, Union, cast

import flax
import jax
import jax.numpy as jnp

import vmcnet.utils as utils
from vmcnet.models.weights import (
    WeightInitializer,
    get_bias_initializer,
    get_kernel_initializer,
)
from vmcnet.utils.typing import SLArray, PyTree

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def log_linear_exp(
    signs: jnp.ndarray,
    vals: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: int = 0,
    register_kfac: bool = True,
) -> SLArray:
    """Stably compute sign and log(abs(.)) of sum_i(sign_i * w_ij * exp(vals_i)) + b_j.

    In order to avoid overflow when computing

        log(abs(sum_i(sign_i * w_ij * exp(vals_i)))),

    the largest exp(val_i) is divided out from all the values and added back in after
    the outer log, i.e.

        log(abs(sum_i(sign_i * w_ij * exp(vals_i - max)))) + max.

    This trick also avoids the underflow issue of when all vals are small enough that
    exp(val_i) is approximately 0 for all i.

    Args:
        signs (jnp.ndarray): array of signs of the input x with shape (..., d, ...),
            where d is the size of the given axis
        vals (jnp.ndarray): array of log|abs(x)| with shape (..., d, ...), where d is
            the size of the given axis
        weights (jnp.ndarray, optional): weights of a linear transformation to apply to
            the given axis, with shape (d, d'). If not provided, a simple sum is taken
            instead, equivalent to (d, 1) weights equal to 1. Defaults to None.
        axis (int, optional): axis along which to take the sum and max. Defaults to 0.
        register_kfac (bool, optional): if weights are not None, whether to register the
            linear part of the computation with KFAC. Defaults to True.

    Returns:
        (SLArray): sign of linear combination, log of linear
        combination. Both outputs have shape (..., d', ...), where d' = 1 if weights is
        None, and d' = weights.shape[1] otherwise.
    """
    max_val = jnp.max(vals, axis=axis, keepdims=True)
    terms_divided_by_max = signs * jnp.exp(vals - max_val)
    if weights is not None:
        # swap axis and -1 to conform to jnp.dot and register_batch_dense api
        terms_divided_by_max = jnp.swapaxes(terms_divided_by_max, axis, -1)
        transformed_divided_by_max = jnp.dot(terms_divided_by_max, weights)
        if register_kfac:
            transformed_divided_by_max = utils.kfac.register_batch_dense(
                transformed_divided_by_max, terms_divided_by_max, weights, None
            )

        # swap axis and -1 back after the contraction and registration
        transformed_divided_by_max = jnp.swapaxes(transformed_divided_by_max, axis, -1)
    else:
        transformed_divided_by_max = jnp.sum(
            terms_divided_by_max, axis=axis, keepdims=True
        )

    signs = jnp.sign(transformed_divided_by_max)
    vals = jnp.log(jnp.abs(transformed_divided_by_max)) + max_val
    return signs, vals


def is_tuple_of_arrays(x: PyTree) -> bool:
    """Returns True if x is a tuple of jnp.ndarray objects."""
    return isinstance(x, tuple) and all(isinstance(x_i, jnp.ndarray) for x_i in x)


def get_alternating_signs(n: int) -> jnp.ndarray:
    """Return alternating series of 1 and -1, of length n."""
    return jax.ops.index_update(jnp.ones(n), jax.ops.index[1::2], -1.0)


def get_nelec_per_spin(
    spin_split: Union[int, Sequence[int]], nelec_total: int
) -> Tuple[int, ...]:
    """From a spin_split and nelec_total, get the number of particles per spin.

    If the number of particles per spin is nelec_per_spin = (n1, n2, ..., nk), then
    spin_split should be jnp.cumsum(nelec_per_spin)[:-1], or an integer of these are all
    equal. This function is the inverse of this operation.
    """
    if isinstance(spin_split, int):
        return (nelec_total // spin_split,) * spin_split
    else:
        spin_diffs = tuple(jnp.diff(jnp.array(spin_split)))
        return (spin_split[0],) + spin_diffs + (nelec_total - spin_split[-1],)


def _valid_skip(x: jnp.ndarray, y: jnp.ndarray):
    return x.shape[-1] == y.shape[-1]


class Dense(flax.linen.Module):
    """A linear transformation applied over the last dimension of the input.

    This is a copy of the flax Dense layer, but with registration of the weights for use
    with KFAC.

    Attributes:
        features (int): the number of output features.
        kernel_init (WeightInitializer, optional): initializer function for the weight
            matrix. Defaults to orthogonal initialization.
        bias_init (WeightInitializer, optional): initializer function for the bias.
            Defaults to random normal initialization.
        use_bias (bool, optional): whether to add a bias to the output. Defaults to
            True.
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


class LogDomainDense(flax.linen.Module):
    """A linear transformation applied on the last axis of the input, in the log domain.

    If the inputs are (sign(x), log(abs(x))), the outputs are
    (sign(Wx + b), log(abs(Wx + b))).

    The bias is implemented by extending the inputs with a vector of ones.

    Attributes:
        features (int): the number of output features.
        kernel_init (WeightInitializer, optional): initializer function for the weight
            matrix. Defaults to orthogonal initialization.
        use_bias (bool, optional): whether to add a bias to the output. Defaults to
            True.
        register_kfac (bool, optional): whether to register the computation with KFAC.
            Defaults to True.
    """

    features: int
    kernel_init: WeightInitializer = get_kernel_initializer("orthogonal")
    use_bias: bool = True
    register_kfac: bool = True

    @flax.linen.compact
    def __call__(self, x: SLArray) -> SLArray:
        """Applies a linear transformation with optional bias along the last dimension.

        Args:
            x (SLArray): The nd-array in slog form to be transformed.

        Returns:
            SLArray: The transformed input, in slog form.
        """
        sign_x, log_abs_x = x
        input_dim = log_abs_x.shape[-1]

        if self.use_bias:
            input_dim += 1
            sign_x = jnp.concatenate([sign_x, jnp.ones_like(sign_x[..., 0:1])], axis=-1)
            log_abs_x = jnp.concatenate(
                [log_abs_x, jnp.zeros_like(log_abs_x[..., 0:1])], axis=-1
            )

        kernel = self.param("kernel", self.kernel_init, (input_dim, self.features))

        return log_linear_exp(
            sign_x,
            log_abs_x,
            kernel,
            axis=-1,
            register_kfac=self.register_kfac,
        )


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
