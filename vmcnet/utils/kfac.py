"""Utils for interfacing with DeepMind's KFAC implementation.

These functions are taken directly from DeepMind's FermiNet jax branch, see
https://github.com/deepmind/ferminet/blob/aade61b3d30883b3238d6b50c85404d0e8176155/ferminet/curvature_tags_and_blocks.py

Some names are slightly modified (e.g. repeated_dense -> batch_dense).
"""
from typing import Mapping, Union

import jax.numpy as jnp

from kfac_ferminet_alpha import curvature_blocks as blocks
from kfac_ferminet_alpha import layers_and_loss_tags as tags
from kfac_ferminet_alpha import utils as kfac_utils

BATCH_DENSE_TAG_NAME = "batch_dense_tag"
batch_dense_tag = tags.LayerTag(BATCH_DENSE_TAG_NAME, 1, 1)


def register_batch_dense(y, x, w, b):
    """Register the weights of a dense layer.

    The dense layer performs y = wx + b.
    """
    if b is None:
        return batch_dense_tag.bind(y, x, w)
    return batch_dense_tag.bind(y, x, w, b)


class BatchDenseBlock(blocks.DenseTwoKroneckerFactored):
    """Dense curvature block corresponding to inputs of shape (..., d)."""

    def compute_extra_scale(self) -> jnp.ndarray:
        """Extra scale factor for the curvature block (relative to other blocks)."""
        (x_shape,) = self.inputs_shapes
        return kfac_utils.product(x_shape) // (x_shape[0] * x_shape[-1])

    def update_curvature_matrix_estimate(
        self,
        info: Mapping[str, blocks._Arrays],
        batch_size: int,
        ema_old: Union[float, jnp.ndarray],
        ema_new: Union[float, jnp.ndarray],
        pmap_axis_name: str,
    ) -> None:
        """Satsify kfac_ferminet_alpha's assumption that the inputs are 2d.

        The inputs are generally of shape (batch_size, ..., d), and the optimizer
        expects that the input `batch_size` matches the first axis of `info["inputs"]`.
        However, the Dense layer itself is batch applied only to the very last axis,
        with all of the other axes acting as batch axes. Thus to compute the correct
        curvature matrix estimate, we reshape the inputs and outputs to be
        (-1, shape[-1]) and provide to the superclass implementation a new `batch_size`
        equal to the product of the batch axes sizes.
        """
        info = dict(**info)
        (x,), (dy,) = info["inputs"], info["outputs_tangent"]
        assert x.shape[0] == batch_size
        info["inputs"] = (x.reshape([-1, x.shape[-1]]),)
        info["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
        super().update_curvature_matrix_estimate(
            info, x.size // x.shape[-1], ema_old, ema_new, pmap_axis_name
        )


blocks.set_default_tag_to_block(BATCH_DENSE_TAG_NAME, BatchDenseBlock)
