"""Utils for interfacing with DeepMind's KFAC implementation.

These functions are taken directly from DeepMind's FermiNet jax branch, see
https://github.com/deepmind/ferminet/blob/aade61b3d30883b3238d6b50c85404d0e8176155/ferminet/curvature_tags_and_blocks.py

Some names are slightly modified (e.g. repeated_dense -> dense).
"""
from typing import Optional, Mapping, Union

import jax
import jax.numpy as jnp

from kfac_ferminet_alpha import curvature_blocks as blocks
from kfac_ferminet_alpha import layers_and_loss_tags as tags
from kfac_ferminet_alpha import utils


vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)

dense_tag = tags.LayerTag("dense_tag", 1, 1)


def register_dense(y, x, w, b):
    """Register the weights of a dense layer."""
    if b is None:
        return dense_tag.bind(y, x, w)
    return dense_tag.bind(y, x, w, b)


class DenseBlock(blocks.DenseTwoKroneckerFactored):
    """Dense block."""

    def compute_extra_scale(self) -> Optional[jnp.ndarray]:
        (x_shape,) = self.inputs_shapes
        return utils.product(x_shape) // (x_shape[0] * x_shape[-1])

    def update_curvature_matrix_estimate(
        self,
        info: Mapping[str, blocks._Arrays],  # pylint: disable=protected-access
        batch_size: int,
        ema_old: Union[float, jnp.ndarray],
        ema_new: Union[float, jnp.ndarray],
        pmap_axis_name: str,
    ) -> None:
        info = dict(**info)
        (x,), (dy,) = info["inputs"], info["outputs_tangent"]
        assert x.shape[0] == batch_size
        info["inputs"] = (x.reshape([-1, x.shape[-1]]),)
        info["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
        super().update_curvature_matrix_estimate(
            info, x.size // x.shape[-1], ema_old, ema_new, pmap_axis_name
        )


blocks.set_default_tag_to_block("dense_tag", DenseBlock)
