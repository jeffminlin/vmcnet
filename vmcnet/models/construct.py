"""Combine pieces to form full models."""
from typing import List, Union

import flax
import jax.numpy as jnp


def compose_antisymmetry_with_equivariance(equivariance, antisymmetry):
    """Make a model out of an equivariant part and an antisymmetric part.

    Args:
        equivariance (Callable): equivariant part of the model
        antisymmetry (Callable): antisymmetric part of the model

    Returns:
        flax.linen.Module: a flax model which evaluates antisymmetry(equivariance(x)),
        where x can be split into different spin inputs
    """

    class AntisymmetricModel(flax.linen.Module):
        """Model which is antisymmetric via (antisymmetry(equivariance(x))).

        Attributes:
            nspin_split: int or list/array of integers, passed to the
                `indices_or_sections` arg of jnp.split
        """

        nspin_split: Union[int, List[int], jnp.ndarray]

        def setup(self):
            self.equivariance = equivariance
            self.antisymmetry = antisymmetry

        def __call__(self, x):
            x_split = jnp.split(x, self.nspin_split, axis=-2)
            backflow = self.equivariance(x_split)
            return self.antisymmetry(backflow)

    return AntisymmetricModel
