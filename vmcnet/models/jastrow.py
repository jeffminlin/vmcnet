"""Jastrow factors."""
import flax
import jax.numpy as jnp

import vmcnet.models as models
from vmcnet.models.weights import WeightInitializer, zeros


def _isotropy_on_leaf(
    r_ei_leaf: jnp.ndarray,
    norbitals: int,
    kernel_initializer: WeightInitializer,
    register_kfac: bool = True,
) -> jnp.ndarray:
    """Isotropic scaling of the electron-ion displacements."""
    nion = r_ei_leaf.shape[-2]

    # swap axes around and inject an axis to go from
    # r_ei_leaf: (..., nelec, nion, d) -> x_nion: (..., nelec, d, nion, 1)
    x_nion = jnp.swapaxes(r_ei_leaf, axis1=-1, axis2=-2)
    x_nion = jnp.expand_dims(x_nion, axis=-1)

    # split x_nion along the ion axis and apply nion parallel maps 1 -> norbitals
    # along the last axis, resulting in
    # [i: (..., nelec, d, 1, norbitals)], where i is the ion index
    split_out = models.equivariance.SplitDense(
        nion,
        (norbitals,) * nion,
        kernel_initializer,
        zeros,
        use_bias=False,
        register_kfac=register_kfac,
    )(x_nion)

    # concatenate over the ion axis, then swap axes to get
    # an output of shape (..., nelec, norbitals, nion, d)
    concat_out = jnp.concatenate(split_out, axis=-2)
    return jnp.swapaxes(concat_out, axis1=-1, axis2=-3)


def _anisotropy_on_leaf(
    r_ei_leaf: jnp.ndarray,
    norbitals: int,
    kernel_initializer: WeightInitializer,
    register_kfac: bool = True,
) -> jnp.ndarray:
    """Anisotropic scaling of the electron-ion displacements."""
    batch_shapes = r_ei_leaf.shape[:-2]
    nion = r_ei_leaf.shape[-2]
    d = r_ei_leaf.shape[-1]

    # split x along the ion axis and apply nion parallel maps d -> d * norbitals
    # along the last axis, resulting in [i: (..., nelec, 1, d * norbitals)]
    split_out = models.equivariance.SplitDense(
        nion,
        (d * norbitals,) * nion,
        kernel_initializer,
        zeros,
        use_bias=False,
        register_kfac=register_kfac,
    )(r_ei_leaf)

    # concatenate over the ion axis, then reshape the last axis to separate out the
    # norbitals dxd transformations and swap axes around to return
    # (..., nelec, norbitals, nion, d)
    concat_out = jnp.concatenate(split_out, axis=-2)
    out = jnp.reshape(concat_out, batch_shapes + (nion, d, norbitals))
    out = jnp.swapaxes(out, axis1=-1, axis2=-3)
    out = jnp.swapaxes(out, axis1=-1, axis2=-2)

    return out


class IsotropicAtomicExpDecay(flax.linen.Module):
    kernel_initializer: WeightInitializer
    logabs: bool = True

    def setup(self):
        self._kernel_initializer = self.kernel_initializer

    @flax.linen.compact
    def __call__(self, r_ei: jnp.ndarray) -> jnp.ndarray:
        # scale_out has shape (..., nelec, 1, nion, d)
        scale_out = _isotropy_on_leaf(
            r_ei, 1, self._kernel_initializer, register_kfac=True
        )
        scaled_distances = jnp.linalg.norm(scale_out, axis=-1)

        abs_lin_comb_distances = jnp.sum(scaled_distances, axis=(-1, -2, -3))

        if self.logabs:
            return -abs_lin_comb_distances

        return jnp.exp(-abs_lin_comb_distances)
