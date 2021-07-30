"""Jastrow factors."""
from typing import Callable
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
    """Creates an isotropic exponential decay one-body Jastrow model.

    The decay is centered at the coordinates of the nuclei, and the electron-nuclei
    displacements are multiplied by trainable params before a sum and exp(-x). The decay
    is isotropic and equal for all electrons, so it computes

        exp(-sum_ij ||a_j * (elec_i - ion_j)||)

    or the logarithm if logabs is True. The tensor a_j * (elec_i - ion_j) is computed
    with a depthwise 2d convolution.

    Attributes:
        kernel_initializer (WeightInitializer): kernel initializer for the decay rates
            a_j. This initializes a single decay rate number per ion. Has signature
            (key, shape, dtype) -> jnp.ndarray
        logabs (bool, optional): whether to compute -sum_ij ||a_j * (elec_i - ion_j)||,
            when logabs is True, or exp of that expression when logabs is False.
            Defaults to True.
    """

    kernel_initializer: WeightInitializer
    logabs: bool = True

    def setup(self):
        """Setup the kernel initializer."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._kernel_initializer = self.kernel_initializer

    @flax.linen.compact
    def __call__(self, r_ei: jnp.ndarray) -> jnp.ndarray:
        """Transform electron-ion displacements into an exp decay one-body Jastrow.

        Args:
            r_ei (jnp.ndarray): electron-ion displacements of shape
                (..., nelec, nion, d)

        Returns:
            jnp.ndarray: -sum_ij ||a_j * (elec_i - ion_j)||, when self.logabs is True,
            or exp of that expression when self.logabs is False. If the input has shape
            (batch_dims, nelec, nion, d), then the output has shape (batch_dims,)
        """
        # scale_out has shape (..., nelec, 1, nion, d)
        scale_out = _isotropy_on_leaf(
            r_ei, 1, self._kernel_initializer, register_kfac=True
        )
        scaled_distances = jnp.linalg.norm(scale_out, axis=-1)

        abs_lin_comb_distances = jnp.sum(scaled_distances, axis=(-1, -2, -3))

        if self.logabs:
            return -abs_lin_comb_distances

        return jnp.exp(-abs_lin_comb_distances)


def make_molecular_decay(
    ion_charges: jnp.ndarray,
    strength: float = 1.0,
    scale_factor: float = 0.0,
    logabs: bool = True,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Make an exp decay jastrow with both electron-ion and electron-electron effects.

    Args:
        ion_charges (jnp.ndarray): an (nion,) array of ion charges, in units of one
            elementary charge (the charge of one electron)
        strength (float, optional): amount to multiply the overall interaction by.
            Defaults to 1.0.
        scale_factor (float, optional): Amount to add to the log jastrow (amounts to a
            multiplicative factor after exponentiation). Defaults to 0.0.
        logabs (bool, optional): whether to return the log jastrow (True) or the jastrow
            (False). Defaults to True.

    Returns:
        Callable: a function with signature (r_ei, r_ee) -> jastrow or log jastrow
    """

    def molecular_decay(r_ei: jnp.ndarray, r_ee: jnp.ndarray) -> jnp.ndarray:
        scaled_ei_distances = ion_charges * jnp.linalg.norm(r_ei, axis=-1)
        n = r_ee.shape[-2]
        eye_n = jnp.expand_dims(jnp.eye(n), axis=-1)
        r_ee_diag_ones = r_ee + eye_n
        ee_distances = jnp.squeeze(
            jnp.linalg.norm(r_ee_diag_ones, axis=-1, keepdims=True) * (1.0 - eye_n),
            axis=-1,
        )
        interaction = strength * (
            jnp.sum(
                jnp.sum(jnp.triu(ee_distances), axis=-1)
                - jnp.sum(scaled_ei_distances, axis=-1),
                axis=-1,
            )
            + scale_factor
        )
        if logabs:
            return interaction

        return jnp.exp(interaction)

    return molecular_decay
