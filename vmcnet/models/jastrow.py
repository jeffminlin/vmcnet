"""Jastrow factors."""
from typing import Optional, Sequence, Union

import flax
import jax.numpy as jnp

import vmcnet.models as models
import vmcnet.physics as physics
from vmcnet.utils.typing import Array, Backflow, Jastrow

from .core import Dense, ElementWiseMultiply, Module, compute_ee_norm_with_safe_diag
from .weights import WeightInitializer, get_constant_init, zeros


def _isotropy_on_leaf(
    r_ei_leaf: Array,
    norbitals: int,
    kernel_initializer: WeightInitializer,
) -> Array:
    """Isotropic scaling of the electron-ion displacements."""
    # swap axes around and inject an axis to go from
    # r_ei_leaf: (..., nelec, nion, d) -> x_nion: (..., nelec, d, nion, 1)
    x_nion = jnp.swapaxes(r_ei_leaf, axis1=-1, axis2=-2)
    x_nion = jnp.expand_dims(x_nion, axis=-1)

    # (..., nelec, d, nion, norbitals)
    x_nion = jnp.broadcast_to(x_nion, (*x_nion.shape[:-1], norbitals))
    iso_out = ElementWiseMultiply(1, kernel_initializer)(x_nion)

    # Swap axes to get (..., nelec, norbitals, nion, d)
    return jnp.swapaxes(iso_out, axis1=-1, axis2=-3)


def _anisotropy_on_leaf(
    r_ei_leaf: Array,
    norbitals: int,
    kernel_initializer: WeightInitializer,
) -> Array:
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
    )(r_ei_leaf)

    # concatenate over the ion axis, then reshape the last axis to separate out the
    # norbitals dxd transformations and swap axes around to return
    # (..., nelec, norbitals, nion, d)
    concat_out = jnp.concatenate(split_out, axis=-2)
    out = jnp.reshape(concat_out, batch_shapes + (nion, d, norbitals))
    out = jnp.swapaxes(out, axis1=-1, axis2=-3)
    out = jnp.swapaxes(out, axis1=-1, axis2=-2)

    return out


class OneBodyExpDecay(Module):
    """Creates an isotropic exponential decay one-body Jastrow model.

    The decay is centered at the coordinates of the nuclei, and the electron-nuclei
    displacements are multiplied by trainable params before a sum and exp(-x). The decay
    is isotropic and equal for all electrons, so it computes

        -sum_ij ||a_j * (elec_i - ion_j)||

    or the exponential if logabs is False. The tensor a_j * (elec_i - ion_j) is computed
    with a split dense operation.

    Attributes:
        kernel_initializer (WeightInitializer): kernel initializer for the decay rates
            a_j. This initializes a single decay rate number per ion. Has signature
            (key, shape, dtype) -> Array
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
    def __call__(  # type: ignore[override]
        self,
        input_stream_1e: Array,
        input_stream_2e: Array,
        stream_1e: Array,
        r_ei: Array,
        r_ee: Array,
    ) -> Array:
        """Transform electron-ion displacements into an exp decay one-body Jastrow.

        Args:
            input_stream_1e (Array): input one-electron stream; unused
            input_stream_2e (Array): input two-electron stream; unused
            stream_1e (Array): one-electron stream, post-backflow; unused
            r_ei (Array): electron-ion displacements of shape
                (..., nelec, nion, d)
            r_ee (Array): electron-electron displacements of shape
                (..., nelec, nelec, d); unused

        Returns:
            Array: -sum_ij ||a_j * (elec_i - ion_j)||, when self.logabs is True,
            or exp of that expression when self.logabs is False. If the input has shape
            (batch_dims, nelec, nion, d), then the output has shape (batch_dims,)
        """
        del input_stream_1e, input_stream_2e, stream_1e, r_ee
        # scale_out has shape (..., nelec, 1, nion, d)
        scale_out = _isotropy_on_leaf(r_ei, 1, self._kernel_initializer)
        scaled_distances = jnp.linalg.norm(scale_out, axis=-1)

        abs_lin_comb_distances = jnp.sum(scaled_distances, axis=(-1, -2, -3))

        if self.logabs:
            return -abs_lin_comb_distances

        return jnp.exp(-abs_lin_comb_distances)


class TwoBodyExpDecay(Module):
    """Isotropic exponential decay two-body Jastrow model.

    The decay is isotropic in the sense that each electron-nuclei and electron-electron
    term is isotropic, i.e. radially symmetric. The computed interactions are:

        sum_i(-sum_j Z_j ||elec_i - ion_j|| + sum_k Q ||elec_i - elec_k||)

    or the exponential if logabs is False. Z_j and Q are initialized to init_ei_strength
    and init_ee_strength, respectively, and are trainable if trainable is True.

    Attributes:
        init_ei_strength (Array or Sequence[float]): 1-d array or sequence of
            length nion which gives the initial strength of the electron-nucleus
            interaction per ion
        init_ee_strength (float, optional): initial strength of the electron-electron
            interaction. Defaults to 1.0.
        log_scale_factor (float, optional): Amount to add to the log jastrow (amounts to
            a multiplicative factor after exponentiation). Defaults to 0.0.
        logabs (bool, optional): whether to return the log jastrow (True) or the jastrow
            (False). Defaults to True.
        trainable (bool, optional): whether to allow the jastrow to be trainable.
            Defaults to True.
    """

    init_ei_strength: Union[Array, Sequence[float]]
    init_ee_strength: float = 1.0
    log_scale_factor: float = 0.0
    logabs: bool = True
    trainable: bool = True

    @flax.linen.compact
    def __call__(  # type: ignore[override]
        self,
        input_stream_1e: Array,
        input_stream_2e: Array,
        stream_1e: Array,
        r_ei: Array,
        r_ee: Array,
    ) -> Array:
        """Compute jastrow with both electron-ion and electron-electron effects.

        Args:
            input_stream_1e (Array): input one-electron stream; unused
            input_stream_2e (Array): input two-electron stream; unused
            stream_1e (Array): one-electron stream, post-backflow; unused
            r_ei (Array): electron-ion displacements with shape
                (..., nelec, nion, d)
            r_ee (Array): electron-electron displacements with shape
                (..., nelec, nelec, d)

        Returns:
            Array:

                sum_i(-sum_j Z_j ||elec_i - ion_j|| + sum_k Q ||elec_i - elec_k||),

            where Z_j and Q are trainable if trainable is true, and an exponential is
            taken if logabs is False
        """
        del input_stream_1e, input_stream_2e, stream_1e
        ei_distances = jnp.linalg.norm(r_ei, axis=-1)
        ee_distances = jnp.squeeze(compute_ee_norm_with_safe_diag(r_ee), axis=-1)
        sum_ee_effect = jnp.sum(jnp.triu(ee_distances), axis=-1, keepdims=True)

        if self.trainable:
            split_over_ions = models.core.split(
                ei_distances, ei_distances.shape[-1], axis=-1
            )
            # TODO: potentially add support for this to SplitDense or otherwise?
            split_scaled_ei_distances = [
                Dense(
                    1,
                    kernel_init=get_constant_init(self.init_ei_strength[i]),
                    use_bias=False,
                )(single_ion_displacement)
                for i, single_ion_displacement in enumerate(split_over_ions)
            ]
            scaled_ei_distances = jnp.concatenate(split_scaled_ei_distances, axis=-1)

            sum_ee_effect = Dense(
                1,
                kernel_init=get_constant_init(self.init_ee_strength),
                use_bias=False,
            )(sum_ee_effect)
        else:
            scaled_ei_distances = self.init_ei_strength * ei_distances
            sum_ee_effect = self.init_ee_strength * sum_ee_effect

        sum_ee_effect = jnp.squeeze(sum_ee_effect, axis=-1)
        sum_ei_effect = jnp.sum(scaled_ei_distances, axis=-1)
        unscaled_interaction = jnp.sum(sum_ee_effect - sum_ei_effect, axis=-1)
        interaction = unscaled_interaction + self.log_scale_factor

        if self.logabs:
            return interaction

        return jnp.exp(interaction)


def get_two_body_decay_scaled_for_chargeless_molecules(
    ion_pos: Array,
    ion_charges: Array,
    init_ee_strength: float = 1.0,
    logabs: bool = True,
    trainable: bool = True,
) -> Jastrow:
    """Make molecular decay jastrow, scaled for chargeless molecules.

    The scale factor is chosen so that the log jastrow is initialized to 0 when
    electrons are at ion positions.

    Args:
        ion_pos (Array): an (nion, d) array of ion positions.
        ion_charges (Array): an (nion,) array of ion charges, in units of one
            elementary charge (the charge of one electron)
        init_ee_strength (float, optional): the initial strength of the
            electron-electron interaction. Defaults to 1.0.
        logabs (bool, optional): whether to return the log jastrow (True) or the jastrow
            (False). Defaults to True.
        trainable (bool, optional): whether to allow the jastrow to be trainable.
            Defaults to True.

    Returns:
        Callable: a flax Module with signature (r_ei, r_ee) -> jastrow or log jastrow
    """
    r_ii, charge_charge_prods = physics.potential._get_ion_ion_info(
        ion_pos, ion_charges
    )
    jastrow_scale_factor = float(
        0.5 * jnp.sum(jnp.linalg.norm(r_ii, axis=-1) * charge_charge_prods)
    )
    jastrow = TwoBodyExpDecay(
        ion_charges,
        init_ee_strength,
        log_scale_factor=jastrow_scale_factor,
        logabs=logabs,
        trainable=trainable,
    )
    return jastrow


class BackflowJastrow(Module):
    """Backflow-based general permutation invariant Jastrow.

    Attributes:
        backflow (Callable or None): function which computes position features from the
            electron positions. Has the signature
            (
                stream_1e of shape (..., n, d'),
                optional stream_2e of shape (..., nelec, nelec, d2),
            ) -> stream_1e of shape (..., n, d').
            Can pass None here to use a stream_1e from an already computed backflow.
        logabs (bool, optional): whether to return the log jastrow (True) or the jastrow
            (False). Defaults to True.
    """

    backflow: Optional[Backflow]
    logabs: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._backflow = self.backflow

    @flax.linen.compact
    def __call__(  # type: ignore[override]
        self,
        input_stream_1e: Array,
        input_stream_2e: Array,
        stream_1e: Array,
        r_ei: Array,
        r_ee: Array,
    ) -> Array:
        """Compute backflow-based general permutation invariant Jastrow.

        Args:
            input_stream_1e (Array): input one-electron stream
            input_stream_2e (Array): input two-electron stream
            stream_1e (Array): one-electron stream, post-backflow
            r_ei (Array): electron-ion displacements with shape
                (..., nelec, nion, d); unused
            r_ee (Array): electron-electron displacements with shape
                (..., nelec, nelec, d); unused

        Returns:
            Array: -mean_i ||Backflow_i||, or exp(-mean_i ||Backflow_i||) if
            logabs is False
        """
        del r_ei, r_ee

        if self._backflow is not None:
            stream_1e = self._backflow(input_stream_1e, input_stream_2e)

        log_jastrow = -jnp.mean(jnp.linalg.norm(stream_1e, axis=-1), axis=-1)

        if self.logabs:
            return log_jastrow

        return jnp.exp(log_jastrow)
