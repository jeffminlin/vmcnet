"""Permutation equivariant functions."""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from vmcnet.physics.potential import _compute_displacements
from vmcnet.models.weights import WeightInitializer

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _split_mean(splits, x, axis=-2, keepdims=True):
    split_x = jnp.split(x, splits, axis=axis)
    split_x_mean = jax.tree_map(
        functools.partial(jnp.mean, axis=axis, keepdims=keepdims), split_x
    )
    return split_x, split_x_mean


def _rolled_concat(arrays, n, axis=-1):
    return jnp.concatenate(arrays[n:] + arrays[:n], axis=axis)


def _tree_sum(tree1, tree2):
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a + b, tree1, tree2)


def _tree_prod(tree1, tree2):
    """Leaf-wise produdct of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a * b, tree1, tree2)


def _valid_skip(x, y):
    return x.shape[-1] == y.shape[-1]


def compute_input_streams(
    elec_pos: jnp.ndarray,
    ion_pos: jnp.ndarray = None,
    include_ee: bool = True,
    include_ei_norm: bool = True,
    include_ee_norm: bool = True,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    input_1e, r_ei = compute_electron_ion(elec_pos, ion_pos, include_ei_norm)
    input_2e = None
    if not include_ee:
        input_2e = compute_electron_electron(elec_pos, include_ee_norm)
    return input_1e, input_2e, r_ei


def compute_electron_ion(
    elec_pos: jnp.ndarray, ion_pos: jnp.ndarray = None, include_ei_norm: bool = True
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    r_ei = None
    input_1e = elec_pos
    if ion_pos is not None:
        r_ei = _compute_displacements(input_1e, ion_pos)
        if include_ei_norm:
            r_ei_norm = jnp.linalg.norm(r_ei, axis=-1, keepdims=True)
            r_ei_with_norm = jnp.concatenate([r_ei, r_ei_norm], axis=-1)
        input_1e = jnp.reshape(r_ei_with_norm, r_ei_with_norm.shape[:-2] + (-1,))
    return input_1e, r_ei


def compute_electron_electron(
    elec_pos: jnp.ndarray, include_ee_norm: bool = True
) -> jnp.ndarray:
    input_2e = _compute_displacements(elec_pos, elec_pos)
    if include_ee_norm:
        n = elec_pos.shape[-2]
        eye_n = jnp.eye(n)
        r_ee_diag_ones = input_2e + eye_n[..., None]
        r_ee_norm = jnp.linalg.norm(r_ee_diag_ones, axis=-1) * (1.0 - eye_n)
        input_2e = jnp.concatenate([input_2e, r_ee_norm], axis=-1)
    return input_2e


class FermiNetResidualBlock(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    ndense_1e: int
    ndense_2e: int
    kernel_initializer_unmixed: WeightInitializer
    kernel_initializer_mixed: WeightInitializer
    kernel_initializer_2e_1e_stream: WeightInitializer
    kernel_initializer_2e_2e_stream: WeightInitializer
    bias_initializer_1e_stream: WeightInitializer
    bias_initializer_2e_stream: WeightInitializer
    activation_fn: Activation
    use_bias: bool = True
    skip_connection: bool = True
    cyclic_spins: bool = True

    @flax.linen.compact
    def __call__(
        self, in_1e: jnp.ndarray, in_2e: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        out_1e = FermiNetOneElectronLayer(
            self.spin_split,
            self.ndense_1e,
            self.kernel_initializer_unmixed,
            self.kernel_initializer_mixed,
            self.kernel_initializer_2e_1e_stream,
            self.bias_initializer_1e_stream,
            self.activation_fn,
            use_bias=self.use_bias,
            skip_connection=self.skip_connection,
            cyclic_spins=self.cyclic_spins,
        )(in_1e, in_2e)

        out_2e = None
        if in_2e is not None:
            out_2e = FermiNetTwoElectronLayer(
                self.ndense_2e,
                self.kernel_initializer_2e_2e_stream,
                self.bias_initializer_2e_stream,
                self.activation_fn,
                use_bias=self.use_bias,
                skip_connection=self.skip_connection,
            )(in_2e)

        return out_1e, out_2e


class FermiNetOneElectronLayer(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    ndense: int
    kernel_initializer_unmixed: WeightInitializer
    kernel_initializer_mixed: WeightInitializer
    kernel_initializer_2e: WeightInitializer
    bias_initializer: WeightInitializer
    activation_fn: Activation
    use_bias: bool = True
    skip_connection: bool = True
    cyclic_spins: bool = True

    def setup(self):
        # workaround MyPy's typing error for callable attribute
        self._activation_fn = self.activation_fn

        self._unmixed_dense = flax.linen.Dense(
            self.ndense,
            kernel_init=self.kernel_initializer_unmixed,
            bias_init=self.bias_initializer,
        )
        self._mixed_dense = flax.linen.Dense(
            self.ndense, kernel_init=self.kernel_initializer_mixed, use_bias=False
        )
        self._dense_2e = flax.linen.Dense(
            self.ndense, kernel_init=self.kernel_initializer_2e, use_bias=False
        )

    def _compute_transformed_1e_means(self, split_means):
        if self.cyclic_spins:
            split_concat = [
                _rolled_concat(split_means, idx) for idx in range(len(split_means))
            ]
            dense_split_1e_means = jax.tree_map(self._mixed_dense, split_concat)
        else:
            split_concat = jnp.concatenate(split_means, axis=-1)
            dense_split_1e_means = self._mixed_dense(split_concat)
            dense_split_1e_means = [dense_split_1e_means, dense_split_1e_means]
        return dense_split_1e_means

    def _compute_transformed_2e_means(self, in_2e):
        split_2e = jnp.split(in_2e, self.spin_split, axis=-3)
        concat_2e = []
        for spin in range(len(split_2e)):
            split_arrays = _split_mean(
                self.spin_split, split_2e[spin], axis=-2, keepdims=False
            )[1]
            if self.cyclic_spins:
                concat_arrays = _rolled_concat(split_arrays, spin)
            else:
                concat_arrays = jnp.concatenate(split_arrays, axis=-1)
            concat_2e.append(concat_arrays)
        return jax.tree_map(self._dense_2e, concat_2e)

    def __call__(self, in_1e: jnp.ndarray, in_2e: jnp.ndarray = None) -> jnp.ndarray:
        split_1e, split_1e_means = _split_mean(
            self.spin_split, in_1e, axis=-2, keepdims=True
        )
        dense_split_1e = jax.tree_map(self._unmixed_dense, split_1e)
        dense_split_1e_means = self._compute_transformed_1e_means(split_1e_means)
        dense_out = _tree_sum(dense_split_1e, dense_split_1e_means)

        if in_2e is not None:
            dense_split_2e = self._compute_transformed_2e_means(in_2e)
            dense_out = _tree_sum(dense_out, dense_split_2e)

        dense_out_concat = jnp.concatenate(dense_out, axis=-2)
        nonlinear_out = self._activation_fn(dense_out_concat)

        if self.skip_connection and _valid_skip(in_1e, nonlinear_out):
            nonlinear_out = nonlinear_out + in_1e

        return nonlinear_out


class FermiNetTwoElectronLayer(flax.linen.Module):
    ndense: int
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    activation_fn: Activation
    use_bias: bool = True
    skip_connection: bool = True

    def setup(self):
        # workaround MyPy's typing error for callable attribute
        self._activation_fn = self.activation_fn
        self._dense = flax.linen.Dense(
            self.ndense,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dense_out = self._dense(x)
        nonlinear_out = self._activation_fn(dense_out)

        if self.skip_connection and _valid_skip(x, nonlinear_out):
            nonlinear_out = nonlinear_out + x

        return nonlinear_out


class FermiNetBackflow(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    ndense_list: List[Tuple[int, int]]
    kernel_initializer_unmixed: WeightInitializer
    kernel_initializer_mixed: WeightInitializer
    kernel_initializer_2e_1e_stream: WeightInitializer
    kernel_initializer_2e_2e_stream: WeightInitializer
    bias_initializer_1e_stream: WeightInitializer
    bias_initializer_2e_stream: WeightInitializer
    activation_fn: Activation
    ion_pos: Optional[jnp.ndarray] = None
    include_2e_stream: bool = True
    include_ei_norm: bool = True
    include_ee_norm: bool = True
    use_bias: bool = True
    skip_connection: bool = True
    cyclic_spins: bool = True

    @flax.linen.compact
    def __call__(
        self, elec_pos: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        stream_1e, stream_2e, r_ei = compute_input_streams(
            elec_pos,
            self.ion_pos,
            include_ee=self.include_2e_stream,
            include_ei_norm=self.include_ei_norm,
            include_ee_norm=self.include_ee_norm,
        )

        for features in self.ndense_list:
            stream_1e, stream_2e = FermiNetResidualBlock(
                spin_split=self.spin_split,
                ndense_1e=features[0],
                ndense_2e=features[1],
                kernel_initializer_unmixed=self.kernel_initializer_unmixed,
                kernel_initializer_mixed=self.kernel_initializer_mixed,
                kernel_initializer_2e_1e_stream=self.kernel_initializer_2e_1e_stream,
                kernel_initializer_2e_2e_stream=self.kernel_initializer_2e_2e_stream,
                bias_initializer_1e_stream=self.bias_initializer_1e_stream,
                bias_initializer_2e_stream=self.bias_initializer_2e_stream,
                activation_fn=self.activation_fn,
                use_bias=self.use_bias,
                skip_connection=self.skip_connection,
                cyclic_spins=self.cyclic_spins,
            )(stream_1e, stream_2e)

        return stream_1e, r_ei


class SplitDense(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    ndense: Sequence[int]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True

    def setup(self):
        if isinstance(self.spin_split, int):
            nspins = self.spin_split
        else:
            nspins = len(self.spin_split) + 1

        if len(self.ndense) != nspins:
            raise ValueError(
                "Incorrect number of dense output shapes specified for number of "
                "spins, should be one shape per spin: shapes {} specified for the "
                "given spin_split {}".format(self.ndense, self.spin_split)
            )

        self._dense_layers = [
            flax.linen.Dense(
                self.ndense[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
            )
            for i in range(nspins)
        ]

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        x_split = jnp.split(x, self.spin_split, axis=-2)
        return [self._dense_layers[i](x_spin) for i, x_spin in enumerate(x_split)]


class FermiNetOrbitalLayer(flax.linen.Module):
    spin_split: Union[int, Sequence[int]]
    norbitals: Sequence[int]
    kernel_initializer_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_linear: WeightInitializer
    use_bias: bool = True
    isotropic_decay: bool = False

    def _compute_exponential_envelopes(
        self, x: jnp.ndarray, norbitals: int, isotropic: bool = False
    ) -> jnp.ndarray:
        # x is (..., nelec, nion, d)
        if isotropic:
            conv_out = self._isotropy(x, norbitals)
        else:
            conv_out = self._anisotropy(x, norbitals)
        # conv_out has shape (..., nelec, norbitals, nion, d)
        distances = jnp.linalg.norm(conv_out, axis=-1)
        inv_exp_distances = jnp.exp(-distances)  # (..., nelec, norbitals, nion)
        lin_comb_nion = flax.linen.Dense(
            1, kernel_init=self.kernel_initializer_envelope_ion, use_bias=False
        )(inv_exp_distances)
        return jnp.squeeze(lin_comb_nion, axis=-1)  # (..., nelec, norbitals)

    def _isotropy(self, x: jnp.ndarray, norbitals: int) -> jnp.ndarray:
        d = x.shape[-1]
        # x_nion is (..., nelec, d, nion)
        x_nion = jnp.swapaxes(x, axis1=-1, axis2=-2)
        batch_shapes = x_nion.shape[:-1]  # (..., nelec, d)
        nion = x.shape[-1]
        conv_in = jnp.reshape(x_nion, (-1, nion, 1))
        # conv_out has shape (batch_shapes, nion, norbitals),
        # this applies nion parallel maps which go 1 -> norbitals
        conv_out = flax.linen.Conv(
            norbitals,
            1,
            kernel_init=self.kernel_initializer_envelope_dim,
            use_bias=False,
        )(conv_in)
        conv_out = jnp.reshape(conv_out, batch_shapes + (nion, norbitals))
        return jnp.swapaxes(conv_out, axis1=-1, axis2=-3)

    def _anisotropy(self, x: jnp.ndarray, norbitals: int) -> jnp.ndarray:
        batch_shapes = x.shape[:-2]  # (..., nelec)
        d = x.shape[-1]
        nion = x.shape[-2]
        conv_in = jnp.reshape(x, (-1, nion, d))
        # conv_out_flat has shape (batch_shapes, nion, d * norbitals)
        # this applies nion parallel maps which go d -> d * norbitals
        conv_out_flat = flax.linen.Conv(
            d * norbitals,
            1,
            kernel_init=self.kernel_initializer_envelope_dim,
            use_bias=False,
        )(conv_in)
        conv_out = jnp.reshape(conv_out_flat, batch_shapes + (nion, d, norbitals))
        conv_out = jnp.swapaxes(
            jnp.swapaxes(conv_out, axis1=-1, axis2=-3), axis1=-1, axis2=-2
        )  # (..., nelec, norbitals, nion, d)
        return conv_out

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray, r_ei: jnp.ndarray = None) -> List[jnp.ndarray]:
        # orbs has shapes [(..., nelec[spin], norbitals[spin])]
        orbs = SplitDense(
            self.spin_split,
            self.norbitals,
            self.kernel_initializer_linear,
            self.bias_initializer_linear,
            use_bias=self.use_bias,
        )(x)
        if r_ei is not None:
            r_ei_split = jnp.split(r_ei, self.spin_split, axis=-3)
            # exp_envelopes has shapes [(..., nelec[spin], norbitals[spin])]
            exp_envelopes = jax.tree_map(
                functools.partial(
                    self._compute_exponential_envelopes, isotropic=self.isotropic_decay
                ),
                r_ei_split,
                list(self.norbitals),
            )
            orbs = _tree_prod(orbs, exp_envelopes)
        return orbs
