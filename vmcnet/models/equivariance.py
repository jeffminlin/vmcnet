"""Permutation equivariant functions."""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from vmcnet.physics.potential import _compute_displacements
from vmcnet.models.weights import WeightInitializer, zeros

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _split_mean(splits, x, axis=-2, keepdims=True):
    """Split an array and then take the mean over an axis in each of the splits."""
    split_x = jnp.split(x, splits, axis=axis)
    split_x_mean = jax.tree_map(
        functools.partial(jnp.mean, axis=axis, keepdims=keepdims), split_x
    )
    return split_x_mean


def _rolled_concat(arrays, n, axis=-1):
    """Concatenate a list of arrays starting from the nth and wrapping back around."""
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
    """Create input streams with electron and optionally ion data.

    If `ion_pos` is given, computes the electron-ion displacements (i.e. nuclear
    coordinates) and concatenates/flattens them along the ion dimension. If
    `include_ei_norm` is True, then the distances are also concatenated, so the map is
    elec_pos = (..., nelec, d) -> input_1e = (..., nelec, nion * (d + 1)).

    If `include_ee` is True, then a two-electron stream of shape (..., nelec, nelec, d)
    is also computed and returned (otherwise None is returned). If `include_ee_norm` is
    True, then this becomes (..., nelec, nelec, d + 1) by concatenating pairwise
    distances onto the stream.

    Args:
        elec_pos (jnp.ndarray): electron positions of shape (..., nelec, d)
        ion_pos (jnp.ndarray, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_ee (bool, optional): whether to compute pairwise electron
            displacements/distances. Defaults to True.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.

    Returns:
        (jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]):

        first output: one-electron input of shape (..., nelec, d'), where
            d' = d if `ion_pos` is None,
            d' = nion * d if `ion_pos` is given and `include_ei_norm` is False, and
            d' = nion * (d + 1) if `ion_pos` is given and `include_ei_norm` is True.

        second output: two-electron input of shape (..., nelec, nelec, d'), where
            d' = d if `include_ee_norm` is False, and
            d' = d + 1 if `include_ee_norm` is True

        third output: electron-ion displacements of shape (..., nelec, nion, d)

        If `include_ee` is False, then the second output is None. If `ion_pos` is None,
        then the third output is None.
    """
    input_1e, r_ei = compute_electron_ion(elec_pos, ion_pos, include_ei_norm)
    input_2e = None
    if include_ee:
        input_2e = compute_electron_electron(elec_pos, include_ee_norm)
    return input_1e, input_2e, r_ei


def compute_electron_ion(
    elec_pos: jnp.ndarray, ion_pos: jnp.ndarray = None, include_ei_norm: bool = True
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Compute electron-ion displacements and optionally add on the distances.

    Args:
        elec_pos (jnp.ndarray): electron positions of shape (..., nelec, d)
        ion_pos (jnp.ndarray, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.

    Returns:
        (jnp.ndarray, Optional[jnp.ndarray]):

        first output: one-electron input of shape (..., nelec, d'), where
            d' = d if `ion_pos` is None,
            d' = nion * d if `ion_pos` is given and `include_ei_norm` is False, and
            d' = nion * (d + 1) if `ion_pos` is given and `include_ei_norm` is True.

        second output: electron-ion displacements of shape (..., nelec, nion, d)

        If `ion_pos` is None, then the second output is None.
    """
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
    """Compute electron-electron displacements and optionally add on the distances.

    Args:
        elec_pos (jnp.ndarray): electron positions of shape (..., nelec, d)
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.

    Returns:
        jnp.ndarray: two-electron input of shape (..., nelec, nelec, d'), where
            d' = d if `include_ee_norm` is False, and
            d' = d + 1 if `include_ee_norm` is True
    """
    input_2e = _compute_displacements(elec_pos, elec_pos)
    if include_ee_norm:
        n = elec_pos.shape[-2]
        eye_n = jnp.eye(n)
        r_ee_diag_ones = input_2e + eye_n[..., None]
        r_ee_norm = jnp.linalg.norm(r_ee_diag_ones, axis=-1, keepdims=True) * (
            1.0 - eye_n
        )
        input_2e = jnp.concatenate([input_2e, r_ee_norm], axis=-1)
    return input_2e


class FermiNetResidualBlock(flax.linen.Module):
    """A single residual block in the FermiNet equivariant part.

    Combines the one-electron and two-electron streams.

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_1e (int): number of dense nodes in the one-electron stream part
        ndense_2e (int): number of dense nodes in the two-electron stream part
        kernel_initializer_unmixed (WeightInitializer): kernel initializer for the
            unmixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the previous one-electron stream output. Has
            signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_1e_stream (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_2e_stream (WeightInitializer): kernel initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_1e_stream (WeightInitializer): bias initializer for the
            one-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_2e_stream (WeightInitializer): bias initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function in the electron streams. Has
            the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether to add a bias term in the electron streams.
            Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and outputs of the streams match. Defaults
            to True.
        cyclic_spins (bool, optional): whether the the concatenation in the one-electron
            stream should satisfy a cyclic equivariance structure, i.e. if there are
            three spins (1, 2, 3), then in the mixed part of the stream, after averaging
            but before the linear transformation, cyclic equivariance means the inputs
            are [(1, 2, 3), (2, 3, 1), (3, 1, 2)]. If False, then the inputs are
            [(1, 2, 3), (1, 2, 3), (1, 2, 3)] (as in the original FermiNet).
            When there are only two spins (spin-1/2 case), then this is equivalent to
            true spin equivariance. Defaults to False (original FermiNet).
        advance_2e (bool, optional): whether to apply the FermiNetTwoElectronLayer to
            the two-electron stream, or to return it untransformed. Defaults to False.
    """

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
    cyclic_spins: bool = False
    advance_2e: bool = False

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

        if not self.advance_2e or in_2e is None:
            return out_1e, in_2e

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
    """A single layer in the one-electron stream of the FermiNet equivariant part.

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense (int): number of dense nodes
        kernel_initializer_unmixed (WeightInitializer): kernel initializer for the
            unmixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the previous one-electron stream output. Has
            signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function. Has the signature
            jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and output match. Defaults to True.
        cyclic_spins (bool, optional): whether the the concatenation in the one-electron
            stream should satisfy a cyclic equivariance structure, i.e. if there are
            three spins (1, 2, 3), then in the mixed part of the stream, after averaging
            but before the linear transformation, cyclic equivariance means the inputs
            are [(1, 2, 3), (2, 3, 1), (3, 1, 2)]. If False, then the inputs are
            [(1, 2, 3), (1, 2, 3), (1, 2, 3)] (as in the original FermiNet).
            When there are only two spins (spin-1/2 case), then this is equivalent to
            true spin equivariance. Defaults to False (original FermiNet).
    """

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
            use_bias=self.use_bias,
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
            all_spins = jnp.concatenate(split_concat, axis=-2)
            dense_mixed = self._mixed_dense(all_spins)
            dense_mixed_split = jnp.split(dense_mixed, len(split_concat), axis=-2)
        else:
            split_concat = jnp.concatenate(split_means, axis=-1)
            dense_mixed = self._mixed_dense(split_concat)
            dense_mixed_split = [dense_mixed, dense_mixed]
        return dense_mixed_split

    def _compute_transformed_2e_means(self, in_2e):
        # [spin: (..., nelec[spin], nelec_total, d)]
        split_2e = jnp.split(in_2e, self.spin_split, axis=-3)

        concat_2e = []
        for spin in range(len(split_2e)):
            split_arrays = _split_mean(
                self.spin_split, split_2e[spin], axis=-2, keepdims=False
            )  # [spin1: [spin2: (..., nelec[spin1], d)]]
            if self.cyclic_spins:
                concat_arrays = _rolled_concat(split_arrays, spin)
            else:
                concat_arrays = jnp.concatenate(split_arrays, axis=-1)
            # concat_arrays is [spin: (..., nelec[spin1], d * nspins)]
            concat_2e.append(concat_arrays)

        all_spins = jnp.concatenate(concat_2e, axis=-2)
        dense_2e = self._dense_2e(all_spins)
        return jnp.split(dense_2e, self.spin_split, axis=-2)

    def __call__(self, in_1e: jnp.ndarray, in_2e: jnp.ndarray = None) -> jnp.ndarray:
        dense_unmixed = self._unmixed_dense(in_1e)
        dense_unmixed_split = jnp.split(dense_unmixed, self.spin_split, axis=-2)

        split_1e_means = _split_mean(self.spin_split, in_1e, axis=-2, keepdims=True)
        dense_mixed_split = self._compute_transformed_1e_means(split_1e_means)
        dense_out = _tree_sum(dense_unmixed_split, dense_mixed_split)

        if in_2e is not None:
            dense_2e_split = self._compute_transformed_2e_means(in_2e)
            dense_out = _tree_sum(dense_out, dense_2e_split)

        dense_out_concat = jnp.concatenate(dense_out, axis=-2)
        nonlinear_out = self._activation_fn(dense_out_concat)

        if self.skip_connection and _valid_skip(in_1e, nonlinear_out):
            nonlinear_out = nonlinear_out + in_1e

        return nonlinear_out


class FermiNetTwoElectronLayer(flax.linen.Module):
    """A single layer in the two-electron stream of the FermiNet equivariance.

    Attributes:
        ndense (int): number of dense nodes
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function. Has the signature
            jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and output match. Defaults to True.
    """

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
    """The FermiNet equivariant part up until, but not including, the orbitals.

    Repeated composition of the residual blocks in the parallel one-electron and
    two-electron streams.

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_list: (list of (int, int)): number of dense nodes in each of the
            residual blocks, (ndense_1e, ndense_2e). The length of this list determines
            the number of residual blocks which are composed on top of the input streams
        kernel_initializer_unmixed (WeightInitializer): kernel initializer for the
            unmixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the previous one-electron stream output. Has
            signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_1e_stream (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_2e_2e_stream (WeightInitializer): kernel initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_1e_stream (WeightInitializer): bias initializer for the
            one-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_2e_stream (WeightInitializer): bias initializer for the
            two-electron stream. Has signature (key, shape, dtype) -> jnp.ndarray
        activation_fn (Activation): activation function in the electron streams. Has
            the signature jnp.ndarray -> jnp.ndarray (shape is preserved)
        use_bias (bool, optional): whether to add a bias term in the electron streams.
            Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and outputs of the streams match. Defaults
            to True.
        cyclic_spins (bool, optional): whether the the concatenation in the one-electron
            stream should satisfy a cyclic equivariance structure, i.e. if there are
            three spins (1, 2, 3), then in the mixed part of the stream, after averaging
            but before the linear transformation, cyclic equivariance means the inputs
            are [(1, 2, 3), (2, 3, 1), (3, 1, 2)]. If False, then the inputs are
            [(1, 2, 3), (1, 2, 3), (1, 2, 3)] (as in the original FermiNet).
            When there are only two spins (spin-1/2 case), then this is equivalent to
            true spin equivariance. Defaults to False (original FermiNet).
    """

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

        for layer, features in enumerate(self.ndense_list):
            advance_2e = True
            if layer == len(self.ndense_list) - 1:
                advance_2e = False
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
                advance_2e=advance_2e,
            )(stream_1e, stream_2e)

        return stream_1e, r_ei


class SplitDense(flax.linen.Module):
    """Split input on the 2nd-to-last axis and apply unique Dense layers to each split.

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense (Sequence[int]): sequence of integers specifying the number of dense
            nodes in the unique dense layer applied to each split of the input. This
            determines the output shapes for each split, i.e. the outputs are shaped
            (..., split_size[i], ndense[i])
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
    """

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
    """Make the FermiNet orbitals (parallel linear layers with exp decay envelopes).

    Attributes:
        spin_split (int or Sequence[int]): number of spins to split inputs equally,
            or specified sequence of locations to split along the electron axis. E.g.,
            if nelec = 10, and `spin_split` = 2, then the electrons are split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the electrons are split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        norbitals (Sequence[int]): sequence of integers specifying the number of
            orbitals to create for each spin. This determines the output shapes for each
            split, i.e. the outputs are shaped (..., split_size[i], norbitals[i])
        kernel_initializer_linear (WeightInitializer): kernel initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer for the
            decay rate in the exponential envelopes. If `isotropic_decay` is True, then
            this initializes a single decay rate number per ion and orbital. If
            `isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer for the
            linear combination over the ions of exponential envelopes. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer_linear (WeightInitializer): bias initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term to the linear part of the
            orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

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
        nion = x.shape[-2]
        # x_nion is (..., nelec, d, nion)
        x_nion = jnp.swapaxes(x, axis1=-1, axis2=-2)
        x_nion = jnp.expand_dims(x_nion, axis=-1)
        # split_out has shape [(... nelec, d, 1, norbitals)] * nion,
        # this applies nion parallel maps which go 1 -> norbitals
        split_out = SplitDense(
            nion,
            (norbitals,) * nion,
            self.kernel_initializer_envelope_dim,
            zeros,
            use_bias=False,
        )(x_nion)
        # concat_out is (..., nelec, d, nion, norbitals)
        concat_out = jnp.concatenate(split_out, axis=-2)
        return jnp.swapaxes(concat_out, axis1=-1, axis2=-3)

    def _anisotropy(self, x: jnp.ndarray, norbitals: int) -> jnp.ndarray:
        batch_shapes = x.shape[:-2]
        nion = x.shape[-2]
        d = x.shape[-1]
        # split_out has shape [(... nelec, 1, d * norbitals)] * nion,
        # this applies nion parallel maps which go d -> d * norbitals
        split_out = SplitDense(
            nion,
            (d * norbitals,) * nion,
            self.kernel_initializer_envelope_dim,
            zeros,
            use_bias=False,
        )(x)

        concat_out = jnp.concatenate(split_out, axis=-2)  # (..., nion, d * norbitals)
        out = jnp.reshape(concat_out, batch_shapes + (nion, d, norbitals))
        out = jnp.swapaxes(out, axis1=-1, axis2=-3)  # (..., norbitals, d, nion)
        out = jnp.swapaxes(out, axis1=-1, axis2=-2)  # (..., norbitals, nion, d)

        return out

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
