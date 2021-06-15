"""Permutation equivariant functions."""
import functools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from vmcnet.physics.potential import _compute_displacements
from vmcnet.models.weights import WeightInitializer, zeros

Activation = Callable[[jnp.ndarray], jnp.ndarray]


def _split_mean(
    x: jnp.ndarray,
    splits: Union[int, Sequence[int]],
    axis: int = -2,
    keepdims: bool = True,
) -> List[jnp.ndarray]:
    """Split x on an axis and take the mean over that axis in each of the splits."""
    split_x = jnp.split(x, splits, axis=axis)
    split_x_mean = jax.tree_map(
        functools.partial(jnp.mean, axis=axis, keepdims=keepdims), split_x
    )
    return split_x_mean


def _rolled_concat(arrays: List[jnp.ndarray], n: int, axis: int = -1) -> jnp.ndarray:
    """Concatenate a list of arrays starting from the nth and wrapping back around.

    The input list of arrays must all have the same shapes, except for along `axis`.
    """
    return jnp.concatenate(arrays[n:] + arrays[:n], axis=axis)


def _tree_sum(tree1, tree2):
    """Leaf-wise sum of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a + b, tree1, tree2)


def _tree_prod(tree1, tree2):
    """Leaf-wise product of two pytrees with the same structure."""
    return jax.tree_map(lambda a, b: a * b, tree1, tree2)


def _valid_skip(x: jnp.ndarray, y: jnp.ndarray) -> bool:
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
        input_1e = r_ei
        if include_ei_norm:
            input_norm = jnp.linalg.norm(input_1e, axis=-1, keepdims=True)
            input_with_norm = jnp.concatenate([input_1e, input_norm], axis=-1)
            input_1e = jnp.reshape(input_with_norm, input_with_norm.shape[:-2] + (-1,))
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
        eye_n = jnp.expand_dims(jnp.eye(n), axis=-1)
        r_ee_diag_ones = input_2e + eye_n
        r_ee_norm = jnp.linalg.norm(r_ee_diag_ones, axis=-1, keepdims=True) * (
            1.0 - eye_n
        )
        input_2e = jnp.concatenate([input_2e, r_ee_norm], axis=-1)
    return input_2e


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
        """Setup Dense layers."""
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
        """Apply a dense layer to the concatenated averages of the 1e stream.

        The mixing of the one-electron part of the one-electron stream takes the form

        [i: (..., n[i], d)]
            -> average along particle dim to get [i: (..., 1, d)]
            -> concatenate all averages for each spin to get [i: (..., 1, d * nspins)]
            -> apply the same linear transformation for all i to get [i: (..., 1, d')]

        This function does the last two steps.

        There is a choice about how to do the concatenation step. For all spins the
        concatenation can be exactly the same, say [(1, 2, 3), (1, 2, 3), (1, 2, 3)],
        which is the approach in the original FermiNet paper, or different, for which
        there may be many possibilities. Here, if self.cyclic_spins is True, then the
        concatenation done is [(1, 2, 3), (2, 3, 1), (3, 1, 2)], which obeys
        an equivariance with respect to cyclic permutations.

        When there are just two spins, this cyclic equivariance is the same has complete
        permutation equivariance (since the cyclic group of order 2 is group isomorphic
        to the permutation group of order 2, a single flip). For more than 2 spins, it's
        not clear if either approach is better from a theoretical standpoint, as both
        impose an ordering.
        """
        nspins = len(split_means)
        if self.cyclic_spins:
            # re-concatenate the averages but as [i: [i, ..., n, 1, ..., i-1]] along
            # the last dimension
            split_concat = [_rolled_concat(split_means, idx) for idx in range(nspins)]

            # concatenate on axis=-2 so a single dense layer can be batch applied to
            # every concatenation
            all_spins = jnp.concatenate(split_concat, axis=-2)
            dense_mixed = self._mixed_dense(all_spins)

            # split the results of the batch applied dense layer back into the spins
            dense_mixed_split = jnp.split(dense_mixed, len(split_concat), axis=-2)
        else:
            split_concat = jnp.concatenate(split_means, axis=-1)
            dense_mixed = self._mixed_dense(split_concat)
            dense_mixed_split = [dense_mixed] * nspins
        return dense_mixed_split

    def _compute_transformed_2e_means(self, in_2e):
        """Apply a dense layer to the concatenated averages of the 2e stream.

        The mixing of the two-electron part of the one-electron stream takes the form

        (..., n_total, n_total, d)
            -> split along a particle axis to get [i: (..., n[i], n_total, d)]
            -> for each i, do a split and average over the other particle axis (but
                don't keep the averaged axis) to get [i: [j: (..., n[i], d)]]
            -> for each i, concatenate the splits to get [i: (..., n[i], d * nspins)]
            -> apply the same linear transformation for all i to get
                [i: (..., n[i], d')]

        As in the mixing of the one-electron part, the concatenation step comes with a
        choice of what order in which to concatenate the averages. Here if
        self.cyclic_spins is True, the ith spin is concatenated so that the j=i average
        is first in the concatenation and the other spins follow cyclically, which is
        invariant under cyclic permutations of the spin. If self.cyclic_spins is False,
        all spins are concatenated in the same order, the spin order induced from the
        particle ordering and the specified spin split.
        """
        # split to get [i: (..., n[i], n_total, d)]
        split_2e = jnp.split(in_2e, self.spin_split, axis=-3)

        # for each i, do a split and average along axis=-2, then concatenate
        concat_2e = []
        for spin in range(len(split_2e)):
            split_arrays = _split_mean(
                split_2e[spin], self.spin_split, axis=-2, keepdims=False
            )  # [j: (..., n[i], d)]
            if self.cyclic_spins:
                # for the ith spin, concatenate as [i, ..., n, 1, ..., i-1] along the
                # last axis
                concat_arrays = _rolled_concat(split_arrays, spin)
            else:
                # otherwise, for all i, concatenate the averages over [1, ..., n] in
                # that order
                concat_arrays = jnp.concatenate(split_arrays, axis=-1)
            concat_2e.append(concat_arrays)  # [i: (..., n[i], d * nspins)]

        # reconcatenate along the split axis to batch apply the same dense layer for all
        # i, then split over the spins again before returning
        all_spins = jnp.concatenate(concat_2e, axis=-2)
        dense_2e = self._dense_2e(all_spins)
        return jnp.split(dense_2e, self.spin_split, axis=-2)

    def __call__(self, in_1e: jnp.ndarray, in_2e: jnp.ndarray = None) -> jnp.ndarray:
        """Add dense outputs on unmixed, mixed, and 2e terms to get the 1e output.

        This implementation breaks the one-electron stream into three parts:
            1) the unmixed one-particle part, which is a linear transformation applied
                in parallel for each particle to the inputs
            2) the mixed one-particle part, which is a linear transformation applied to
                the averages of the inputs (concatenated over spin)
            3) the two-particle part, which is a linear transformation applied in
                parallel for each particle to the average of the input interactions
                between that particle and all the other particles.

        For 1), we take `in_1e` of shape (..., n_total, d_1e), batch apply a linear
        transformation to get (..., n_total, d'), and split over the spins i to get
        [i: (..., n[i], d')].

        For 2), we split `in_1e` over the spins along the particle axis to get
        [i: (..., n[i], d_1e)], average over each spin to get [i: (..., 1, d_1e)],
        concatenate all averages for each spin to get [i: (..., 1, d_1e * nspins)], and
        apply a linear transformation to get [i: (..., 1, d')].

        For 3) we split in_2e of shape (..., n_total, n_total, d_2e) over the spins
        along a particle axis to get [i: (..., n[i], n_total, d_2e)], average over the
        other particle axis to get [i: [j: (..., n[i], d_2e)]], concatenate the averages
        for each spin to get [i: (..., n[i], d_2e * nspins)], and apply a linear
        transformation to get [i: (..., n[i], d')].

        Finally, for each spin, we add the three parts, each equivariant or symmetric,
        to get a final equivariant linear transformation of the inputs, to which a
        non-linearity is then applied and a skip connection optionally added.
        """
        dense_unmixed = self._unmixed_dense(in_1e)
        dense_unmixed_split = jnp.split(dense_unmixed, self.spin_split, axis=-2)

        split_1e_means = _split_mean(in_1e, self.spin_split, axis=-2, keepdims=True)
        dense_mixed_split = self._compute_transformed_1e_means(split_1e_means)

        # adds the unmixed [i: (..., n[i], d')] to the mixed [i: (..., 1, d')] to get
        # an equivariant function. Without the two-electron mixing, this is a spinful
        # version of DeepSet's Lemma 3: https://arxiv.org/pdf/1703.06114.pdf
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
        """Setup Dense layer."""
        # workaround MyPy's typing error for callable attribute
        self._activation_fn = self.activation_fn
        self._dense = flax.linen.Dense(
            self.ndense,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply a Dense layer in parallel to all electron pairs.

        The expected use-case of this is to batch apply a dense layer to an input x of
        shape (..., n_total, n_total, d), getting an output of shape
        (..., n_total, n_total, d'), and optionally adding a skip connection. The
        function itself is just a standard residual network layer.
        """
        dense_out = self._dense(x)
        nonlinear_out = self._activation_fn(dense_out)

        if self.skip_connection and _valid_skip(x, nonlinear_out):
            nonlinear_out = nonlinear_out + x

        return nonlinear_out


class FermiNetResidualBlock(flax.linen.Module):
    """A single residual block in the FermiNet equivariant part.

    Combines the one-electron and two-electron streams.

    Attributes:
        one_electron_layer (Callable): function which takes in a previous one-electron
            stream output and two-electron stream output and mixes/transforms them to
            create a new one-electron stream output. Has the signature:
            (array of shape (..., n, d_1e), optional array of shape (..., n, n, d_2e))
            -> array of shape (..., n, d_1e')
        two_electron_layer (Callable): function which takes in a previous two-electron
            stream output and batch applies a Dense layer along the last axis. Has the
            signature:
            array of shape (..., n, n, d_2e) -> array of shape (..., n, n, d_2e')
    """

    one_electron_layer: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
    two_electron_layer: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        """Setup called one- and two- electron layers."""
        self._one_electron_layer = self.one_electron_layer
        self._two_electron_layer = self.two_electron_layer

    def __call__(
        self, in_1e: jnp.ndarray, in_2e: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Apply the one-electron layer and optionally the two-electron layer."""
        out_1e = self._one_electron_layer(in_1e, in_2e)

        out_2e = in_2e
        if self.two_electron_layer is not None and in_2e is not None:
            out_2e = self._two_electron_layer(in_2e)

        return out_1e, out_2e


class FermiNetBackflow(flax.linen.Module):
    """The FermiNet equivariant part up until, but not including, the orbitals.

    Repeated composition of the residual blocks in the parallel one-electron and
    two-electron streams.

    Attributes:
        residual_blocks (Sequence): sequence of callable residual blocks which apply
            the one- and two- electron layers. Each residual block has the signature
            (in_1e, optional in_2e) -> (out_1e, optional out_2e), where
                in_1e has shape (..., n, d_1e)
                out_1e has shape (..., n, d_1e')
                in_2e has shape (..., n, n, d_2e)
                out_2d has shape (..., n, n, d_2e')
        ion_pos (jnp.ndarray, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_2e_stream (bool, optional): whether to include pairwise electron
            displacements/distances in the input. Defaults to True.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.
    """

    residual_blocks: Sequence[
        Callable[
            [jnp.ndarray, Optional[jnp.ndarray]],
            Tuple[jnp.ndarray, Optional[jnp.ndarray]],
        ]
    ]
    ion_pos: Optional[jnp.ndarray] = None
    include_2e_stream: bool = True
    include_ei_norm: bool = True
    include_ee_norm: bool = True

    def setup(self):
        """Setup called residual blocks."""
        self._residual_block_list = [block for block in self.residual_blocks]

    def __call__(
        self, elec_pos: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Create input streams and iteratively apply residual blocks."""
        stream_1e, stream_2e, r_ei = compute_input_streams(
            elec_pos,
            self.ion_pos,
            include_ee=self.include_2e_stream,
            include_ei_norm=self.include_ei_norm,
            include_ee_norm=self.include_ee_norm,
        )

        for block in self._residual_block_list:
            stream_1e, stream_2e = block(stream_1e, stream_2e)

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
        ndense_per_spin (Sequence[int]): sequence of integers specifying the number of
            dense nodes in the unique dense layer applied to each split of the input.
            This determines the output shapes for each split, i.e. the outputs are
            shaped (..., split_size[i], ndense[i])
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> jnp.ndarray
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
    """

    spin_split: Union[int, Sequence[int]]
    ndense_per_spin: Sequence[int]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    use_bias: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        if isinstance(self.spin_split, int):
            nspins = self.spin_split
        else:
            nspins = len(self.spin_split) + 1

        if len(self.ndense_per_spin) != nspins:
            raise ValueError(
                "Incorrect number of dense output shapes specified for number of "
                "spins, should be one shape per spin: shapes {} specified for the "
                "given spin_split {}".format(self.ndense_per_spin, self.spin_split)
            )

        self._dense_layers = [
            flax.linen.Dense(
                self.ndense_per_spin[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
            )
            for i in range(nspins)
        ]

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Split the input and apply a dense layer to each split."""
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
    norbitals_per_spin: Sequence[int]
    kernel_initializer_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_linear: WeightInitializer
    use_bias: bool = True
    isotropic_decay: bool = False

    def _isotropy_on_leaf(self, r_ei_leaf: jnp.ndarray, norbitals: int) -> jnp.ndarray:
        """Isotropic scaling of the electron-ion displacements."""
        nion = r_ei_leaf.shape[-2]

        # swap axes around and inject an axis to go from
        # r_ei_leaf: (..., nelec, nion, d) -> x_nion: (..., nelec, d, nion, 1)
        x_nion = jnp.swapaxes(r_ei_leaf, axis1=-1, axis2=-2)
        x_nion = jnp.expand_dims(x_nion, axis=-1)

        # split x_nion along the ion axis and apply nion parallel maps 1 -> norbitals
        # along the last axis, resulting in
        # [i: (..., nelec, d, 1, norbitals)], where i is the ion index
        split_out = SplitDense(
            nion,
            (norbitals,) * nion,
            self.kernel_initializer_envelope_dim,
            zeros,
            use_bias=False,
        )(x_nion)

        # concatenate over the ion axis, then swap axes to get
        # an output of shape (..., nelec, norbitals, nion, d)
        concat_out = jnp.concatenate(split_out, axis=-2)
        return jnp.swapaxes(concat_out, axis1=-1, axis2=-3)

    def _anisotropy_on_leaf(
        self, r_ei_leaf: jnp.ndarray, norbitals: int
    ) -> jnp.ndarray:
        """Anisotropic scaling of the electron-ion displacements."""
        batch_shapes = r_ei_leaf.shape[:-2]
        nion = r_ei_leaf.shape[-2]
        d = r_ei_leaf.shape[-1]

        # split x along the ion axis and apply nion parallel maps d -> d * norbitals
        # along the last axis, resulting in [i: (..., nelec, 1, d * norbitals)]
        split_out = SplitDense(
            nion,
            (d * norbitals,) * nion,
            self.kernel_initializer_envelope_dim,
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

    def _compute_exponential_envelopes_on_leaf(
        self, r_ei_leaf: jnp.ndarray, norbitals: int, isotropic: bool = False
    ) -> jnp.ndarray:
        """Pick a type of exp envelope and multiply by the linear part element-wise."""
        if isotropic:
            conv_out = self._isotropy_on_leaf(r_ei_leaf, norbitals)
        else:
            conv_out = self._anisotropy_on_leaf(r_ei_leaf, norbitals)
        # conv_out has shape (..., nelec, norbitals, nion, d)
        distances = jnp.linalg.norm(conv_out, axis=-1)
        inv_exp_distances = jnp.exp(-distances)  # (..., nelec, norbitals, nion)
        lin_comb_nion = flax.linen.Dense(
            1, kernel_init=self.kernel_initializer_envelope_ion, use_bias=False
        )(inv_exp_distances)
        return jnp.squeeze(lin_comb_nion, axis=-1)  # (..., nelec, norbitals)

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray, r_ei: jnp.ndarray = None) -> List[jnp.ndarray]:
        """Apply a dense layer R -> R^n for each spin and multiply by exp envelopes."""
        orbs = SplitDense(
            self.spin_split,
            self.norbitals_per_spin,
            self.kernel_initializer_linear,
            self.bias_initializer_linear,
            use_bias=self.use_bias,
        )(x)
        if r_ei is not None:
            r_ei_split = jnp.split(r_ei, self.spin_split, axis=-3)
            exp_envelopes = jax.tree_map(
                functools.partial(
                    self._compute_exponential_envelopes_on_leaf,
                    isotropic=self.isotropic_decay,
                ),
                r_ei_split,
                list(self.norbitals_per_spin),
            )
            orbs = _tree_prod(orbs, exp_envelopes)
        return orbs
