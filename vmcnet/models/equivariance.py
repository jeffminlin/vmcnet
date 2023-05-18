"""Permutation equivariant functions."""
import functools
from typing import Callable, Optional, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp

from vmcnet.physics.potential import compute_displacements
from vmcnet.utils.pytree_helpers import tree_prod, tree_sum
from vmcnet.utils.typing import Array, ArrayList, InputStreams, ParticleSplit
from .core import (
    Activation,
    Dense,
    ElementWiseMultiply,
    Module,
    _split_mean,
    _valid_skip,
    compute_ee_norm_with_safe_diag,
    get_nsplits,
    split,
)
from flax.linen import SelfAttention  # noqa
from .jastrow import _anisotropy_on_leaf, _isotropy_on_leaf
from .weights import WeightInitializer, get_bias_initializer


def _rolled_concat(arrays: ArrayList, n: int, axis: int = -1) -> Array:
    """Concatenate a list of arrays starting from the nth and wrapping back around.

    The input list of arrays must all have the same shapes, except for along `axis`.
    """
    return jnp.concatenate(arrays[n:] + arrays[:n], axis=axis)


def compute_input_streams(
    elec_pos: Array,
    ion_pos: Optional[Array] = None,
    include_2e_stream: bool = True,
    include_ei_norm: bool = True,
    ei_norm_softening: chex.Scalar = 0.0,
    include_ee_norm: bool = True,
    ee_norm_softening: chex.Scalar = 0.0,
) -> InputStreams:
    """Create input streams with electron and optionally ion data.

    If `ion_pos` is given, computes the electron-ion displacements (i.e. nuclear
    coordinates) and concatenates/flattens them along the ion dimension. If
    `include_ei_norm` is True, then the distances are also concatenated, so the map is
    elec_pos = (..., nelec, d) -> input_1e = (..., nelec, nion * (d + 1)).

    If `include_2e_stream` is True, then a two-electron stream of shape
    (..., nelec, nelec, d) is also computed and returned (otherwise None is returned).
    If `include_ee_norm` is True, then this becomes (..., nelec, nelec, d + 1) by
    concatenating pairwise distances onto the stream.

    Args:
        elec_pos (Array): electron positions of shape (..., nelec, d)
        ion_pos (Array, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_2e_stream (bool, optional): whether to compute pairwise electron
            displacements/distances. Defaults to True.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.
        ei_norm_softening (float, optional): constant used to soften the cusp of the ei
            norm. If set to c, then an ei norm of r is replaced by sqrt(r^2 + c^2) - c.
            Defaults to 0.
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.
        ee_norm_softening (float, optional): constant used to soften the cusp of the ee
            norm. If set to c, then an ee norm of r is replaced by sqrt(r^2 + c^2) - c.
            Defaults to 0.

    Returns:
        (
            Array,
            Optional[Array],
            Optional[Array],
            Optional[Array],
        ):

        first output: one-electron input of shape (..., nelec, d'), where
            d' = d if `ion_pos` is None,
            d' = nion * d if `ion_pos` is given and `include_ei_norm` is False, and
            d' = nion * (d + 1) if `ion_pos` is given and `include_ei_norm` is True.

        second output: two-electron input of shape (..., nelec, nelec, d'), where
            d' = d if `include_ee_norm` is False, and
            d' = d + 1 if `include_ee_norm` is True

        third output: electron-ion displacements of shape (..., nelec, nion, d)

        fourth output: electron-electron displacements of shape (..., nelec, nelec, d)

        If `include_2e_stream` is False, then the second and fourth outputs are None. If
        `ion_pos` is None, then the third output is None.
    """
    input_1e, r_ei = compute_electron_ion(
        elec_pos, ion_pos, include_ei_norm, ei_norm_softening
    )
    input_2e = None
    r_ee = None
    if include_2e_stream:
        input_2e, r_ee = compute_electron_electron(
            elec_pos, include_ee_norm, ee_norm_softening
        )
    return input_1e, input_2e, r_ei, r_ee


def compute_electron_ion(
    elec_pos: Array,
    ion_pos: Optional[Array] = None,
    include_ei_norm: bool = True,
    ei_norm_softening: chex.Scalar = 0.0,
) -> Tuple[Array, Optional[Array]]:
    """Compute electron-ion displacements and optionally add on the distances.

    Args:
        elec_pos (Array): electron positions of shape (..., nelec, d)
        ion_pos (Array, optional): locations of (stationary) ions to compute
            relative electron positions, 2-d array of shape (nion, d). Defaults to None.
        include_ei_norm (bool, optional): whether to include electron-ion distances in
            the one-electron input. Defaults to True.
        ei_norm_softening (float, optional): constant used to soften the cusp of the ei
            norm. If set to c, then an ei norm of r is replaced by sqrt(r^2 + c^2) - c.
            Defaults to 0.

    Returns:
        (Array, Optional[Array]):

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
        r_ei = compute_displacements(input_1e, ion_pos)
        input_1e = r_ei
        if include_ei_norm:
            input_norm = jnp.linalg.norm(input_1e, axis=-1, keepdims=True)
            softened_input_norm = (
                jnp.sqrt(input_norm**2 + ei_norm_softening**2) - ei_norm_softening
            )
            input_with_norm = jnp.concatenate([input_1e, softened_input_norm], axis=-1)
            input_1e = jnp.reshape(input_with_norm, input_with_norm.shape[:-2] + (-1,))
    return input_1e, r_ei


def compute_electron_electron(
    elec_pos: Array,
    include_ee_norm: bool = True,
    ee_norm_softening: chex.Scalar = 0.0,
) -> Tuple[Array, Array]:
    """Compute electron-electron displacements and optionally add on the distances.

    Args:
        elec_pos (Array): electron positions of shape (..., nelec, d)
        include_ee_norm (bool, optional): whether to include electron-electron distances
            in the two-electron input. Defaults to True.
        ee_norm_softening (float, optional): constant used to soften the cusp of the ee
            norm. If set to c, then an ee norm of r is replaced by sqrt(r^2 + c^2) - c.
            Defaults to 0.

    Returns:
        (Array, Array):

        first output: two-electron input of shape (..., nelec, nelec, d'), where
            d' = d if `include_ee_norm` is False, and
            d' = d + 1 if `include_ee_norm` is True

        second output: two-electron displacements of shape (..., nelec, nelec, d)
    """
    r_ee = compute_displacements(elec_pos, elec_pos)
    input_2e = r_ee
    if include_ee_norm:
        r_ee_norm = compute_ee_norm_with_safe_diag(
            r_ee, softening_term=ee_norm_softening
        )
        input_2e = jnp.concatenate([input_2e, r_ee_norm], axis=-1)
    return input_2e, r_ee


def _transformer_mix(
    x: Array,
    self_attention_layer: Callable[[Array], Array],
    splits: ParticleSplit,
    axis: int = -2,
) -> ArrayList:
    """Split x on an axis and apply the self attention layer to each of the splits."""
    # in_1e has shape (..., n, d_1e)
    # split_x has shape [i: (..., n[i], d_1e)]

    split_x = split(x, splits, axis=axis)
    split_x_mix = jax.tree_map(self_attention_layer, split_x)

    return split_x_mix


class FermiNetOneElectronLayer(Module):
    """A single layer in the one-electron stream of the FermiNet equivariant part.

    Attributes:
        spin_split (ParticleSplit): number of spins to split the input equally,
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
            signature (key, shape, dtype) -> Array
        kernel_initializer_transformer (WeightInitializer): kernel initializer for the
            mixed part of the transformer stream. This initializes the part of the
            dense kernel which multiplies the average of the previous transformer
            stream output. Has signature (key, shape, dtype) -> Array
        kernel_initializer_mixed (WeightInitializer): kernel initializer for the
            mixed part of the one-electron stream. This initializes the part of the
            dense kernel which multiplies the average of the previous one-electron
            stream output. Has signature (key, shape, dtype) -> Array
        kernel_initializer_2e (WeightInitializer): kernel initializer for the
            two-electron part of the one-electron stream. This initializes the part of
            the dense kernel which multiplies the average of the previous two-electron
            stream which is mixed into the one-electron stream. Has signature
            (key, shape, dtype) -> Array
        bias_initializer (WeightInitializer): bias initializer for the transformer
            steam. Has signature (key, shape, dtype) -> Array
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> Array
        activation_fn (Activation): activation function. Has the signature
            Array -> Array (shape is preserved)
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and output match. Defaults to True.
        skip_connection_scale (float, optional): quantity to scale the final output by
            if a skip connection is added. Defaults to 1.0.
        cyclic_spins (bool, optional): whether the the concatenation in the one-electron
            stream should satisfy a cyclic equivariance structure, i.e. if there are
            three spins (1, 2, 3), then in the mixed part of the stream, after averaging
            but before the linear transformation, cyclic equivariance means the inputs
            are [(1, 2, 3), (2, 3, 1), (3, 1, 2)]. If False, then the inputs are
            [(1, 2, 3), (1, 2, 3), (1, 2, 3)] (as in the original FermiNet).
            When there are only two spins (spin-1/2 case), then this is equivalent to
            true spin equivariance. Defaults to False (original FermiNet).
        use_transformer (bool, optional): whether to use the transformer stream.
        num_heads (int, optional): number of heads. If num_heads == 1, then multi-head
            attention layers are reduced to self-attention layers. if use_transformer is
            False, then the num_heads argument is ignored. Defaults to 1.
    """

    spin_split: ParticleSplit
    ndense: int
    kernel_initializer_transformer: WeightInitializer
    kernel_initializer_unmixed: WeightInitializer
    kernel_initializer_mixed: WeightInitializer
    kernel_initializer_2e: WeightInitializer
    bias_initializer: WeightInitializer
    bias_initializer_transformer: WeightInitializer
    activation_fn: Activation
    use_bias: bool = True
    skip_connection: bool = True
    skip_connection_scale: float = 1.0
    cyclic_spins: bool = True
    use_transformer: bool = False
    num_heads: int = 1

    def setup(self):
        """Setup Dense layers."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._activation_fn = self.activation_fn

        self._unmixed_dense = Dense(
            self.ndense,
            kernel_init=self.kernel_initializer_unmixed,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )
        self._dense_2e = Dense(
            self.ndense, kernel_init=self.kernel_initializer_2e, use_bias=False
        )

        if self.use_transformer:
            self._attention_1e = SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.ndense * self.num_heads,
                out_features=self.ndense,
                kernel_init=self.kernel_initializer_transformer,
                bias_init=self.bias_initializer_transformer,
            )
        else:
            self._mixed_dense = Dense(
                self.ndense, kernel_init=self.kernel_initializer_mixed, use_bias=False
            )

    def _compute_transformed_1e_means(self, split_means: ArrayList) -> ArrayList:
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
            split_concats = [_rolled_concat(split_means, idx) for idx in range(nspins)]

            # concatenate on axis=-2 so a single dense layer can be batch applied to
            # every concatenation
            all_spins = jnp.concatenate(split_concats, axis=-2)
            dense_mixed = self._mixed_dense(all_spins)

            # split the results of the batch applied dense layer back into the spins
            dense_mixed_split = split(dense_mixed, len(split_concats), axis=-2)
        else:
            split_concat = jnp.concatenate(split_means, axis=-1)
            dense_mixed = self._mixed_dense(split_concat)
            dense_mixed_split = [dense_mixed] * nspins
        return dense_mixed_split

    def _compute_transformed_2e_means(self, in_2e: Array) -> ArrayList:
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
        split_2e = split(in_2e, self.spin_split, axis=-3)

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
        return split(dense_2e, self.spin_split, axis=-2)

    def _compute_mixed_split(self, in_1e: Array) -> ArrayList:
        """Compute the 1e mixed for the given input.

        If use_transformer is True, then the mixed is computed using transformer layer.
        Else, the mixed is computed using the dense layer with the reduce-mean layer.
        """
        if self.use_transformer:
            return _transformer_mix(in_1e, self._attention_1e, self.spin_split, axis=-2)
        else:
            split_1e_means = _split_mean(in_1e, self.spin_split, axis=-2, keepdims=True)
            return self._compute_transformed_1e_means(split_1e_means)

    def __call__(  # type: ignore[override]
        self, in_1e: Array, in_2e: Optional[Array] = None
    ) -> Array:
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

        Args:
            in_1e (Array): array of shape (..., n_total, d_1e)
            in_2e (Array, optional): array of shape (..., n_total, n_total, d_2e).
                Defaults to None.

        Returns:
            Array of shape (..., n_total, self.ndense), the output one-electron
            stream
        """
        dense_unmixed = self._unmixed_dense(in_1e)
        dense_unmixed_split = split(dense_unmixed, self.spin_split, axis=-2)

        dense_mixed_split = self._compute_mixed_split(in_1e)

        # adds the unmixed [i: (..., n[i], d')] to the mixed [i: (..., 1, d')] to get
        # an equivariant function. Without the two-electron mixing, this is a spinful
        # version of DeepSet's Lemma 3: https://arxiv.org/pdf/1703.06114.pdf
        dense_out = tree_sum(dense_unmixed_split, dense_mixed_split)

        if in_2e is not None:
            dense_2e_split = self._compute_transformed_2e_means(in_2e)
            dense_out = tree_sum(dense_out, dense_2e_split)

        dense_out_concat = jnp.concatenate(dense_out, axis=-2)
        nonlinear_out = self._activation_fn(dense_out_concat)

        if self.skip_connection and _valid_skip(in_1e, nonlinear_out):
            nonlinear_out = self.skip_connection_scale * (nonlinear_out + in_1e)

        return nonlinear_out


class FermiNetTwoElectronLayer(Module):
    """A single layer in the two-electron stream of the FermiNet equivariance.

    Attributes:
        ndense (int): number of dense nodes
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> Array
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> Array
        activation_fn (Activation): activation function. Has the signature
            Array -> Array (shape is preserved)
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
        skip_connection (bool, optional): whether to add residual skip connections
            whenever the shapes of the input and output match. Defaults to True.
        skip_connection_scale (float, optional): quantity to scale the final output by
            if a skip connection is added. Defaults to 1.0.
    """

    ndense: int
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer
    activation_fn: Activation
    use_bias: bool = True
    skip_connection: bool = True
    skip_connection_scale: float = 1.0

    def setup(self):
        """Setup Dense layer."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._activation_fn = self.activation_fn
        self._dense = Dense(
            self.ndense,
            kernel_init=self.kernel_initializer,
            bias_init=self.bias_initializer,
            use_bias=self.use_bias,
        )

    def __call__(self, x: Array) -> Array:  # type: ignore[override]
        """Apply a Dense layer in parallel to all electron pairs.

        The expected use-case of this is to batch apply a dense layer to an input x of
        shape (..., n_total, n_total, d), getting an output of shape
        (..., n_total, n_total, d'), and optionally adding a skip connection. The
        function itself is just a standard residual network layer.
        """
        dense_out = self._dense(x)
        nonlinear_out = self._activation_fn(dense_out)

        if self.skip_connection and _valid_skip(x, nonlinear_out):
            nonlinear_out = self.skip_connection_scale * (nonlinear_out + x)

        return nonlinear_out


class FermiNetResidualBlock(Module):
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

    one_electron_layer: Callable[[Array, Optional[Array]], Array]
    two_electron_layer: Optional[Callable[[Array], Array]] = None

    def setup(self):
        """Setup called one- and two- electron layers."""
        self._one_electron_layer = self.one_electron_layer
        self._two_electron_layer = self.two_electron_layer

    def __call__(  # type: ignore[override]
        self, in_1e: Array, in_2e: Optional[Array] = None
    ) -> Tuple[Array, Optional[Array]]:
        """Apply the one-electron layer and optionally the two-electron layer.

        Args:
            in_1e (Array): array of shape (..., n_total, d_1e)
            in_2e (Array, optional): array of shape (..., n_total, n_total, d_2e).
                Defaults to None.

        Returns:
            (Array, optional Array): tuple of (out_1e, out_2e) where out_1e
            is the output from the one-electron layer and out_2e is the output of the
            two-electron stream
        """
        out_1e = self._one_electron_layer(in_1e, in_2e)

        out_2e = in_2e
        if self.two_electron_layer is not None and in_2e is not None:
            out_2e = self._two_electron_layer(in_2e)

        return out_1e, out_2e


class FermiNetBackflow(Module):
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
    """

    residual_blocks: Sequence[
        Callable[
            [Array, Optional[Array]],
            Tuple[Array, Optional[Array]],
        ]
    ]

    def setup(self):
        """Setup called residual blocks."""
        self._residual_block_list = [block for block in self.residual_blocks]

    def __call__(  # type: ignore[override]
        self,
        stream_1e: Array,
        stream_2e: Optional[Array] = None,
    ) -> Array:
        """Iteratively apply residual blocks to Ferminet input streams.

        Args:
            stream_1e (Array): one-electron input stream of shape
                (..., nelec, d1).
            stream_2e (Array, optional): two-electron input of shape
                (..., nelec, nelec, d2).

        Returns:
            (Array): the output of the one-electron stream after applying
            self.residual_blocks to the initial input streams.
        """
        for block in self._residual_block_list:
            stream_1e, stream_2e = block(stream_1e, stream_2e)

        return stream_1e


class SplitDense(Module):
    """Split input on the 2nd-to-last axis and apply unique Dense layers to each split.

    Attributes:
        split (ParticleSplit): number of pieces to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `split` = 2, then the input is split (5, 5).
            If nelec = 10, and `split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `split` is a sequence, there will be one more
            split than the length of the sequence. In the original use-case of spin-1/2
            particles, `split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        ndense_per_split (Sequence[int]): sequence of integers specifying the number of
            dense nodes in the unique dense layer applied to each split of the input.
            This determines the output shapes for each split, i.e. the outputs are
            shaped (..., split_size[i], ndense[i])
        kernel_initializer (WeightInitializer): kernel initializer. Has signature
            (key, shape, dtype) -> Array
        bias_initializer (WeightInitializer): bias initializer. Has signature
            (key, shape, dtype) -> Array. Defaults to random normal
            initialization.
        use_bias (bool, optional): whether to add a bias term. Defaults to True.
    """

    split: ParticleSplit
    ndense_per_split: Sequence[int]
    kernel_initializer: WeightInitializer
    bias_initializer: WeightInitializer = get_bias_initializer("normal")
    use_bias: bool = True

    def setup(self):
        """Set up the dense layers for each split."""
        nsplits = get_nsplits(self.split)

        if len(self.ndense_per_split) != nsplits:
            raise ValueError(
                "Incorrect number of dense output shapes specified for number of "
                "splits, should be one shape per split: shapes {} specified for the "
                "given split {}".format(self.ndense_per_split, self.split)
            )

        self._dense_layers = [
            Dense(
                self.ndense_per_split[i],
                kernel_init=self.kernel_initializer,
                bias_init=self.bias_initializer,
                use_bias=self.use_bias,
            )
            for i in range(nsplits)
        ]

    def __call__(self, x: Array) -> ArrayList:  # type: ignore[override]
        """Split the input and apply a dense layer to each split.

        Args:
            x (Array): array of shape (..., n, d)

        Returns:
            [(..., n[i], self.ndense_per_split[i])]: list of length nsplits, where
            nsplits is the number of splits created by
            split(x, self.split, axis=-2), and the ith entry of the output is the
            ith split transformed by a dense layer with self.ndense_per_split[i] nodes.
        """
        x_split = split(x, self.split, axis=-2)
        return [self._dense_layers[i](split) for i, split in enumerate(x_split)]


def _compute_exponential_envelopes_on_leaf(
    r_ei_leaf: Array,
    norbitals: int,
    kernel_initializer_dim: WeightInitializer,
    kernel_initializer_ion: WeightInitializer,
    isotropic: bool = False,
    envelope_softening: chex.Scalar = 0.0,
) -> Array:
    """Calculate exponential envelopes for orbitals of a single split."""
    if isotropic:
        scale_out = _isotropy_on_leaf(
            r_ei_leaf,
            norbitals,
            kernel_initializer_dim,
        )
    else:
        scale_out = _anisotropy_on_leaf(
            r_ei_leaf,
            norbitals,
            kernel_initializer_dim,
        )
    # scale_out has shape (..., nelec, norbitals, nion, d)
    distances = jnp.linalg.norm(scale_out, axis=-1)
    softened_distances = (
        jnp.sqrt(distances**2 + envelope_softening**2) - envelope_softening
    )
    inv_exp_distances = jnp.exp(-softened_distances)  # (..., nelec, norbitals, nion)

    # Multiply elementwise over final two axes and sum over final axis, returning
    # (..., nelec, norbitals)
    return jnp.sum(
        ElementWiseMultiply(naxes=2, kernel_init=kernel_initializer_ion)(
            inv_exp_distances
        ),
        axis=-1,
    )


def _compute_exponential_envelopes_all_splits(
    r_ei: Array,
    orbitals_split: ParticleSplit,
    norbitals_per_spin: Sequence[int],
    kernel_initializer_dim: WeightInitializer,
    kernel_initializer_ion: WeightInitializer,
    isotropic: bool = False,
    envelope_softening: chex.Scalar = 0.0,
) -> ArrayList:
    """Calculate exponential envelopes for all splits."""
    r_ei_split = split(r_ei, orbitals_split, axis=-3)
    return jax.tree_map(
        functools.partial(
            _compute_exponential_envelopes_on_leaf,
            kernel_initializer_dim=kernel_initializer_dim,
            kernel_initializer_ion=kernel_initializer_ion,
            isotropic=isotropic,
            envelope_softening=envelope_softening,
        ),
        r_ei_split,
        list(norbitals_per_spin),
    )


class FermiNetOrbitalLayer(Module):
    """Make the FermiNet orbitals (parallel linear layers with exp decay envelopes).

    Attributes:
        orbitals_split (ParticleSplit): number of pieces to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `orbitals_split` = 2, then the input is split
            (5, 5). If nelec = 10, and `orbitals_split` = (2, 4), then the input is
            split into (2, 4, 4) -- note when `orbitals_split` is a sequence, there will
            be one more split than the length of the sequence. In the original use-case
            of spin-1/2 particles, `split` should be either the number 2 (for
            closed-shell systems) or should be a Sequence with length 1 whose element is
            less than the total number of electrons.
        norbitals_per_split (Sequence[int]): sequence of integers specifying the number
            of orbitals to create for each split. This determines the output shapes for
            each split, i.e. the outputs are shaped (..., split_size[i], norbitals[i])
        kernel_initializer_linear (WeightInitializer): kernel initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> Array
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer for the
            decay rate in the exponential envelopes. If `isotropic_decay` is True, then
            this initializes a single decay rate number per ion and orbital. If
            `isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> Array
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer for the
            linear combination over the ions of exponential envelopes. Has signature
            (key, shape, dtype) -> Array
        bias_initializer_linear (WeightInitializer): bias initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> Array
        envelope_softening (float): amount by which to soften the cusp of the
            exponential envelope. If set to c, then an ei distance of r is replaced by
            sqrt(r^2 + c^2) - c. Defaults to 0.
        use_bias (bool, optional): whether to add a bias term to the linear part of the
            orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    orbitals_split: ParticleSplit
    norbitals_per_split: Sequence[int]
    kernel_initializer_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_linear: WeightInitializer
    envelope_softening: chex.Scalar = 0.0
    use_bias: bool = True
    isotropic_decay: bool = False

    def setup(self):
        """Setup envelope kernel initializers."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._kernel_initializer_envelope_dim = self.kernel_initializer_envelope_dim
        self._kernel_initializer_envelope_ion = self.kernel_initializer_envelope_ion

    @flax.linen.compact
    def __call__(  # type: ignore[override]
        self, x: Array, r_ei: Optional[Array] = None
    ) -> ArrayList:
        """Apply a dense layer R -> R^n for each split and multiply by exp envelopes.

        Args:
            x (Array): array of shape (..., nelec, d)
            r_ei (Array): array of shape (..., nelec, nion, d)

        Returns:
            [(..., nelec[i], self.norbitals_per_split[i])]: list of FermiNet orbital
            matrices computed from an output stream x and the electron-ion displacements
            r_ei. Here n[i] is the number of particles in the ith split. The exponential
            envelopes are computed only when r_ei is not None (so, when connected to
            FermiNetBackflow, when ion locations are specified). To output square
            matrices, say for composing with the determinant anti-symmetry,
            nelec[i] should be equal to self.norbitals_per_split[i].
        """
        orbs = SplitDense(
            self.orbitals_split,
            self.norbitals_per_split,
            self.kernel_initializer_linear,
            self.bias_initializer_linear,
            use_bias=self.use_bias,
        )(x)
        if r_ei is not None:
            exp_envelopes = _compute_exponential_envelopes_all_splits(
                r_ei,
                self.orbitals_split,
                self.norbitals_per_split,
                self._kernel_initializer_envelope_dim,
                self._kernel_initializer_envelope_ion,
                self.isotropic_decay,
                self.envelope_softening,
            )
            orbs = tree_prod(orbs, exp_envelopes)
        return orbs


class DoublyEquivariantOrbitalLayer(Module):
    """Equivariantly generate an orbital matrix corresponding to each input stream.

    The calculation being done here is a bit subtle, so it's worth explaining here
    in some detail. Let the equivariant input vectors to this layer be y_i. Then, this
    layer will generate an orbital matrix M_p for each particle P, such that the
    (i,j)th element of M_p satisfies M_(p,i,j) = phi_j(y_p, y_i). This is essentially
    the usual orbital matrix formula M_(i,j) = phi_j(y_i), except with an added
    dependence on the particle index p which allows us to generate a distinct matrix
    for each input particle. This construction allows us to generate a unique
    antisymmetric determinant D_p = det(M_p) for each input particle, which can then
    be the basis for an expressive antiequivariant layer.

    If r_ei is provided in addition to the main inputs y_i, then an exponentially
    decaying envelope is also applied equally to every orbital matrix M_p in order to
    ensure that the orbital values decay to zero far from the ions.

    Attributes:
        orbitals_split (ParticleSplit): number of pieces to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `orbitals_split` = 2, then the input is split
            (5, 5). If nelec = 10, and `orbitals_split` = (2, 4), then the input is
            split into (2, 4, 4) -- note when `orbitals_split` is a sequence, there will
            be one more split than the length of the sequence. In the original use-case
            of spin-1/2 particles, `split` should be either the number 2 (for
            closed-shell systems) or should be a Sequence with length 1 whose element is
            less than the total number of electrons.
        norbitals_per_split (Sequence[int]): sequence of integers specifying the number
            of orbitals to create for each split. This determines the output shapes for
            each split, i.e. the outputs are shaped (..., split_size[i], norbitals[i])
        kernel_initializer_linear (WeightInitializer): kernel initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> Array
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer for the
            decay rate in the exponential envelopes. If `isotropic_decay` is True, then
            this initializes a single decay rate number per ion and orbital. If
            `isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> Array
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer for the
            linear combination over the ions of exponential envelopes. Has signature
            (key, shape, dtype) -> Array
        bias_initializer_linear (WeightInitializer): bias initializer for the linear
            part of the orbitals. Has signature (key, shape, dtype) -> Array
        use_bias (bool, optional): whether to add a bias term to the linear part of the
            orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should be
            anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    orbitals_split: ParticleSplit
    norbitals_per_split: Sequence[int]
    kernel_initializer_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_linear: WeightInitializer
    use_bias: bool = True
    isotropic_decay: bool = False

    def setup(self):
        """Setup envelope kernel initializers."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._kernel_initializer_envelope_dim = self.kernel_initializer_envelope_dim
        self._kernel_initializer_envelope_ion = self.kernel_initializer_envelope_ion

    def _get_orbital_matrices_one_split(self, x: Array, norbitals: int) -> Array:
        """Get the equivariant orbital matrices for a single split.

        Args:
            x (Array): input array of shape (..., nelec[i], d).
            norbitals (int): number of orbitals to generate. For square matrices,
                norbitals should equal nelec[i]

        Returns:
            (Array): the equivariant orbitals for this split block, as an array
                of shape (..., nelec[i], nelec[i], norbitals). Both the -2 and -3 axes
                are equivariant with respect to the input particles.
        """
        batch_dims = x.shape[:-2]
        nelec = x.shape[-2]
        d = x.shape[-1]
        dense_input_piece_shape = (*batch_dims, nelec, nelec, d)

        # Since the goal is to calculate M_(p,i,j) = phi_j(y_p, y_i), we need to create
        # the right combinations (y_p, y_i) before we can apply a dense layer to
        # generate the orbitals. The below lines build up these combinations so that,
        # for example, if x is [[1, 2],[3,4]], then dense_inputs will be equal to
        # [[[1,2,1,2],[3,4,1,2]], [[1,2,3,4],[3,4,3,4]].
        axis_2_repeated_inputs = jnp.reshape(
            jnp.repeat(x, nelec, axis=-2), dense_input_piece_shape
        )
        axis_3_repeated_inputs = jnp.reshape(
            # Expand dim of x here to handle case where there are no batch dimensions
            jnp.repeat(jnp.expand_dims(x, -3), nelec, axis=-3),
            dense_input_piece_shape,
        )
        dense_inputs = jnp.concatenate(
            [axis_2_repeated_inputs, axis_3_repeated_inputs], axis=-1
        )

        return Dense(
            norbitals,
            self.kernel_initializer_linear,
            self.bias_initializer_linear,
            use_bias=self.use_bias,
        )(dense_inputs)

    @flax.linen.compact
    def __call__(  # type: ignore[override]
        self, x: Array, r_ei: Optional[Array] = None
    ) -> ArrayList:
        """Calculate an equivariant orbital matrix for each input particle.

        Args:
            x (Array): array of shape (..., nelec, d)
            r_ei (Array): array of shape (..., nelec, nion, d)

        Returns:
            (ArrayList): list of length nsplits of arrays of shape
            (..., nelec[i], nelec[i], self.norbitals_per_split[i]). Here nelec[i] is the
            number of particles in the ith split. The output arrays have both their -2
            and -3 axes equivariant with respect to the input particles. The exponential
            envelopes are computed only when r_ei is not None (so, when connected to
            FermiNetBackflow, when ion locations are specified). To output square
            matrices, say in order to be able to take antiequivariant per-particle
            determinants, nelec[i] should be equal to self.norbitals_per_split[i].
        """
        # split_x is a list of nsplits arrays of shape (..., nelec[i], d)]
        split_x = split(x, self.orbitals_split, -2)
        # orbs is a list of nsplits arrays of shape
        # (..., nelec[i], nelec[i], norbitals[i])
        orbs = [
            self._get_orbital_matrices_one_split(x, self.norbitals_per_split[i])
            for (i, x) in enumerate(split_x)
        ]

        if r_ei is not None:
            exp_envelopes = _compute_exponential_envelopes_all_splits(
                r_ei,
                self.orbitals_split,
                self.norbitals_per_split,
                self._kernel_initializer_envelope_dim,
                self._kernel_initializer_envelope_ion,
                self.isotropic_decay,
            )
            # Envelope must be expanded to apply equally to each per-particle matrix.
            exp_envelopes = jax.tree_map(
                lambda x: jnp.expand_dims(x, axis=-3), exp_envelopes
            )
            orbs = tree_prod(orbs, exp_envelopes)
        return orbs
