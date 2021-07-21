"""Antiequivariant parts to compose into a model."""
from typing import Tuple

import flax
import jax
import jax.numpy as jnp

from vmcnet.utils.slog_helpers import array_list_to_slog, array_to_slog, slog_multiply
from vmcnet.utils.pytree_helpers import tree_prod
from vmcnet.utils.typing import ArrayList, SLArray, SLArrayList, SpinSplit
from .core import get_alternating_signs, get_nelec_per_spin, is_tuple_of_arrays
from .equivariance import DoublyEquivariantOrbitalLayer, FermiNetOrbitalLayer
from .weights import WeightInitializer


def get_submatrices_along_first_col(x: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """Get the submatrices of x by deleting row i and col 0, for all rows of x.

    Args:
        x (jnp.ndarray): a tensor of orbital matrices which is square in the last two
            dimensions, thus of shape (..., n, n). The second last dimension is the
            particle dimension, and the last is the orbital dimension.

    Returns:
        (int, jnp.ndarray): n, submatrices of shape (..., n, n-1, n-1), obtained by
        deleting row (..., i, :) and deleted column is (..., :, 0), for 0 <= i <= n - 1.
    """
    if len(x.shape) < 2 or x.shape[-1] != x.shape[-2]:
        msg = "Calculating cofactors requires shape (..., n, n), got {}"
        raise ValueError(msg.format(x.shape))

    n = x.shape[-1]
    # Calculate minor_(0,i) by deleting the first orbital and ith particle indices
    submats = [jnp.delete(jnp.delete(x, i, axis=-2), 0, axis=-1) for i in range(n)]

    # Stack on axis -3 to ensure shape (..., n) once det removes the last two axes
    stacked_submats = jnp.stack(submats, axis=-3)
    return n, stacked_submats


def cofactor_antieq(x: jnp.ndarray) -> jnp.ndarray:
    """Compute a cofactor-based antiequivariance.

    Input must be square in the last two dimensions, of shape (..., n, n). The second
    last dimension is assumed to be the particle dimension, and the last is assumed
    to be the orbital dimension. The output will be of shape (..., n), preserving the
    particle dimension but getting rid of the orbital dimension.

    The transformation applies to each square matrix separately. Given an nxn matrix M,
    with cofactor matrix C, the output for that matrix will be a single vector of length
    n, whose ith component is M_(i,0)*C_(i,0)*(-1)**i. This is a single term in the
    cofactor expansion of the determinant of M. Thus, the sum of the returned vector
    values will always be equal to the determinant of M.

    This function implements an antiequivariant transformation, meaning that permuting
    two particle indices in the input will result in the output having 1) the same two
    particle indices permuted, and 2) ALL values multiplied by -1.

    Args:
        x (jnp.ndarray): a tensor of orbital matrices which is square in the last two
            dimensions, thus of shape (..., n, n). The second last dimension is the
            particle dimension, and the last is the orbital dimension.

    Returns:
        (jnp.ndarray): array of shape (..., n), with the ith output (along the last
        axis) given by the ith term in the cofactor expansion of det(x) along the first
        entry of the last axis
    """
    first_orbital_vals = x[..., 0]
    n, stacked_submatrices = get_submatrices_along_first_col(x)
    cofactors = get_alternating_signs(n) * jnp.linalg.det(stacked_submatrices)
    return first_orbital_vals * cofactors


def slog_cofactor_antieq(x: jnp.ndarray) -> SLArray:
    """Compute a cofactor-based antiequivariance, returning results in slogabs form.

    See :func:`~vmcnet.models.antiequivariance.cofactor_antieq`. This function performs
    the same operations, but gives a result in the (sign, log) domain, going through
    a jnp.linalg.slogdet call instead of a jnp.linalg.det call.

    Args:
        x (jnp.ndarray): a tensor of orbital matrices which is square in the last two
            dimensions, thus of shape (..., n, n). The second last dimension is the
            particle dimension, and the last is the orbital dimension.

    Returns:
        (jnp.ndarray, jnp.ndarray): tuple of arrays, each of shape (..., n). The first
        is sign(result), and the second is log(abs(result)).
    """
    # Calculate x_(i, 0) by selecting orbital index 0
    first_orbital_vals = x[..., 0]
    orbital_signs, orbital_logs = array_to_slog(first_orbital_vals)

    n, stacked_submatrices = get_submatrices_along_first_col(x)

    # TODO(ggoldsh): find a faster way to calculate these overlapping determinants.
    (cofactor_signs, cofactor_logs) = jnp.linalg.slogdet(stacked_submatrices)

    signs_and_logs = (
        orbital_signs * cofactor_signs * get_alternating_signs(n),
        orbital_logs + cofactor_logs,
    )
    return signs_and_logs


def _multiply_eq_inputs_by_split_antieq(
    eq_inputs: jnp.ndarray,
    split_antieq: ArrayList,
    spin_split: SpinSplit,
) -> ArrayList:
    """Multiply equivariant input array with a spin-split antiequivariance.

    Args:
        eq_inputs (jnp.ndarray): array of shape (..., nelec, d)
        split_antieq (ArrayList): list of arrays containing nspins arrays of shape
            broadcastable to (..., nelec[i])
        spin_split (SpinSplit): the spin split.

    Returns:
        (ArrayList): list of per-spin arrays of shape (..., nelec[i], d) which
        represent the product of the equivariant inputs with the antiequivariance.
    """
    # Expand antiequivariance to shape (..., nelec[i], 1) to broadcast with inputs
    split_antieq = jax.tree_map(lambda c: jnp.expand_dims(c, -1), split_antieq)
    split_inputs = jnp.split(eq_inputs, spin_split, axis=-2)
    return tree_prod(split_inputs, split_antieq)


def _multiply_eq_inputs_by_split_slog_antieq(
    eq_inputs: jnp.ndarray,
    split_slog_antieq: SLArrayList,
    spin_split: SpinSplit,
) -> SLArrayList:
    """Multiply equivariant input array with a spin-split slog-form antiequivariance.

    Args:
        eq_inputs (jnp.ndarray): array of shape (..., nelec, d)
        split_slog_antieq (SLArrayList): SLArrayList containing nspins arrays of shape
            broadcastable to (..., nelec[i])
        spin_split (SpinSplit): the spin split.

    Returns:
        (SLArrayList): list of per-spin slog arrays of shape (..., nelec[i], d) which
        represent the product of the equivariant inputs with the antiequivariance.
    """
    # Expand antiequivariance to shape (..., nelec[i], 1)] to broadcast with inputs
    split_slog_antieq = jax.tree_map(
        lambda c: jnp.expand_dims(c, -1), split_slog_antieq
    )
    split_slog_inputs = array_list_to_slog(jnp.split(eq_inputs, spin_split, axis=-2))

    return jax.tree_map(
        slog_multiply,
        split_slog_inputs,
        split_slog_antieq,
        is_leaf=is_tuple_of_arrays,
    )


class OrbitalCofactorAntiequivarianceLayer(flax.linen.Module):
    """Apply a cofactor antiequivariance multiplicatively to equivariant inputs.

    Attributes:
         spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear  part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer
            for the decay rate in the exponential envelopes. If `isotropic_decay` is
            True, then this initializes a single decay rate number per ion and orbital.
            If`isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer
            for the linear combination over the ions of exponential envelopes. Has
            signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term to the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should
            be anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    spin_split: SpinSplit
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False

    @flax.linen.compact
    def __call__(self, eq_inputs: jnp.ndarray, r_ei: jnp.ndarray = None) -> ArrayList:
        """Calculate the orbitals and the cofactor-based antiequivariance.

        For a single spin, if the equivariant inputs are y_i, the orbital matrix is M,
        and the cofactor matrix of the orbital matrix is C, the ith output will
        be equal to y_i * M_(i,0) * C_(i,0) * (-1)**i. For multiple spins, each spin is
        handled separately in this same way.

        Args:
            eq_inputs: (jnp.ndarray): array of shape (..., nelec, d), which should
                contain values that are equivariant with respect to the particle
                positions.
            r_ei (jnp.ndarray, optional): array of shape (..., nelec, nion, d)
                representing electron-ion displacements, which if present will be used
                as an extra input to the orbital layer.

        Returns:
            (ArrayList): per-spin list where each list entry is an array of shape
            (..., nelec, d).
        """
        nelec_total = eq_inputs.shape[-2]
        nelec_per_spin = get_nelec_per_spin(self.spin_split, nelec_total)
        ferminet_orbital_layer = FermiNetOrbitalLayer(
            self.spin_split,
            nelec_per_spin,
            self.kernel_initializer_orbital_linear,
            self.kernel_initializer_envelope_dim,
            self.kernel_initializer_envelope_ion,
            self.bias_initializer_orbital_linear,
            self.orbitals_use_bias,
            self.isotropic_decay,
        )

        # Calculate orbital matrices as list of shape [(..., nelec[i], nelec[i])]
        orbital_matrix_list = ferminet_orbital_layer(eq_inputs, r_ei)
        # Calculate cofactors as list of shape [(..., nelec[i])]
        cofactors = jax.tree_map(cofactor_antieq, orbital_matrix_list)
        return _multiply_eq_inputs_by_split_antieq(
            eq_inputs, cofactors, self.spin_split
        )


class SLogOrbitalCofactorAntiequivarianceLayer(flax.linen.Module):
    """Apply a cofactor antieq. multiplicatively to equivariant inputs with slog out.

    Attributes:
         spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear  part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer
            for the decay rate in the exponential envelopes. If `isotropic_decay` is
            True, then this initializes a single decay rate number per ion and orbital.
            If`isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer
            for the linear combination over the ions of exponential envelopes. Has
            signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term to the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should
            be anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    spin_split: SpinSplit
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False

    @flax.linen.compact
    def __call__(self, eq_inputs: jnp.ndarray, r_ei: jnp.ndarray = None) -> SLArrayList:
        """Calculate the orbitals and the cofactor-based antiequivariance.

        For a single spin, if the equivariant inputs are y_i, the orbital matrix is M,
        and the cofactor matrix of the orbital matrix is C, the ith output will
        be equal to y_i * M_(i,0) * C_(i,0) * (-1)**i. For multiple spins, each spin is
        handled separately in this same way.

        Args:
            eq_inputs: (jnp.ndarray): array of shape (..., nelec, d), which should
                contain values that are equivariant with respect to the particle
                positions.
            r_ei (jnp.ndarray, optional): array of shape (..., nelec, nion, d)
                representing electron-ion displacements, which if present will be used
                as an extra input to the orbital layer.

        Returns:
            (SLArrayList): per-spin list where each list entry is an slog array of
            shape (..., nelec, d).
        """
        nelec_total = eq_inputs.shape[-2]
        nelec_per_spin = get_nelec_per_spin(self.spin_split, nelec_total)
        ferminet_orbital_layer = FermiNetOrbitalLayer(
            self.spin_split,
            nelec_per_spin,
            self.kernel_initializer_orbital_linear,
            self.kernel_initializer_envelope_dim,
            self.kernel_initializer_envelope_ion,
            self.bias_initializer_orbital_linear,
            self.orbitals_use_bias,
            self.isotropic_decay,
        )

        # Calculate orbital matrices as list of shape [(..., nelec[i], nelec[i])]
        orbital_matrix_list = ferminet_orbital_layer(eq_inputs, r_ei)
        # Calculate slog cofactors as list of shape [((..., nelec[i]), (..., nelec[i]))]
        slog_cofactors = jax.tree_map(slog_cofactor_antieq, orbital_matrix_list)
        return _multiply_eq_inputs_by_split_slog_antieq(
            eq_inputs, slog_cofactors, self.spin_split
        )


class PerParticleDeterminantAntiequivarianceLayer(flax.linen.Module):
    """Antieq. layer based on determinants of per-particle orbital matrices, slog out.

    Attributes:
         spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear  part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer
            for the decay rate in the exponential envelopes. If `isotropic_decay` is
            True, then this initializes a single decay rate number per ion and orbital.
            If`isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer
            for the linear combination over the ions of exponential envelopes. Has
            signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term to the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should
            be anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    spin_split: SpinSplit
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False

    @flax.linen.compact
    def __call__(self, eq_inputs: jnp.ndarray, r_ei: jnp.ndarray = None) -> ArrayList:
        """Calculate the per-particle orbitals and the antiequivariant determinants.

        For a single spin, if the equivariant inputs are y_p, and the orbital matrix for
        particle p is M_p, the output at index p will be equal to y_p * det(M_p). For
        multiple spins, each spin is handled separately in this same way.

        Args:
            eq_inputs: (jnp.ndarray): array of shape (..., nelec, d), which should
                contain values that are equivariant with respect to the particle
                positions.
            r_ei (jnp.ndarray, optional): array of shape (..., nelec, nion, d)
                representing electron-ion displacements, which if present will be used
                as an extra input to the orbital layer.

        Returns:
            (ArrayList): per-spin list where each list entry is an array of shape
            (..., nelec, d).
        """
        nelec_total = eq_inputs.shape[-2]
        nelec_per_spin = get_nelec_per_spin(self.spin_split, nelec_total)
        equivariant_orbital_layer = DoublyEquivariantOrbitalLayer(
            self.spin_split,
            nelec_per_spin,
            self.kernel_initializer_orbital_linear,
            self.kernel_initializer_envelope_dim,
            self.kernel_initializer_envelope_ion,
            self.bias_initializer_orbital_linear,
            self.orbitals_use_bias,
            self.isotropic_decay,
        )

        # ArrayList of shape (..., nelec[i], nelec[i], nelec[i])
        orbital_matrix_list = equivariant_orbital_layer(eq_inputs, r_ei)
        # ArrayList of nspins arrays of shape (..., nelec[i])
        dets = jax.tree_map(jnp.linalg.det, orbital_matrix_list)
        return _multiply_eq_inputs_by_split_antieq(eq_inputs, dets, self.spin_split)


class SLogPerParticleDeterminantAntiequivarianceLayer(flax.linen.Module):
    """Antieq. layer based on determinants of per-particle orbital matrices, slog out.

    Attributes:
         spin_split (SpinSplit): number of spins to split the input equally,
            or specified sequence of locations to split along the 2nd-to-last axis.
            E.g., if nelec = 10, and `spin_split` = 2, then the input is split (5, 5).
            If nelec = 10, and `spin_split` = (2, 4), then the input is split into
            (2, 4, 4) -- note when `spin_split` is a sequence, there will be one more
            spin than the length of the sequence. In the original use-case of spin-1/2
            particles, `spin_split` should be either the number 2 (for closed-shell
            systems) or should be a Sequence with length 1 whose element is less than
            the total number of electrons.
        kernel_initializer_orbital_linear (WeightInitializer): kernel initializer for
            the linear  part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_dim (WeightInitializer): kernel initializer
            for the decay rate in the exponential envelopes. If `isotropic_decay` is
            True, then this initializes a single decay rate number per ion and orbital.
            If`isotropic_decay` is False, then this initializes a 3x3 matrix per ion and
            orbital. Has signature (key, shape, dtype) -> jnp.ndarray
        kernel_initializer_envelope_ion (WeightInitializer): kernel initializer
            for the linear combination over the ions of exponential envelopes. Has
            signature (key, shape, dtype) -> jnp.ndarray
        bias_initializer_orbital_linear (WeightInitializer): bias initializer for the
            linear part of the orbitals. Has signature
            (key, shape, dtype) -> jnp.ndarray
        orbitals_use_bias (bool, optional): whether to add a bias term to the linear
            part of the orbitals. Defaults to True.
        isotropic_decay (bool, optional): whether the decay for each ion should
            be anisotropic (w.r.t. the dimensions of the input), giving envelopes of the
            form exp(-||A(r - R)||) for a dxd matrix A or isotropic, giving
            exp(-||a(r - R||)) for a number a.
    """

    spin_split: SpinSplit
    kernel_initializer_orbital_linear: WeightInitializer
    kernel_initializer_envelope_dim: WeightInitializer
    kernel_initializer_envelope_ion: WeightInitializer
    bias_initializer_orbital_linear: WeightInitializer
    orbitals_use_bias: bool = True
    isotropic_decay: bool = False

    @flax.linen.compact
    def __call__(self, eq_inputs: jnp.ndarray, r_ei: jnp.ndarray = None) -> SLArrayList:
        """Calculate the per-particle orbitals and the antiequivariant determinants.

        For a single spin, if the equivariant inputs are y_p, and the orbital matrix for
        particle p is M_p, the output at index p will be equal to y_p * det(M_p). For
        multiple spins, each spin is handled separately in this same way.

        Args:
            eq_inputs: (jnp.ndarray): array of shape (..., nelec, d), which should
                contain values that are equivariant with respect to the particle
                positions.
            r_ei (jnp.ndarray, optional): array of shape (..., nelec, nion, d)
                representing electron-ion displacements, which if present will be used
                as an extra input to the orbital layer.

        Returns:
            (SLArrayList): per-spin list where each list entry is an slog array of
            shape (..., nelec, d).
        """
        nelec_total = eq_inputs.shape[-2]
        nelec_per_spin = get_nelec_per_spin(self.spin_split, nelec_total)
        equivariant_orbital_layer = DoublyEquivariantOrbitalLayer(
            self.spin_split,
            nelec_per_spin,
            self.kernel_initializer_orbital_linear,
            self.kernel_initializer_envelope_dim,
            self.kernel_initializer_envelope_ion,
            self.bias_initializer_orbital_linear,
            self.orbitals_use_bias,
            self.isotropic_decay,
        )

        # ArrayList of shape (..., nelec[i], nelec[i], nelec[i])
        orbital_matrix_list = equivariant_orbital_layer(eq_inputs, r_ei)
        # SLArrayList of nspins slog arrays of shape (..., nelec[i])
        slog_dets = jax.tree_map(jnp.linalg.slogdet, orbital_matrix_list)
        return _multiply_eq_inputs_by_split_slog_antieq(
            eq_inputs, slog_dets, self.spin_split
        )
