"""Antiequivariant parts to compose into a model."""
from typing import Callable, List, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from .core import get_alternating_signs, is_tuple_of_arrays


def slog_cofactor_antieq(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a cofactor-based antiequivariance, returning results in slogabs form.

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
        dimensions, thus of shape (..., n, n). The second last dimension is the particle
        dimension, and the last is the orbital dimension.

    Returns:
        (jnp.ndarray, jnp.ndarray): tuple of arrays, each of shape (..., n). The first
        is sign(result), and the second is log(abs(result)).
    """
    if len(x.shape) < 2 or x.shape[-1] != x.shape[-2]:
        msg = "Calculating cofactors requires shape (..., n, n), got {}"
        raise ValueError(msg.format(x.shape))

    # Calculate M_(0,i) by selecting orbital index 0
    first_orbital_vals = x[..., 0]
    orbital_signs = jnp.sign(first_orbital_vals)
    orbital_logs = jnp.log(jnp.abs(first_orbital_vals))

    n = x.shape[-1]
    # Calculate C_(0,i) by deleting the first orbital and ith particle indices
    cofactor_matrices = [
        jnp.delete(jnp.delete(x, i, axis=-2), 0, axis=-1) for i in range(n)
    ]
    # Stack on axis -3 to ensure shape (..., n) once slogdet removes the last two axes
    stacked_cofactor_matrices = jnp.stack(cofactor_matrices, axis=-3)

    # TODO(ggoldsh): find a faster way to calculate these overlapping determinants.
    (cofactor_signs, cofactor_logs) = jnp.linalg.slogdet(stacked_cofactor_matrices)

    signs_and_logs = (
        orbital_signs * cofactor_signs * get_alternating_signs(n),
        orbital_logs + cofactor_logs,
    )
    return signs_and_logs


class OrbitalCofactorAntiequivarianceLayer(flax.linen.Module):
    """Apply a cofactor antiequivariance multiplicatively to equivariant inputs.

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
        ferminet_orbital_layer (callable): a FerminetOrbitalLayer instance. This will be
        used to generate the orbital matrices to which the cofactor equivariance will
        be applied. The orbital layer must generate square orbitals, i.e. norbitals must
        equal nelec for each spin.
    """

    spin_split: Union[int, Sequence[int]]
    ferminet_orbital_layer: Callable[[jnp.ndarray, jnp.ndarray], List[jnp.ndarray]]

    def setup(self):
        """Setup the orbital layer."""
        # workaround MyPy's typing error for callable attribute, see
        # https://github.com/python/mypy/issues/708
        self._ferminet_orbital_layer = self.ferminet_orbital_layer

    @flax.linen.compact
    def __call__(
        self, eq_inputs: jnp.ndarray, r_ei: jnp.ndarray = None
    ) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Calculate the orbitals and the cofactor-based antiequivariance.

        For a single spin, if the equivariant inputs are y_i, the orbital matrix is M,
        and the cofactor matrix of the orbital matrix is C, the ith output will
        be equal to y_i * M_(i,0) * C_(i,0) * (-1)**i. For multiple spins, each spin is
        handled separately in this same way.

        Results are returned as a tuple of arrays, where the first is the sign of the
        results and the second is the log of the absolute value of the results.

        Args:
            eq_inputs: (jnp.ndarray): array of shape (..., nelec, d), which should
                contain values that are equivariant with respect to the particle
                positions.
            r_ei (jnp.ndarray, optional): array of shape (..., nelec, nion, d)
                representing electron-ion displacements, which if present will be used
                as an extra input to the orbital layer.

        Returns:
            (List[Tuple[jnp.ndarray, jnp.ndarray]]): per-spin list where each list
             element is a tuples of arrays of shape (..., nelec, d), where the first
             array in the tuple is sign(results) and the second is log(abs(results)).
        """
        # Calculate orbital matrices as list of shape [(..., nelec[i], nelec[i])]
        orbital_matrix_list = self._ferminet_orbital_layer(eq_inputs, r_ei)
        # Calculate slog cofactors as list of shape [((..., nelec[i]), (..., nelec[i]))]
        slog_cofactors = jax.tree_map(slog_cofactor_antieq, orbital_matrix_list)
        # Expand arrays to (..., nelec[i], 1)] to allow broadcasting to input shape
        expanded_cofactors = jax.tree_map(
            lambda c: jnp.expand_dims(c, -1), slog_cofactors
        )

        # Calculate signs and logs and split results each to shape [(..., nelec[i], d)]
        sign_inputs = jnp.split(jnp.sign(eq_inputs), self.spin_split, axis=-2)
        log_inputs = jnp.split(jnp.log(jnp.abs(eq_inputs)), self.spin_split, axis=-2)

        # Combine signs and logs into shape [((..., nelec[i], d), (..., nelec[i], d))]
        return jax.tree_multimap(
            lambda si, li, c: (si * c[0], li + c[1]),
            sign_inputs,
            log_inputs,
            expanded_cofactors,
            is_leaf=is_tuple_of_arrays,
        )