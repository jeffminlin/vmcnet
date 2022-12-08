import jax
import jax.numpy as jnp
import functools
import flax
from vmcnet.utils.typing import Array, SLArray, PyTree
from vmcnet.models.core import is_tuple_of_arrays
from vmcnet.models.construct import FermiNet, DeterminantFnMode, array_to_slog, slog_sum_over_axis


# compare with models.antisymmetry

def slogdiagprod_product(xs: PyTree) -> SLArray:
    slogdiagprods=jax.tree_map(slogdiagprod, xs)
    leaves, _ = jax.tree_flatten(slogdiagprods, is_tuple_of_arrays)
    sign_prod, log_prod = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), leaves
    )
    return sign_prod, log_prod

def slogdiagprod(M: Array):
    diag=jnp.diagonal(M,axis1=-2,axis2=-1)
    return jnp.product(jnp.sign(diag),axis=-1), jnp.sum(jnp.log(jnp.abs(diag)),axis=-1)

def diagprod(M: Array):
    diag=jnp.diagonal(M,axis1=-2,axis2=-1)
    return jnp.product(diag,axis=-1)
    


# same definition as vmcnet.models.construct.FermiNet
# except det->diagprod

class FermiNet_Nonsym(FermiNet):
    @flax.linen.compact
    def __call__(self, elec_pos: Array) -> SLArray:  # type: ignore[override]
        """Compose FermiNet backflow -> orbitals -> logabs determinant product.

        Args:
            elec_pos (Array): array of particle positions (..., nelec, d)

        Returns:
            Array: FermiNet output; logarithm of the absolute value of a
            anti-symmetric function of elec_pos, where the anti-symmetry is with respect
            to the second-to-last axis of elec_pos. The anti-symmetry holds for
            particles within the same split, but not for permutations which swap
            particles across different spin splits. If the inputs have shape
            (batch_dims, nelec, d), then the output has shape (batch_dims,).
        """
        elec_pos, orbitals_split = self._get_elec_pos_and_orbitals_split(elec_pos)

        input_stream_1e, input_stream_2e, r_ei, _ = self._compute_input_streams(
            elec_pos
        )
        stream_1e = self._backflow(input_stream_1e, input_stream_2e)

        norbitals_per_split = self._get_norbitals_per_split(elec_pos, orbitals_split)
        # orbitals is [norb_splits: (ndeterminants, ..., nelec[i], norbitals[i])]
        orbitals = self._eval_orbitals(
            orbitals_split,
            norbitals_per_split,
            input_stream_1e,
            input_stream_2e,
            stream_1e,
            r_ei,
        )

        if self.full_det:
            orbitals = [jnp.concatenate(orbitals, axis=-2)]

        if self._symmetrized_det_fn is not None:
            raise ValueError

            # dets is ArrayList of shape [norb_splits: (ndeterminants, ...)]
            prods = jax.tree_map(diagprod, orbitals)
            # Move axis to get shape [norb_splits: (..., ndeterminants)]
            fn_inputs = jax.tree_map(lambda x: jnp.moveaxis(x, 0, -1), prods)
            if self.determinant_fn_mode == DeterminantFnMode.SIGN_COVARIANCE:
                psi = jnp.squeeze(self._symmetrized_det_fn(fn_inputs), -1)
            elif self.determinant_fn_mode == DeterminantFnMode.PARALLEL_EVEN:
                psi = self._calculate_psi_parallel_even(fn_inputs)
            elif self.determinant_fn_mode == DeterminantFnMode.PAIRWISE_EVEN:
                psi = self._calculate_psi_pairwise_even(fn_inputs)
            else:
                raise self._get_bad_determinant_fn_mode_error()
            return array_to_slog(psi)

        # slog_det_prods is SLArray of shape (ndeterminants, ...)
        slog_det_prods = slogdiagprod_product(orbitals)
        return slog_sum_over_axis(slog_det_prods)

