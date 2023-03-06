from jax import lax
from jax._src.lax import linalg as lax_linalg
from jax._src import dtypes

from jax.numpy import lax_numpy as jnp

_T = lambda x: jnp.swapaxes(x, -1, -2)
_H = lambda x: jnp.conjugate(jnp.swapaxes(x, -1, -2))

# Stolen from jax.numpy.linalg
def _promote_arg_dtypes(*args):
    """Promotes `args` to a common inexact type."""

    def _to_inexact_type(type):
        return type if jnp.issubdtype(type, jnp.inexact) else jnp.float_

    inexact_types = [_to_inexact_type(jnp._dtype(arg)) for arg in args]
    dtype = dtypes.canonicalize_dtype(jnp.result_type(*inexact_types))
    args = [lax.convert_element_type(arg, dtype) for arg in args]
    if len(args) == 1:
        return args[0]
    else:
        return args


def cofactor_solve(a, b):
    """Equivalent to det(a)*solve(a, b) for nonsingular mat.

    Intermediate function used for jvp and vjp of det.
    This function borrows heavily from jax.numpy.linalg.solve and
    jax.numpy.linalg.slogdet to compute the gradient of the determinant
    in a way that is well defined even for low rank matrices.

    This function handles two different cases:
    * rank(a) == n or n-1
    * rank(a) < n-1

    For rank n-1 matrices, the gradient of the determinant is a rank 1 matrix.
    Rather than computing det(a)*solve(a, b), which would return NaN, we work
    directly with the LU decomposition. If a = p @ l @ u, then
    det(a)*solve(a, b) =
    prod(diag(u)) * u^-1 @ l^-1 @ p^-1 b =
    prod(diag(u)) * triangular_solve(u, solve(p @ l, b))
    If a is rank n-1, then the lower right corner of u will be zero and the
    triangular_solve will fail.
    Let x = solve(p @ l, b) and y = det(a)*solve(a, b).
    Then y_{n}
    x_{n} / u_{nn} * prod_{i=1...n}(u_{ii}) =
    x_{n} * prod_{i=1...n-1}(u_{ii})
    So by replacing the lower-right corner of u with prod_{i=1...n-1}(u_{ii})^-1
    we can avoid the triangular_solve failing.
    To correctly compute the rest of y_{i} for i != n, we simply multiply
    x_{i} by det(a) for all i != n, which will be zero if rank(a) = n-1.

    For the second case, a check is done on the matrix to see if `solve`
    returns NaN or Inf, and gives a matrix of zeros as a result, as the
    gradient of the determinant of a matrix with rank less than n-1 is 0.
    This will still return the correct value for rank n-1 matrices, as the check
    is applied *after* the lower right corner of u has been updated.

    Args:
      a: A square matrix or batch of matrices, possibly singular.
      b: A matrix, or batch of matrices of the same dimension as a.

    Returns:
      det(a) and cofactor(a)^T*b, aka adjugate(a)*b
    """
    a = _promote_arg_dtypes(jnp.asarray(a))
    b = _promote_arg_dtypes(jnp.asarray(b))
    a_shape = jnp.shape(a)
    b_shape = jnp.shape(b)
    a_ndims = len(a_shape)
    if not (
        a_ndims >= 2 and a_shape[-1] == a_shape[-2] and b_shape[-2:] == a_shape[-2:]
    ):
        msg = (
            "The arguments to _cofactor_solve must have shapes "
            "a=[..., m, m] and b=[..., m, m]; got a={} and b={}"
        )
        raise ValueError(msg.format(a_shape, b_shape))
    if a_shape[-1] == 1:
        return a[..., 0, 0], b
    # lu contains u in the upper triangular matrix and l in the strict lower
    # triangular matrix.
    # The diagonal of l is set to ones without loss of generality.
    lu, pivots, permutation = lax_linalg.lu(a)
    dtype = lax.dtype(a)
    batch_dims = lax.broadcast_shapes(lu.shape[:-2], b.shape[:-2])
    x = jnp.broadcast_to(b, batch_dims + b.shape[-2:])
    lu = jnp.broadcast_to(lu, batch_dims + lu.shape[-2:])
    # Compute (partial) determinant, ignoring last diagonal of LU
    diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
    parity = jnp.count_nonzero(pivots != jnp.arange(a_shape[-1]), axis=-1)
    sign = jnp.asarray(-2 * (parity % 2) + 1, dtype=dtype)
    # partial_det[:, -1] contains the full determinant and
    # partial_det[:, -2] contains det(u) / u_{nn}.
    partial_det = jnp.cumprod(diag, axis=-1) * sign[..., None]
    lu = lu.at[..., -1, -1].set(1.0 / partial_det[..., -2])
    permutation = jnp.broadcast_to(permutation, batch_dims + (a_shape[-1],))
    iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims + (1,)))
    # filter out any matrices that are not full rank
    d = jnp.ones(x.shape[:-1], x.dtype)
    d = lax_linalg.triangular_solve(lu, d, left_side=True, lower=False)
    d = jnp.any(jnp.logical_or(jnp.isnan(d), jnp.isinf(d)), axis=-1)
    d = jnp.tile(d[..., None, None], d.ndim * (1,) + x.shape[-2:])
    x = jnp.where(d, jnp.zeros_like(x), x)  # first filter
    x = x[iotas[:-1] + (permutation, slice(None))]
    x = lax_linalg.triangular_solve(
        lu, x, left_side=True, lower=True, unit_diagonal=True
    )
    x = jnp.concatenate(
        (x[..., :-1, :] * partial_det[..., -1, None, None], x[..., -1:, :]), axis=-2
    )
    x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)
    x = jnp.where(d, jnp.zeros_like(x), x)  # second filter

    return partial_det[..., -1], x
