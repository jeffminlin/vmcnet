"""Kinetic energy terms."""

from typing import Callable

import jax
import jax.numpy as jnp

from vmcnet.utils.typing import Array, P, ModelApply


def laplacian_psi_over_psi(
    grad_log_psi_apply: ModelApply,
    params: P,
    x: Array,
) -> Array:
    """Compute (nabla^2 psi) / psi at x given a function which evaluates psi'(x)/psi.

    The computation is done by computing (forward-mode) derivatives of the gradient to
    get the columns of the Hessian, and accumulating the (i, i)th entries (but this
    implementation is significantly more memory efficient than directly computing the
    Hessian).

    This function uses the identity

        (nabla^2 psi) / psi = (nabla^2 log|psi|) + (nabla log|psi|)^2

    to avoid leaving the log domain during the computation.

    This function should be vmapped in order to be applied to batches of inputs, as it
    completely flattens x in order to take second derivatives w.r.t. each component.

    This is approach is extremely similar to the one in the FermiNet repo
    (in the jax branch, as of this writing -- see
    https://github.com/deepmind/ferminet/blob/aade61b3d30883b3238d6b50c85404d0e8176155/ferminet/hamiltonian.py).

    The main difference is that we are being explicit about the flattening of x within
    the Laplacian calculation, so that it does not have to be handled outside of this
    function (psi is free to take x shapes which are not flat).

    Args:
        grad_log_psi_apply (Callable): function which evaluates the derivative of
            log|psi(x)|, i.e. (nabla psi)(x) / psi(x), with respect to x. Has the
            signature (params, x) -> (nabla psi)(x) / psi(x), so the derivative should
            be over the second arg, x, and the output shape should be the same as x
        params (pytree): model parameters, passed as the first arg of grad_log_psi
        x (Array): second input to grad_log_psi

    Returns:
        Array: "local" laplacian calculation, i.e. (nabla^2 psi) / psi
    """
    x_shape = x.shape
    flat_x = jnp.reshape(x, (-1,))
    n = flat_x.shape[0]
    identity_mat = jnp.eye(n)

    def flattened_grad_log_psi_of_flat_x(flat_x_in):
        """Flattened input to flattened output version of grad_log_psi."""
        grad_log_psi_out = grad_log_psi_apply(params, jnp.reshape(flat_x_in, x_shape))
        return jnp.reshape(grad_log_psi_out, (-1,))

    length = n
    vecs = identity_mat

    def step_fn(carry, unused):
        del unused
        i = carry[0]
        primals, tangents = jax.jvp(
            flattened_grad_log_psi_of_flat_x, (flat_x,), (vecs[i],)
        )
        return (
            i + 1,
            carry[1]
            + jnp.square(jnp.dot(primals, vecs[i]))
            + jnp.dot(tangents, vecs[i]),
        ), None

    out, _ = jax.lax.scan(step_fn, (0, jnp.array(0.0)), xs=None, length=length)
    return out[1]


def create_laplacian_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> -0.5 (nabla^2 psi(x) / psi(x)).

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on only a single configuration so must be externally vmapped
        to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)

    def kinetic_energy_fn(params: P, x: Array) -> Array:
        return -0.5 * laplacian_psi_over_psi(grad_log_psi_apply, params, x)

    return kinetic_energy_fn
