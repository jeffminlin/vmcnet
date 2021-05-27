"""Local energy calculations."""
from typing import Callable, TypeVar

import jax
from jax._src.api import grad
import jax.numpy as jnp

P = TypeVar("P")  # represents a pytree or pytree-like object containing model params


def laplacian_psi_over_psi(
    grad_log_psi: Callable[[P, jnp.ndarray], jnp.ndarray], params: P, x: jnp.ndarray
) -> jnp.ndarray:
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
        grad_log_psi (Callable): function which evaluates the derivative of log|psi(x)|,
            i.e. (nabla psi)(x) / psi(x), with respect to x. Has the signature
            (params, x) -> (nabla psi)(x) / psi(x), so the derivative should be over the
            second arg, x, and the output shape should be the same as x
        params (pytree): model parameters, passed as the first arg of grad_log_psi
        x (jnp.ndarray): second input to grad_log_psi

    Returns:
        jnp.ndarray: "local" laplacian calculation, i.e. (nabla^2 psi) / psi
    """
    x_shape = x.shape
    flat_x = jnp.reshape(x, (-1,))
    n = flat_x.shape[0]
    identity_mat = jnp.eye(n)

    def flattened_grad_log_psi_of_flat_x(flat_x_in):
        """Flattened input to flattened output version of grad_log_psi."""
        grad_log_psi_out = grad_log_psi(params, jnp.reshape(flat_x_in, x_shape))
        return jnp.reshape(grad_log_psi_out, (-1,))

    def step_fn(carry, unused):
        del unused
        i = carry[0]
        primals, tangents = jax.jvp(
            flattened_grad_log_psi_of_flat_x, (flat_x,), (identity_mat[i],)
        )
        return (i + 1, carry[1] + primals[i] ** 2 + tangents[i]), None

    out, _ = jax.lax.scan(step_fn, (0, jnp.array([0.0])), xs=None, length=n)
    return out[1]


def harmonic_oscillator_potential(omega: jnp.float32, x: jnp.ndarray) -> jnp.float32:
    """Potential energy for independent harmonic oscillators with spring constant omega.

    This function computes sum_i 0.5 * (omega * x_i)^2. If x has more than one axis,
    these are simply summed over, so this corresponds to an isotropic harmonic
    oscillator potential.

    This function should be vmapped in order to be applied to batches of inputs, as it
    expects the first axis of x to correspond to the particle index.

    Args:
        omega (jnp.float32): spring constant
        x (jnp.ndarray): array of particle positions, where the first axis corresponds
            to particle index

    Returns:
        jnp.float32: potential energy value for this configuration x
    """
    return 0.5 * jnp.sum(jnp.square(omega * x))


def make_harmonic_oscillator_local_energy(
    omega: jnp.float32, log_psi: Callable[[P, jnp.ndarray], jnp.ndarray]
) -> Callable[[P, jnp.ndarray], jnp.ndarray]:
    """Factory to create a local energy fn for the harmonic oscillator log|psi|.

    Args:
        omega (jnp.float32): spring constant for the harmonic oscillator
        log_psi (Callable): function which evaluates log|psi| for a harmonic oscillator
            model wavefunction psi. Has the signature (params, x) -> log|psi(x)|.

    Returns:
        Callable: local energy function with the signature (params, x) -> local energy
        associated to the wavefunction psi
    """
    grad_log_psi = jax.grad(log_psi, argnums=1)

    def local_energy(params, x):
        kinetic = -0.5 * laplacian_psi_over_psi(grad_log_psi, params, x)
        potential = harmonic_oscillator_potential(omega, x)
        return kinetic + potential

    return jax.vmap(local_energy, in_axes=(None, 0), out_axes=0)