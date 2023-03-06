import jax
import jax.numpy as jnp
from cofactor import cofactor_solve


def get_laplace_old_and_new(n_coord, Phi0, Phi1, Phi2, dtype=jnp.float32):
    def get_phi(x):
        return Phi0 + Phi1 @ x + Phi2 @ x**2

    def log_abs_psi(x):
        # IMPORTANT to use slogdet here. Differentiating det is 3x slower than
        # differentiating slogdet, so using det makes the new method look way too good.
        return jnp.linalg.slogdet(get_phi(x))[1]

    def laplace_autodiff(x):
        result = 0.0
        grad_log_psi = jax.grad(log_abs_psi)
        # NOTE: Perf is the same with linearize and without, but this formulation is a bit
        # more elegant and is what the ferminet repo currently uses.
        grad_log_psi_val, jvp_grad_log_psi = jax.linearize(grad_log_psi, x)

        result += jnp.sum(grad_log_psi_val**2)

        for i in range(n_coord):
            in_tangents = jnp.eye(1, n_coord, i, dtype=dtype)[0]
            out_tangents = jvp_grad_log_psi(in_tangents)
            result += out_tangents[i]

        return result

    def grad_phi(x, tangent_vec):
        A, gAi = jax.jvp(get_phi, (x,), (tangent_vec,))
        return gAi

    def grad_phi_2(x, tangent_vec):
        def grad_phi_i(x):
            return grad_phi(x, tangent_vec)

        gAi, g2Ai = jax.jvp(grad_phi_i, (x,), (tangent_vec,))
        return gAi, g2Ai

    def laplace_det_lemma(x):
        phi = get_phi(x)
        eye = jnp.eye(phi.shape[-1], phi.shape[-1])
        det_phi, phi_adj = cofactor_solve(phi, eye)

        result = dtype(0.0)

        # NOTE: It's possible to batch these computations more aggressively by
        # stacking across all coordinates i before doing the einsums. However, this
        # uses a lot of memory and can cause OOM errors for larger determinants.
        identity_matrix = jnp.eye(n_coord, dtype=dtype)

        def handle_coord(i, result):
            d_phi_dxi, d2_phi_dxi = grad_phi_2(x, identity_matrix[i])

            result += jnp.einsum("ij,ji->", d2_phi_dxi, phi_adj)

            B = d_phi_dxi @ phi_adj
            # Clever way to calculate positive diagonal terms of the 2x2 determinants
            result += jnp.trace(B) ** 2 - jnp.sum(jnp.diag(B) ** 2)
            # Negative diagonal terms of the 2x2 determinants
            result -= 2 * jnp.einsum("ij,ji->", jax.numpy.triu(B, k=1), B)
            return result

        result = jax.lax.fori_loop(0, n_coord, handle_coord, result)

        return result / det_phi

    # NOTE: jax.jit is necessary, vmap does NOT do it automatically.
    laplace_old = jax.vmap(jax.jit(laplace_autodiff))
    laplace_new = jax.vmap(jax.jit(laplace_det_lemma))

    return laplace_old, laplace_new
