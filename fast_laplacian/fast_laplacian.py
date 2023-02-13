import jax
import jax.numpy as jnp
import jax.scipy as scipy


def get_laplace_old_and_new(
    n_coord, Phi0, Phi1, Phi2, dtype=jnp.float32, use_linsolve=False
):
    def get_phi(x):
        return Phi0 + Phi1 @ x + Phi2 @ x**2

    def log_abs_psi(x):
        # IMPORTANT to use slogdet here. Differentiating det is 3x slower than
        # differentiating slogdet, so using det makes the new method look way too good.
        return jnp.linalg.slogdet(get_phi(x))[1]

    def laplace_autodiff(x):
        result = 0.0
        grad_log_psi = jax.grad(log_abs_psi)

        for i in range(n_coord):
            primals, tangents = jax.jvp(
                grad_log_psi, (x,), (jnp.eye(1, n_coord, i, dtype=dtype)[0],)
            )
            result += jnp.square(primals[i]) + tangents[i]

        return result

    def grad_phi(x, tangent_vec):
        A, gAi = jax.jvp(get_phi, (x,), (tangent_vec,))
        return gAi

    def grad_phi_2(x, tangent_vec):
        def grad_phi_i(x):
            return grad_phi(x, tangent_vec)

        gAi, g2Ai = jax.jvp(grad_phi_i, (x,), (tangent_vec,))
        return gAi, g2Ai

    def laplace_inv(x):
        phi = get_phi(x)
        phi_inv = jnp.linalg.inv(phi)

        result = dtype(0.0)

        # NOTE: It's possible to batch these computations more aggressively by
        # stacking across all coordinates i before doing the einsums. However, this
        # uses a lot of memory and can cause OOM errors for larger determinants.

        identity_matrix = jnp.eye(n_coord, dtype=dtype)

        def handle_coord(i, result):
            d_phi_dxi, d2_phi_dxi = grad_phi_2(x, identity_matrix[i])
            # Second derivative contribution
            result += jnp.einsum("ij,ji->", d2_phi_dxi, phi_inv)
            # First derivative contribution
            B = d_phi_dxi @ phi_inv
            # Clever way to calculate positive diagonal terms of the 2x2 determinants
            result += jnp.trace(B) ** 2 - jnp.sum(jnp.diag(B) ** 2)
            # Negative diagonal terms of the 2x2 determinants
            result -= 2 * jnp.einsum("ij,ji->", jax.numpy.triu(B, k=1), B)
            return result

        result = jax.lax.fori_loop(0, n_coord, handle_coord, result)

        return result

    def laplace_linsolve(x):
        phi = get_phi(x)

        d_phis = []
        d2_phis = []
        for i in range(n_coord):
            dd_phi_dxi, d2_phi_dxi = grad_phi_2(x, i)
            d_phis.append(dd_phi_dxi.T)
            d2_phis.append(d2_phi_dxi.T)

        dphiT_stack = jnp.stack(d_phis, axis=-1)
        # TODO: this is wasteful, should be able to speed-up the d2A part
        d2phiT_stack = jnp.stack(d2_phis, axis=-1)
        alldphiT_stack = jnp.concatenate([dphiT_stack, d2phiT_stack], axis=-1)

        o_shape = alldphiT_stack.shape
        flat_shape = (o_shape[0], o_shape[1] * o_shape[2])
        dphiT_reshaped = jnp.reshape(alldphiT_stack, flat_shape)
        matProds_flat = scipy.linalg.solve(phi.T, dphiT_reshaped)
        inv_prods = jnp.reshape(matProds_flat, o_shape)
        dphi_prods, d2_phi_prods = jnp.split(inv_prods, (n_coord,), axis=-1)

        result = jnp.einsum("iid->", d2_phi_prods)

        # Move batch axis to front (denoted d in einsum)
        B = jnp.swapaxes(dphi_prods, 0, -1)
        result += jnp.sum(jnp.trace(B, axis1=1, axis2=2) ** 2)
        result -= jnp.sum(jnp.einsum("dii->di", B) ** 2)
        result -= 2 * jnp.einsum("dij,dji->", jax.numpy.triu(B, k=1), B)

        return result

    # NOTE: jax.jit is necessary, vmap does NOT do it automatically.
    laplace_old = jax.vmap(jax.jit(laplace_autodiff))
    laplace_new = (
        jax.vmap(jax.jit(laplace_linsolve))
        if use_linsolve
        else jax.vmap(jax.jit(laplace_inv))
    )

    return laplace_old, laplace_new
