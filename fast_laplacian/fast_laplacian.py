import jax
import jax.scipy as scipy
import numpy as np
import jax.numpy as jnp
import time

# TODO: test against log domain version to make sure it's not faster.
# TODO: investigate numerical stability of laplace_inv to make sure it's not an issue.
# TODO: implement in FermiNet, either as custom local energy or preferably as a custom
# determinant function that has this behavior when differentiated TWICE (could be tricky).
# TODO: test in FermiNet context.
# TODO: debug why first time makes things slow even after the attempted throw-away run

# PERFORMANCE NOTES
# WITHOUT BATCHES:
#   for all versions: consistently >=3 times faster than old version, and it gets
#       better for larger determinants. By n=40 (largest systems studied so far) it's ~7x faster,
#       by n=80 (not yet in reach for existing methods) it's almost 10x faster
#
# WITH BATCHES of size 1000 (more realistic):
#   for laplace_linsolve the speed-up is ~3x for n=10 but goes down to almost nothing by n=40
#   for laplace_inv the speed-up is ~3x for n=10 and goes UP to almost ~10x for n=20, n=40


def getA(x):
    return B @ x + C @ x**2


def getDetA(x):
    return jnp.linalg.det(getA(x))


def laplace_old(x, n):
    result = 0.0
    gradDetA = jax.grad(getDetA)

    for i in range(n):
        result += jax.jvp(gradDetA, (x,), (np.eye(1, n, i)[0],))[1][i]

    return result


def gradA(x, i, n):
    A, gAi = jax.jvp(getA, (x,), (np.eye(1, n, i)[0],))
    return gAi


def grad2A(x, i, n):
    def gradA_i(x):
        return gradA(x, i, n)

    gAi, g2Ai = jax.jvp(gradA_i, (x,), (np.eye(1, n, i)[0],))
    return gAi, g2Ai


def laplace_linsolve(x, n):
    A = getA(x)
    detA = getDetA(x)

    dATs = []
    d2ATs = []
    for i in range(n):
        dAi, d2Ai = grad2A(x, i, n)
        dATs.append(dAi.T)
        d2ATs.append(d2Ai.T)

    dAT_stack = jnp.stack(dATs, axis=-1)
    # TODO: this is wasteful, should be able to speed-up the d2A part
    d2AT_stack = jnp.stack(d2ATs, axis=-1)
    alldAT = jnp.concatenate([dAT_stack, d2AT_stack], axis=-1)

    o_shape = alldAT.shape
    flat_shape = (o_shape[0], o_shape[1] * o_shape[2])
    dA_reshaped = jnp.reshape(alldAT, flat_shape)
    matProds_flat = scipy.linalg.solve(A.T, dA_reshaped)
    matProds = jnp.reshape(matProds_flat, o_shape)
    dA_prods, dA2_prods = jnp.split(matProds, (n,), axis=-1)

    result = jnp.einsum("iid->", dA2_prods)

    # Move batch axis to front (denoted d in einsum)
    B = jnp.swapaxes(dA_prods, 0, -1)
    result += jnp.sum(jnp.trace(B, axis1=1, axis2=2) ** 2)
    result -= jnp.sum(jnp.einsum("dii->di", B) ** 2)
    result -= 2 * jnp.einsum("dij,dji->", jax.numpy.triu(B, k=1), B)

    return result * detA


def laplace_inv(x, n):
    A = getA(x)
    Ainv = jnp.linalg.inv(A)
    detA = getDetA(x)

    dAs = []
    d2As = []
    for i in range(n):
        dAi, d2Ai = grad2A(x, i, n)
        dAs.append(dAi)
        d2As.append(d2Ai)

    d2As = jnp.stack(d2As)
    # Rows of d2A multiplied by just the matching columns of Ainv
    result = jnp.einsum("dij,ji->", d2As, Ainv)

    dAs = jnp.stack(dAs)
    B = dAs @ Ainv
    # Clever way to calculate positive diagonal terms of the 2x2 determinants
    result += jnp.sum(jnp.trace(B, axis1=1, axis2=2) ** 2)
    result -= jnp.sum(jnp.einsum("dii->di", B) ** 2)
    # Negative diagonal terms of the 2x2 determinants
    result -= 2 * jnp.einsum("dij,dji->", jax.numpy.triu(B, k=1), B)

    return result * detA


#
# def laplace_mix(x, n):
#     A = getA(x)
#     Ainv = jnp.linalg.inv(A)
#     detA = getDetA(x)
#
#     dAs = []
#     d2As = []
#     for i in range(n):
#         dAi, d2Ai = grad2A(x, i, n)
#         dAs.append(dAi)
#         d2As.append(d2Ai)
#
#     d2As = jnp.stack(d2As, axis=0)
#     result = jnp.einsum("dij,ji->", d2As, Ainv)
#
#     dAT_stack = jnp.swapaxes(jnp.stack(dAs, axis=-1), 0, 1)
#
#     o_shape = dAT_stack.shape
#     flat_shape = (o_shape[0], o_shape[1] * o_shape[2])
#     dA_reshaped = jnp.reshape(dAT_stack, flat_shape)
#     matProds_reshaped = scipy.linalg.solve(A.T, dA_reshaped)
#     dA_prods = jnp.reshape(matProds_reshaped, o_shape)
#
#     B = jnp.swapaxes(dA_prods, 0, -1)
#     result += jnp.sum(jnp.trace(B, axis1=1, axis2=2) ** 2)
#     result -= jnp.sum(jnp.einsum("dii->di", B) ** 2)
#     result -= 2 * jnp.einsum("dij,dji->", jax.numpy.triu(B, k=1), B)
#
#     return result * detA

# I think jit is not necessary but leaving it in just in case
laplace_old = jax.vmap(jax.jit(laplace_old, static_argnums=1), in_axes=(0, None))
laplace_new = jax.vmap(jax.jit(laplace_inv, static_argnums=1), in_axes=(0, None))


nsample = 10
nbatch = 1000
ns = [1, 5, 10, 20, 40]

old_times = []
new_times = []

for n in ns:
    print(f"Running n={n}")
    B = np.random.normal(0, 1.0 / n, (n, n, n))
    C = np.random.normal(0, 1.0 / n, (n, n, n))
    x = np.linspace(1, n, n)

    xs = np.random.normal(0, 1.0 / n, (nsample + 1, nbatch, n))

    # Warm-up round to force compilation
    result_old = laplace_old(xs[0], n)
    result_new = laplace_new(xs[0], n)
    result_old.block_until_ready()
    result_new.block_until_ready()

    # Test old
    start = time.perf_counter()
    for i in range(1, nsample + 1):
        result_old += laplace_old(xs[i], n)

    result_old.block_until_ready()
    stop = time.perf_counter()

    time_old = (stop - start) / nsample

    # Test new
    start = time.perf_counter()
    for i in range(1, nsample + 1):
        result_new += laplace_new(xs[i], n)

    result_new.block_until_ready()
    stop = time.perf_counter()
    time_new = (stop - start) / nsample

    print(f"Time for old approach: {time_old}")
    print(f"Time for new approach: {time_new}")
    old_times.append(time_old)
    new_times.append(time_new)

    if np.any(np.abs((result_new - result_old) / result_new) > 1e-2):
        print(f"WARNING: max relative err between old, new greater than 1e-2")

print(old_times)
print(new_times)
