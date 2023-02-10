import jax
import jax.scipy as scipy
import numpy as np
import jax.numpy as jnp
import time

# TODO: implement in terms of log det, log Psi
# NOTE: currently for n=10 it's 2x as fast but for n=20 its same speed :(


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


def laplace_new(x, n):
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
    matProds_reshaped = scipy.linalg.solve(A.T, dA_reshaped)
    matProds = jnp.reshape(matProds_reshaped, o_shape)

    dA_prods, dA2_prods = jnp.split(matProds, (n,), axis=-1)

    result = 0.0

    for d in range(n):
        result += jnp.trace(dA2_prods[:, :, d])

    for d in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                result += 2 * jnp.linalg.det(dA_prods[:, :, d][[i, j], :][:, [i, j]])

    return result * detA


laplace_old = jax.jit(laplace_old, static_argnums=1)
laplace_new = jax.jit(laplace_new, static_argnums=1)


nsample = 1000
ns = [5, 10, 15, 20, 25]

old_times = []
new_times = []

for n in ns:
    print(f"Running n={n}")
    B = np.random.normal(0, 1.0 / n, (n, n, n))
    C = np.random.normal(0, 1.0 / n, (n, n, n))
    x = np.linspace(1, n, n)

    xs = np.random.normal(0, 1.0 / n, (nsample + 1, n))

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

    if np.abs((result_new - result_old) / result_new) > 1e-3:
        print(
            f"WARNING: relative err between old, new greater than 1e-3: ({result_old}, {result_new})"
        )

print(old_times)
print(new_times)
