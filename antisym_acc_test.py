"""Test model antisymmetries."""
import time
import itertools
import math

import jax.numpy as jnp
import jax
import numpy as np

n = (7,7)
dy = 256
ndense = 64
activation = jnp.tanh
key = jax.random.PRNGKey(0)


def get_alternating_signs(n: int):
    """Return alternating series of 1 and -1, of length n."""
    return jax.ops.index_update(jnp.ones(n), jax.ops.index[1::2], -1.0)


def _get_lexicographic_signs(n: int):
    """Get the signs of the n! permutations of (1,...,n) in lexicographic order."""
    signs = jnp.ones(1)

    for i in range(2, n + 1):
        alternating_signs = get_alternating_signs(i)
        signs = jnp.concatenate([sign * signs for sign in alternating_signs])

    return signs


perms = jnp.array(list(itertools.permutations(range(n[0])))), jnp.array(
    list(itertools.permutations(range(n[1])))
)
signs = _get_lexicographic_signs(n[0]), _get_lexicographic_signs(n[1])

weights1 = jax.random.normal(key, (n[0], dy, ndense)), jax.random.normal(
    key, (n[1], dy, ndense)
)
weights2 = jax.random.normal(key, (ndense,))

# x should be (n, dy)
@jax.jit
def simple_ffnn(x):
    dense0 = jnp.einsum("...ij,ijk->...k", x[0], weights1[0])
    dense1 = jnp.einsum("...ij,ijk->...k", x[1], weights1[1])
    dense = dense0 + dense1
    dense = activation(dense)
    return dense @ weights2


# x should be (n, dy)
# TODO: add in signs
@jax.jit
def bruteforce_antisym(x):
    x_perms = jnp.take(x[0], perms[0], axis=0), jnp.take(x[1], perms[1], axis=0)
    x_perms = jnp.expand_dims(x_perms[0], 1), jnp.expand_dims(x_perms[1], 0)
    desired_shape0 = math.factorial(n[0]), math.factorial(n[1]), n[0], dy
    desired_shape1 = math.factorial(n[0]), math.factorial(n[1]), n[1], dy
    x_perms = jnp.broadcast_to(x_perms[0], desired_shape0), jnp.broadcast_to(
        x_perms[1], desired_shape1
    )
    perms_out = simple_ffnn(x_perms)
    return jnp.sum(perms_out)


# x should be (n, dy)
# TODO: add in signs
@jax.jit
def fast_antisym(x):
    # x is (n,dy)
    # weights are [n, dy, ndense)
    # Now we have (n,n,ndense)
    matmul_matrix = jnp.einsum("ij,kjl", x[0], weights1[0]), jnp.einsum(
        "ij,kjl", x[1], weights1[1]
    )
    matmul_matrix = (
        matmul_matrix[0][perms[0], jnp.arange(n[0]), :],
        matmul_matrix[1][perms[1], jnp.arange(n[1]), :],
    )
    dense = jnp.sum(matmul_matrix[0], axis=1), jnp.sum(matmul_matrix[1], axis=1)

    desired_shape = math.factorial(n[0]), math.factorial(n[1]), ndense
    dense = jnp.broadcast_to(
        jnp.expand_dims(dense[0], 1), desired_shape
    ), jnp.broadcast_to(jnp.expand_dims(dense[1], 0), desired_shape)

    dense = dense[0] + dense[1]
    dense = activation(dense)
    out = dense @ weights2

    return jnp.sum(out, axis=(0, 1))


inputs = jax.random.normal(key, (n[0], dy)), jax.random.normal(key, (n[1], dy))

start = time.time()
out = simple_ffnn(inputs)
print(f"FFNN time: {time.time() - start}, out {out}")


start = time.time()
out = bruteforce_antisym(inputs)
print(f"Brute antisym time: {time.time() - start}, out {out}")


start = time.time()
out = fast_antisym(inputs)
print(f"Fast antisym time: {time.time() - start}, out {out}")
