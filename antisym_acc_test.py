"""Test model antisymmetries."""
import time
import itertools

import jax.numpy as jnp
import jax
import numpy as np

n = 2
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

perms = jnp.array(list(itertools.permutations(range(n))))
signs = _get_lexicographic_signs(n)

weights1 = jax.random.normal(key, (n, dy, ndense))
weights2 = jax.random.normal(key, (ndense,))

# x should be (n, dy)
@jax.jit
def simple_ffnn(x):
    dense = jnp.einsum('...ij,ijk->...k', x, weights1)
    dense = activation(dense)
    return dense @ weights2


# x should be (n, dy)
@jax.jit
def bruteforce_antisym(x):
    x_perms = jnp.take(x, perms, axis=0)
    perms_out = simple_ffnn(x_perms)
    signed_perms_out = signs * perms_out
    return jnp.sum(signed_perms_out)

# x should be (n, dy)
@jax.jit
def fast_antisym(x):
    # x is (n,dy)
    # weights are [n, dy, ndense)
    # Now we have (n,n,ndense)
    pairwise_dense = jnp.einsum('ij,kjl', x, weights1)
    pairwise_per_perm = pairwise_dense[perms, jnp.arange(n), :]
    dense_per_perm = jnp.sum(pairwise_per_perm, axis=1)
    dense_per_perm = activation(dense_per_perm)
    outs_per_perm = dense_per_perm @ weights2 * signs
    return jnp.sum(outs_per_perm)
    # return contributions

inputs = jax.random.normal(key, (n, dy))

start = time.time()
out = simple_ffnn(inputs)
print(f"FFNN time: {time.time() - start}, out {out}")


start = time.time()
out = bruteforce_antisym(inputs)
print(f"Brute antisym time: {time.time() - start}, out {out}")


start = time.time()
out = fast_antisym(inputs)
print(f"Fast antisym time: {time.time() - start}, out {out}")