import time

import jax

from fast_laplacian import get_laplace_old_and_new

"""
PERFORMANCE NOTES
LAPLACE_INV: 3x faster than laplace_autodiff on highly batched computation
    Batch size 1,000
      n=7 (N): no speed-up
      n=14 (N2): 1.4x speed-up
      n=42 (benzene): 2.8x speed-up
      n=84 (benzene dimer): runs to completion; but old approach fails so can't compare
            5s per epoch -> 
    Batch size 10,000
      n=7 (N): 1.6x speed-up
      n=14 (N2): 1.7x speed-up
      n=42 (benzene): 2.5x speed-up
      n=84 (benzene dimer): didn't try
  
LAPLACE_LINSOLVE: slower or same speed as laplace_autodiff, and uses more memory.
    Batch size 1,000:
      n=7 (N): no speed-up
      n=14 (N2): 10% speed-up
      n=42 (benzene): 25% slow-down
      n=84 (benzene dimer): 40% slow-down
   Batch size 10,000:
      n=7 (N): 15% speed-up
      n=14 (N2): no speed-up
      n=42 (benzene): OOM

MISC: 
    PSIFORMER benzene dimer uses ~1000 determinants per A100 GPU and this takes
        3 seconds per iteration (wall-clock time). 
"""


def run_test(xs, laplace_old, laplace_new, n):
    # Warm-up round to force compilation
    result_old = laplace_old(xs[0])
    result_new = laplace_new(xs[0])
    result_old.block_until_ready()
    result_new.block_until_ready()

    # Test new
    start = time.perf_counter()
    for i in range(1, nsample + 1):
        result_new += laplace_new(xs[i])

    result_new.block_until_ready()
    stop = time.perf_counter()
    time_new = (stop - start) / nsample

    # Test old
    start = time.perf_counter()
    for i in range(1, nsample + 1):
        result_old += laplace_old(xs[i])

    result_old.block_until_ready()
    stop = time.perf_counter()
    time_old = (stop - start) / nsample

    if n > 1:
        print(f"Time for old approach: {time_old}")
        print(f"Time for new approach: {time_new}")

    return result_old, result_new


nsample = 10
nbatch = 10000
n_orbs = [1, 84]
d = 3

if __name__ == "__main__":
    print("Starting with dummy run for n=1.")

    for n_orb in n_orbs:
        if n_orb > 1:
            print(f"Running n={n_orb}")

        n_coord = n_orb * d

        key = jax.random.PRNGKey(1)
        key1, key2, key3, key4 = jax.random.split(key, 4)

        Phi0 = jax.random.normal(key1, (n_orb, n_orb)) / n_orb
        Phi1 = jax.random.normal(key2, (n_orb, n_orb, n_coord)) / n_orb
        Phi2 = jax.random.normal(key3, (n_orb, n_orb, n_coord)) / n_orb
        xs = jax.random.normal(key4, (nsample + 1, nbatch, n_coord))

        laplace_old, laplace_new = get_laplace_old_and_new(
            n_coord, Phi0, Phi1, Phi2, use_linsolve=False
        )
        laplace_old = laplace_new
        result_old, result_new = run_test(xs, laplace_old, laplace_new, n_orb)
