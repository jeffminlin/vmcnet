import jax
from jax.config import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from fast_laplacian import get_laplace_old_and_new

"""
LAPLACE_INV
  Usually at least as accurate as the old method
  Suffers when K is high, M is low, but the laplace_autodiff suffers too
"""


def run_test(xs, xs_64, laplace_64, laplace_old, laplace_new):
    mean_local_diff_old = jnp.zeros((nbatch,))
    mean_batch_diff_old = 0.0
    mean_local_diff_new = jnp.zeros((nbatch,))
    mean_batch_diff_new = 0.0

    for i in range(nsample):
        result_64 = laplace_64(xs_64[i])
        assert result_64.dtype == jnp.float64

        result_old = laplace_old(xs[i])
        result_new = laplace_new(xs[i])
        assert result_old.dtype == jnp.float32
        assert result_new.dtype == jnp.float32

        mean_local_diff_old += jnp.abs(result_64 - result_old)
        mean_batch_diff_old += jnp.abs(jnp.mean(result_64) - jnp.mean(result_old))
        mean_local_diff_new += jnp.abs(result_64 - result_new)
        mean_batch_diff_new += jnp.abs(jnp.mean(result_64) - jnp.mean(result_new))
        print(f"{i+1}/{nsample}")

    print(f"Mean local diff old: {jnp.mean(mean_local_diff_old / nsample)}")
    print(f"Mean local diff new: {jnp.mean(mean_local_diff_new / nsample)}")

    print(f"Mean batch diff old: {mean_batch_diff_old / nsample}")
    print(f"Mean batch diff new: {mean_batch_diff_new / nsample}")


if __name__ == "__main__":
    nsample = 10
    nbatch = 1000
    n_orb = 14
    d = 3
    n_coord = n_orb * 3

    for K in [1, 1e3, 1e6]:
        for M in [1]:
            for seed in [0]:
                print(f"Testing with condition number {K}, multiplier {M}, seed {seed}")

                key = jax.random.PRNGKey(seed)
                key1, key2, key3, key4, key5 = jax.random.split(key, 5)

                U = jax.random.normal(key1, (n_orb, n_orb), dtype=jnp.float32)
                U, _ = jnp.linalg.qr(U)
                S = jnp.diag(jnp.linspace(1, K, n_orb, dtype=jnp.float32))
                V = jax.random.normal(key2, (n_orb, n_orb), dtype=jnp.float32)
                V, _ = jnp.linalg.qr(V)

                Phi0 = U @ S @ V * M / n_coord
                Phi1 = (
                    jax.random.normal(key3, (n_orb, n_orb, n_coord), dtype=jnp.float32)
                    / n_coord
                )
                Phi2 = (
                    jax.random.normal(key4, (n_orb, n_orb, n_coord), dtype=jnp.float32)
                    / n_coord
                )
                xs = jax.random.normal(
                    key5, (nsample, nbatch, n_coord), dtype=jnp.float32
                )

                Phi0_64 = jnp.array(Phi0, dtype=jnp.float64)
                Phi1_64 = jnp.array(Phi1, dtype=jnp.float64)
                Phi2_64 = jnp.array(Phi2, dtype=jnp.float64)
                xs_64 = jnp.array(xs, dtype=jnp.float64)

                laplace_64, _ = get_laplace_old_and_new(
                    n_coord, Phi0_64, Phi1_64, Phi2_64, dtype=jnp.float64
                )

                laplace_old, laplace_new = get_laplace_old_and_new(
                    n_coord, Phi0, Phi1, Phi2, dtype=jnp.float32
                )

                run_test(xs, xs_64, laplace_64, laplace_old, laplace_new)
