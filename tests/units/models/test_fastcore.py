from vmcnet.train import default_config
from vmcnet.models import construct
import jax.numpy as jnp
import jax.random as rnd
import numpy as np


# pytest function needs to return None
def test_fastcore():
    runtest()


def runtest(mode="test", resolution=100):
    config = default_config.get_default_config()
    config.model.type = "fastcore"

    modelconfig = config.model
    modelconfig = default_config.choose_model_type_in_model_config(modelconfig)

    ratio = 1
    modelconfig.auto.fc_ratio = ratio
    modelconfig.auto.full_det = True

    ion_charges = config.problem.ion_charges
    ion_pos = config.problem.ion_pos
    nelec = config.problem.nelec

    ion_charges = jnp.array(ion_charges)
    ion_pos = jnp.array(ion_pos)
    nelec = jnp.array(nelec)
    model = construct.get_model_from_config(modelconfig, nelec, ion_pos, ion_charges)

    elec_pos = rnd.normal(rnd.PRNGKey(0), (nelec[0] + nelec[1], 3))

    R = np.max(ion_pos)
    L = 1.5 * R
    eps = 2 * L / resolution

    if mode == "plot":
        Y = jnp.arange(-L / 2, L / 2, eps)
        Z = jnp.arange(-L, L, eps)
        Y, Z = jnp.meshgrid(Y, Z)
        a, b = Y.shape

        Y, Z = jnp.ravel(Y), jnp.ravel(Z)
        X = jnp.zeros_like(Y)

    else:
        Z = jnp.arange(-L, L, eps)
        Y = jnp.zeros_like(Z)
        X = jnp.zeros_like(Z)

    Elec_pos = np.array(elec_pos)[None, :, :] * np.ones_like(Y)[:, None, None]
    Elec_pos[:, 0, 0] = X
    Elec_pos[:, 0, 1] = Y
    Elec_pos[:, 0, 2] = Z
    Elec_pos = jnp.array(Elec_pos)

    """ This line is the slow one, even for a single sample
    (elec_pos instead of Elec_pos) """
    # params=model.init(rnd.PRNGKey(0),elec_pos)
    params = model.init(rnd.PRNGKey(0), Elec_pos)

    orbitals = model.apply(params, Elec_pos, get_orbitals=True)[0][
        0
    ]  # first (typically only) pmap slice, spin up
    change_rate = jnp.sum((orbitals[1:] - orbitals[:-1]) ** 2, axis=-1)

    if mode == "test":
        radius = R * ratio / 20
        dists = [
            jnp.linalg.norm(Elec_pos[:, 0, :] - pos[None, :], axis=-1)
            for pos in ion_pos
        ]
        core_regions = [jnp.where(dist < radius) for dist in dists]
        for core in core_regions:
            if isinstance(core, tuple):
                (core,) = core
            np.testing.assert_allclose(
                jnp.mean(change_rate[core, 1:]) / jnp.mean(change_rate),
                0,
                atol=10 ** (-6),
            )
    else:
        orbitals = orbitals.reshape((a, b, 4, 4))

    return orbitals


if __name__ == "__main__":
    orbitals = runtest(mode="plot", resolution=100)
    import matplotlib.pyplot as plt

    def PCA(X, k):
        X_ = X.reshape((-1, X.shape[-1]))
        _, V = jnp.linalg.eigh(jnp.dot(X_.T, X_))
        return jnp.tensordot(X, V[:, -k:], axes=1)

    fig, axs = plt.subplots(2)
    for i in [0, 1]:
        data = PCA(orbitals[:, :, i], 3)
        data = data / jnp.std(data)
        axs[i].imshow(jnp.swapaxes(jnp.sin(10 * data), 0, 1))
    plt.show()
