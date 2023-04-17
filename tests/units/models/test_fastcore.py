from vmcnet.models.construct import FermiNet
from vmcnet.train import default_config
from vmcnet.models import construct
import jax.numpy as jnp
import jax.random as rnd
import os
import numpy as np
import jax
from functools import partial
import pickle
from vmcnet.utils import io
import sys

# pytest function needs to return None
def test_fastcore():
    runtest()

def runtest(resolution=25):

    config=default_config.get_default_config()
    config.model.type='fastcore'

    modelconfig=config.model
    modelconfig=default_config.choose_model_type_in_model_config(modelconfig)

    ratio=1
    modelconfig.auto.fc_ratio=ratio
    modelconfig.auto.full_det=True

    ion_charges=config.problem.ion_charges
    ion_pos=config.problem.ion_pos
    nelec=config.problem.nelec

    ion_charges=jnp.array(ion_charges)
    ion_pos=jnp.array(ion_pos)
    nelec=jnp.array(nelec)
    model=construct.get_model_from_config(modelconfig,nelec,ion_pos,ion_charges)

    key=rnd.PRNGKey(0)
    k1,k2=rnd.split(key)
    elec_pos=rnd.normal(rnd.PRNGKey(1),(nelec[0]+nelec[1],3))
    N=elec_pos.shape[0]

    R=1.5*np.max(ion_pos)
    print(R)

    eps=R/resolution
    Y=jnp.arange(-R/2,R/2,eps)
    Z=jnp.arange(-R,R,eps)
    Y,Z=jnp.meshgrid(Y,Z)
    a,b=Y.shape

    Y,Z=jnp.ravel(Y),jnp.ravel(Z)
    X=jnp.zeros_like(Y)

    Elec_pos=np.array(elec_pos)[None,:,:]*np.ones_like(Y)[:,None,None]
    Elec_pos[:,0,0]=X
    Elec_pos[:,0,1]=Y
    Elec_pos[:,0,2]=Z
    Elec_pos=jnp.array(Elec_pos)

    params=model.init(rnd.PRNGKey(0),Elec_pos)
    orbitals=model.apply(params,Elec_pos,get_orbitals=True)[0][0]

    orbitals=orbitals.reshape((a,b,4,4))

    dx=jnp.linalg.norm(orbitals[1:]-orbitals[:-1],axis=-1)

    dists=[jnp.linalg.norm(Elec_pos[:,0,:]-pos[None,:],axis=-1) for pos in ion_pos]
    mindist=jnp.minimum(*dists).reshape((a,b))
    core_region=jnp.where(mindist<np.max(ion_pos)*ratio/10)
    I,J=core_region

    assert(jnp.std(dx[I,J,1])/jnp.std(dx)<1/1000)
    return orbitals


if __name__=='__main__':
    orbitals=runtest(50)
    import matplotlib.pyplot as plt

    def PCA(X,k):
        X_=X.reshape((-1,X.shape[-1]))
        _,V=jnp.linalg.eigh(jnp.dot(X_.T,X_))
        return jnp.tensordot(X,V[:,-k:],axes=1)

    fig,axs=plt.subplots(2)
    for i in [0,1]:
        data=PCA(orbitals[:,:,i],3)
        data=data/jnp.std(data)
        axs[i].imshow(jnp.swapaxes(jnp.sin(10*data),0,1))
    plt.show()