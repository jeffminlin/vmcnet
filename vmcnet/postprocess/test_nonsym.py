import jax.numpy as jnp
import jax.random as rnd
from vmcnet.utils.distribute import pmap
from vmcnet import models
from numpy.testing import assert_allclose
from jax.config import config


def direct_AS(f,n1,n2,full=False):
    print('This test is agnostic to the structure of psi and it is very slow.')

    if full:
        n=n1+n2
        def Af(params,X):
            pl = models.antisymmetry.ParallelPermutations(n)
            (PX, signs), _ = pl.init_with_output(rnd.PRNGKey(0), X)
            AfX=0
            outblock=f(params,PX)
            AfX+=jnp.inner(signs,outblock)
            return AfX
    else:
        def Af(params,X):
            pl1 = models.antisymmetry.ParallelPermutations(n1)
            (PX1, signs1), _ = pl1.init_with_output(rnd.PRNGKey(0), X[:,0:n1,:])
            pl2 = models.antisymmetry.ParallelPermutations(n2)
            (PX2, signs2), _ = pl2.init_with_output(rnd.PRNGKey(0), X[:,n1:n1+n2,:])
            AfX=0
            for i,s in enumerate(signs1):
                print('slow nonsym-antisym test {:,d}/{:,d}'.format((i+1)*signs2.size,signs1.size*signs2.size))
                PX1i=jnp.repeat(PX1[:,i:i+1,:,:],PX2.shape[1],axis=1)
                block=jnp.concatenate([PX1i,PX2],axis=-2)
                signblock=s*signs2
                outblock=f(params,block)
                AfX+=jnp.inner(signblock,outblock)
            return AfX
    return Af

def assert_f_Af(f,Af,params,X,n1,n2,full=False):
    Af_=direct_AS(f,n1,n2,full)
    assert_allclose(Af(params,X),Af_(params,X),rtol=1/1000)
    print('Test passed: Correct nonsym-antisym relation.')

def assert_f_Af_slog(slogf,slogAf,*args,**kw):
    assert_f_Af(noslog(slogf),noslog(slogAf),*args,**kw)

def noslog(slog):
    def f(params,X):
        signs,logs=slog(params,X)
        return signs*jnp.exp(logs)
    return f