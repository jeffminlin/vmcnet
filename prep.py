import pyscf
import pickle

###################################################################################################
mol=pyscf.gto.M(atom='Li 0 0 0; H 3.014 0 0', basis='sto-3g')

ints=dict(
    ints1=mol.get_hcore(),
    ints2=mol.intor('int2e', aosym='s1'),
)

with open('ints.pickle','wb') as f:
    pickle.dump(ints,f)


#breakpoint()
#def local_energy_fn(params,positions,key):
#    return 0*positions[...,0,0]
###################################################################################################