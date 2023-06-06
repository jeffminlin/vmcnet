import pyscf
import pickle

###################################################################################################
mol=pyscf.gto.M(atom='Li 0 0 0; H 3.014 0 0', basis='sto-3g',unit='Bohr')
test=pyscf.scf.UHF(mol)
test.kernel()
coeff=test.mo_coeff

ints=dict(
    ints1e=mol.get_hcore(),
    ints2e=mol.intor('int2e', aosym='s1'),
    overlap=mol.intor('int1e_ovlp', aosym='s1'),
    C_a=coeff[0,:,:2],
    C_b=coeff[1,:,:2],
)

with open('ints.pickle','wb') as f:
    pickle.dump(ints,f)
