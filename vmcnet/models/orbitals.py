from .core import Module
import jax.numpy as jnp
import flax


class core_orbital_fns(Module):
    """A module representing a core orbital function.

    Attributes:
        n (int): Number of core orbitals. Will be stacked in the last dimension
        orbitaltype (str): The type of orbital function to use.

    Methods:
        __call__(X): Computes the value of the core orbital function at the given points.

    """

    n: int
    orbitaltype: str

    @flax.linen.compact
    def __call__(self, X):
        """Computes the value of the core orbital function at the given points.

        Args:
            X (jax.numpy.ndarray): The input array of shape (batch_size, ndim).

        Returns:
            The output array of shape (batch_size,) containing the values of the core orbital function
            evaluated at the input points.

        """
        # temporary simple core orbitals
        if self.orbitaltype == "exp":
            c = self.param("c", flax.linen.initializers.uniform(1.0), ())
            # soften = self.param("soften", flax.linen.initializers.uniform(1.0), ())
            # purely exponential function
            soften = 0.0
            r = jnp.sqrt(jnp.abs(soften) + jnp.sum(X**2, axis=-1))
            return jnp.exp(-jnp.abs(c) * r)[...,None]

        # gaussian function
        if self.orbitaltype == "sto-3g_H":
            # params from: https://www.basissetexchange.org/basis/sto-3g/format/nwchem/?version=1&elements=1
            gHSexp1 = 3.425250914  # gaussian H S orbital sto-3G exponent #1
            gHSexp2 = 0.6239137298
            gHSexp3 = 0.1688554040

            gHScoeff1 = 0.1543289673  # gaussian H orbital sto-3G coefficient #1
            gHScoeff2 = 0.5353281423
            gHScoeff3 = 0.4446345422

            # soften = self.param("soften", flax.linen.initializers.uniform(1.0), ())
            soften = 0.0
            r = jnp.sqrt(jnp.abs(soften) + jnp.sum(X**2, axis=-1))

            g1 = jnp.power(2 * gHSexp1 / jnp.pi, 0.75) * jnp.exp(-gHSexp1 * r**2)
            g2 = jnp.power(2 * gHSexp2 / jnp.pi, 0.75) * jnp.exp(-gHSexp2 * r**2)
            g3 = jnp.power(2 * gHSexp3 / jnp.pi, 0.75) * jnp.exp(-gHSexp3 * r**2)

            orb = gHScoeff1 * g1 + gHScoeff2 * g2 + gHScoeff3 * g3
            return orb[...,None]

        if self.orbitaltype == "sto-3g_Li":
            exp_1s=jnp.array([
                            0.1611957475E+02,
                            0.2936200663E+01,
                            0.7946504870E+00
                        ])
            coeff_1s=jnp.array([
                            0.1543289673E+00,
                            0.5353281423E+00,
                            0.4446345422E+00
                        ])
            
            exp_sp=jnp.array([
                            0.6362897469E+00,
                            0.1478600533E+00,
                            0.4808867840E-01
                        ])
            coeff_2s=jnp.array([
                            -0.9996722919E-01,
                            0.3995128261E+00,
                            0.7001154689E+00
                        ])
            coeff_2p=jnp.array([
                            0.1559162750E+00,
                            0.6076837186E+00,
                            0.3919573931E+00
                        ])
            
            r2=jnp.sum(X**2,axis=-1)

            def g(a):
                return a**(1/4)*jnp.exp(-a*r2[...,None])
            
            

            _1s=jnp.inner(jnp.power(2 * exp_1s / jnp.pi, 0.75) * jnp.exp(-exp_1s * r2[...,None]), coeff_1s)
            _2s=jnp.inner(jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp*r2[...,None]), coeff_2s)

            _2px = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,0, None], coeff_2p)
            _2py = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,1, None], coeff_2p)
            _2pz = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,2, None], coeff_2p)

            orbs=[_1s, _2s, _2px, _2py, _2pz]
            orbs=jnp.stack(orbs,axis=-1)
            return orbs




#
#
#class core_orbital_fns(Module):
#    """A module representing a core orbital function.
#
#    Attributes:
#        n (int): Number of core orbitals. Will be stacked in the last dimension
#        orbitaltype (str): The type of orbital function to use.
#
#    Methods:
#        __call__(X): Computes the value of the core orbital function at the given points.
#
#    """
#
#    n: int
#    orbitaltype: str
#
#    @flax.linen.compact
#    def __call__(self, X):
#        """Computes the value of the core orbital function at the given points.
#
#        Args:
#            X (jax.numpy.ndarray): The input array of shape (batch_size, ndim).
#
#        Returns:
#            The output array of shape (batch_size,) containing the values of the core orbital function
#            evaluated at the input points.
#
#        """
#
#        if 'sto-3g' in self.orbitaltype:
#            a,c=self.get_gaussian_params()
#            r2 = jnp.sum(X**2, axis=-1)[...,None,None]
#            gs = jnp.power(2 * a / jnp.pi, 0.75) * jnp.exp(-a * r2)
#            s_orbitals=jnp.sum(gs*c,axis=-1)
#
#        # gaussian function
#        if self.orbitaltype == "sto-3g_H":
#            return s_orbitals
#
#        if self.orbitaltype == "sto-3g_Li":
#            #exp_1s=jnp.array([
#            #                0.1611957475E+02,
#            #                0.2936200663E+01,
#            #                0.7946504870E+00
#            #            ])
#            #coeff_1s=jnp.array([
#            #                0.1543289673E+00,
#            #                0.5353281423E+00,
#            #                0.4446345422E+00
#            #            ])
#            
#            exp_sp=jnp.array([
#                            0.6362897469E+00,
#                            0.1478600533E+00,
#                            0.4808867840E-01
#                        ])
#            #coeff_2s=jnp.array([
#            #                -0.9996722919E-01,
#            #                0.3995128261E+00,
#            #                0.7001154689E+00
#            #            ])
#            coeff_2p=jnp.array([
#                            0.1559162750E+00,
#                            0.6076837186E+00,
#                            0.3919573931E+00
#                        ])
#            
#            #r2=jnp.sum(X**2,axis=-1)
#
#            #_1s=jnp.inner(jnp.power(2 * exp_1s / jnp.pi, 0.75) * jnp.exp(-exp_1s * r2[...,None]), coeff_1s)
#            #_2s=jnp.inner(jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp*r2[...,None]), coeff_2s)
#
#
#
#            _2px = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,0, None], coeff_2p)
#            _2py = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,1, None], coeff_2p)
#            _2pz = jnp.inner( jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp * r2[...,None]) * jnp.sqrt(4*exp_sp) * X[...,2, None], coeff_2p)
#            p_orbitals=jnp.stack([_2px,_2py,_2pz],axis=-1)
#
#            return jnp.concatenate([s_orbitals,p_orbitals],axis=-1)
#
#            #orbs=[_1s, _2s, _2px, _2py, _2pz]
#            #orbs=jnp.stack(orbs,axis=-1)
#            #return orbs
#        
#    def get_gaussian_params(self):
#        if self.orbitaltype == "sto-3g_H":
#            # params from: https://www.basissetexchange.org/basis/sto-3g/format/nwchem/?version=1&elements=1
#            # gaussian H S orbital sto-3G exponent #1
#            gHSexps = [[
#                3.425250914,
#                0.6239137298,
#                0.1688554040
#            ]]
#            # gaussian H orbital sto-3G coefficient #1
#            gHScoeffs = [[
#                0.1543289673,
#                0.5353281423,
#                0.4446345422
#            ]]
#            return jnp.array(gHSexps), jnp.array(gHScoeffs)
#        
#        if self.orbitaltype == "sto-3g_Li":
#            exp_1s=[
#                    0.1611957475E+02,
#                    0.2936200663E+01,
#                    0.7946504870E+00
#                ]
#            coeff_1s=[
#                    0.1543289673E+00,
#                    0.5353281423E+00,
#                    0.4446345422E+00
#                ]
#            
#            exp_sp=[
#                    0.6362897469E+00,
#                    0.1478600533E+00,
#                    0.4808867840E-01
#                ]
#            coeff_2s=[
#                    -0.9996722919E-01,
#                    0.3995128261E+00,
#                    0.7001154689E+00
#                ]
#            coeff_2p=[
#                    0.1559162750E+00,
#                    0.6076837186E+00,
#                    0.3919573931E+00
#                ]
#            #_1s=jnp.inner(jnp.power(2 * exp_1s / jnp.pi, 0.75) * jnp.exp(-exp_1s * r2[...,None]), coeff_1s)
#            #_2s=jnp.inner(jnp.power(2 * exp_sp / jnp.pi, 0.75) * jnp.exp(-exp_sp*r2[...,None]), coeff_2s)
#
#            exps=jnp.array([exp_1s,exp_sp])
#            coeffs=jnp.array([coeff_1s,coeff_2s])
#
#            return exps,coeffs 
#
#        raise NotImplementedError
#        
#    @staticmethod
#    def get_overlap_matrix(
#        orbitals1,
#        orbitals2,
#        shift
#        ):
#        A1,C1=orbitals1.get_gaussian_params()
#        A2,C2=orbitals2.get_gaussian_params()
#
#        return analytic.overlap_of_Gaussians(A1[:,None],A2[None,:],shift[None,None])*C1[:,None]*C2[None,:]
#
#    @staticmethod
#    def get_Laplacian_matrix(
#        orbitals1,
#        orbitals2,
#        shift
#        ):
#        A1,C1=orbitals1.get_gaussian_params()
#        A2,C2=orbitals2.get_gaussian_params()
#
#        return analytic.Laplacian_of_Gaussians(A1[:,None],A2[None,:])*C1[:,None]*C2[None,:]
#    
#    @staticmethod
#    def get_ee_matrix(
#        orbitals1,
#        orbitals2,
#        shift
#        ):
#        A1,C1=orbitals1.get_gaussian_params()
#        A2,C2=orbitals2.get_gaussian_params()
#
#        return overlap_of_Gaussians(A1[:,None],A2[None,:])*C1[:,None]*C2[None,:]
#    
#    @staticmethod
#    def get_ei_matrix(
#        orbitals1,
#        orbitals2,
#        p1,
#        p2,
#        ion
#        ):
#        A1,C1=orbitals1.get_gaussian_params()
#        A2,C2=orbitals2.get_gaussian_params()
#
#        return overlap_of_Gaussians(A1[:,None],A2[None,:])*C1[:,None]*C2[None,:]
#