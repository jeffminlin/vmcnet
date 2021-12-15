# changelog

## vmcnet v0.1.0
Release accompanying the release of the preprint at https://arxiv.org/abs/2112.03491.

[GitHub commits](https://github.com/jeffminlin/vmcnet/compare/v0.1.0...master)

### Features (not comprehensive)
* `vmcnet.train`
    * core VMC loop at `vmcnet.train.vmc.vmc_loop`
    * runners which interface with `ConfigDict`s and CLI parsers
* `vmcnet.updates`
    * functions to interface with optax optimizers
    * functions to parse KFAC configuration
    * Stochastic Reconfiguration optimizer (highly experimental)
* `vmcnet.physics`
    * utility to initialize a set of electron positions
    * molecular local energy terms
    * gradient of the expected energy
* `vmcnet.models`
    * FermiNet and a few variants
        * standard
        * full determinant (implemented in the FermiNet repository at commit [de5c7cdeb39641e8a41331d4f342b0c9e1af501b](https://github.com/deepmind/ferminet/commit/de5c7cdeb39641e8a41331d4f342b0c9e1af501b))
        * hidden fermion models (EmbeddedParticleFermiNet and ExtendedOrbitalMatrixFermiNet)
    * two classes of antiequivariance models, closely related: cofactor-based and per-particle determinant based (slightly more general)
    * factorized and generic antisymmetric models (see https://arxiv.org/abs/2112.03491)
    * jastrow factors, one-body, two-body, and backflow-based
* `vmcnet.mcmc`
    * basic `PositionAmplitudeData` TypedDict which holds positions and wavefunction amplitudes as well as any move metadata
    * gaussian proposal and symmetric Metropolis acceptance
    * dynamic proposal width `DynamicWidthPositionAmplitudeData` TypedDict
    * multi-chain autocorrelation calculations
* `vmcnet.utils`
    * checkpointing logic
    * distributing helpers
    * I/O helpers
    * stable (in a forward pass) log-linear-exp implementation
    * Sign, log array helpers
    * pytree helpers
* `vmcnet.examples`
    * Harmonic oscillator ground-state model and local energy calculation
    * Hydrogen-like atom ground-state model