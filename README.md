# vmcnet
Flexible, general-purpose VMC framework, built on [JAX](https://github.com/google/jax).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

Python 3.9 is required, and a virtual environment is recommended. After cloning, use `pip install -e .` to install the package in editable/develop mode, or `pip install -e .[testing]` if planning on running tests.

If running on a GPU, CUDA needs to be set up properly to work with JAX, and you will need to install the correct `jaxlib` wheel. See, e.g., https://github.com/google/jax#installation.

## Philosophy and usage

This repository is built to serve two purposes:
1. Provide a general python API for variational Monte Carlo calculations compatible with JAX, with a number of built-in neural network architectures for ready-use. 
2. Provide a command-line interface exposing a large number of options for more streamlined (but somewhat less custom) experimentation with architecture/optimization/sampling hyperparameters.

This repository was built as a JAX port of an internal TensorFlow project started in 2019 which itself was initially inspired by the work of [David Pfau, James S. Spencer, Alexander G. D. G. Matthews, and W. M. C. Foulkes](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429). Their repository (and its own JAX branch) can be found [here](https://github.com/deepmind/ferminet).

### Python API

The primary routine exposed by this repository which implements the VMC training loop is the [`train.vmc.vmc_loop`](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/vmc.py#L13) function. This function implements a very generic unsupervised training loop. A skeleton of a script which performs varational Monte Carlo would look something like:

```python
import jax

import vmcnet.mcmc as mcmc
import vmcnet.train as train

# Training hyperparameters
nchains = ...
nburn = ...
nepochs = ...

seed = ...
logdir = ...
checkpoint_every = ...
checkpoint_dir = ...

# Initialize parameters, data, and optimization state
params = ...
data = ...
optimizer_state = ...

# Walker function (get new data)
def walker_fn(params, data, key):
    ...
    return accept_ratio, data, key

# Define how the parameters are updated
def update_param_fn(params, data, optimizer_state, key):
    ...
    return params, optimizer_state, metrics, key

# (Optionally) burn samples
def burning_step(params, data, key):
    ...
    return data, key

key = jax.random.PRNGKey(seed)
data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)

# Train!
params, optimizer_state, data, key, _ = train.vmc.vmc_loop(
    params,
    optimizer_state,
    data,
    nchains,
    nepochs,
    walker_fn,
    update_param_fn,
    key,
    False,
    logdir,
    checkpoint_every=checkpoint_every,
    checkpoint_dir=checkpoint_dir,
)
```
Note the required function signatures. A simple but complete working example can be found in the [hydrogen-like atom](https://github.com/jeffminlin/vmcnet/blob/master/tests/integrations/examples/test_hydrogen_like_atom.py) example in the test suite.

### Command-line

Alternatively, a command-line interface has been implemented which provides more streamlined access to subsets of the repository via setting [ConfigDict](https://github.com/google/ml_collections) objects.

There are two scripts which have been exposed thus far: `vmc-molecule` and `vmc-statistics`.

The primary command `vmc-molecule` calls `train.runners.run_molecule`. See [`train.default_config.get_default_config()`](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/default_config.py#L60) to explore the options which have been exposed to the command-line. To edit these options at the command-line, use the "`--config.`" prefix. For example,
```sh
vmc-molecule \
    --config.problem.ion_pos="((0.0, 0.0, -2.0), (0.0, 0.0, 2.0))" \
    --config.problem.ion_charges="(7.0, 7.0)" \
    --config.problem.nelec="(7, 7)" \
    --config.model.ferminet.full_det="True" \
    --config.logging_level="INFO"
```
will train the full single-determinant FermiNet on the nitrogen molecule at dissociating bond length 4.0 for 2e5 epochs on 2000 walkers (which are distributed across any available GPUs, if supported by the installation).

You can also reload and evaluate or continue training from previous checkpoints via the "`--reload.`" prefix. The options can be seen in [`train.default_config.get_default_reload_config()`](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/default_config.py#L47). The reloading will only occur if `--reload.logdir` is set.

The `vmc-statistics` command calls `train.runners.vmc_statistics`. This simple script is designed to be compatible with the output of an evaluation run with `vmc-molecule`, but can accept any path to a file which contains local energies (a file with nchains x nepochs energies). It computes and saves a json file containing the average energy, the sample variance, the estimated integrated autocorrelation, and the estimated standard error. The options can be viewed simply via `vmc-statistics -h`.

## Contributing

See [how to contribute](CONTRIBUTING.md).

## Cite

A preprint of the paper which originally introduced this repository can be found at https://arxiv.org/abs/2112.03491, which can be cited via:
```
@misc{lin2021explicitly,
      title={Explicitly antisymmetrized neural network layers for variational Monte Carlo simulation}, 
      author={Jeffmin Lin and Gil Goldshlager and Lin Lin},
      year={2021},
      eprint={2112.03491},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```
If citing version `0.1.0` of this GitHub repository directly, you can use the following citation:

```
@software{vmcnet2021github,
  author = {Jeffmin Lin and Gil Goldshlager and Lin Lin},
  title = {{VMCNet}: Flexible, general-purpose {VMC} framework, built on {JAX}},
  url = {http://github.com/jeffminlin/vmcnet},
  version = {0.1.0},
  year = {2021},
}
```
