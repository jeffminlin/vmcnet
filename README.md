# VMCNet
Framework for training first-quantized neural network wavefunctions using VMC, built on [JAX](https://github.com/google/jax).

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository is built to serve two purposes:
1. To enable the development and testing of new architectures and algorithms for training neural network wavefunctions in first quantization.
2. To serve as a companion codebase to several papers, listed below.

This repository was built as a JAX port of an internal TensorFlow project started in 2019 which itself was initially inspired by the work of [David Pfau, James S. Spencer, Alexander G. D. G. Matthews, and W. M. C. Foulkes](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429). Their repository can be found [here](https://github.com/deepmind/ferminet).

## Installation

Python 3.9 is required, and a virtual environment is recommended. After cloning, use `pip install -e .` to install the package in editable/develop mode, or `pip install -e .[testing]` if planning on running tests.

If running on a GPU, CUDA needs to be set up properly to work with JAX, and you will need to install the correct `jaxlib` wheel. See, e.g., https://github.com/google/jax#installation.

## Python API

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

## Command-line

Alternatively, a command-line interface has been implemented which provides more streamlined access to subsets of the repository via setting [ConfigDict](https://github.com/google/ml_collections) objects.

There are two scripts which have been exposed thus far: `vmc-molecule` and `vmc-statistics`.

The primary command `vmc-molecule` calls `train.runners.run_molecule`. See [`train.default_config.get_default_config()`](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/default_config.py#L60) to explore the options which have been exposed to the command-line. To edit these options at the command-line, use the "`--config.`" prefix. For example,
```sh
vmc-molecule \
    --config.problem.ion_pos="((0.0, 0.0, -2.0), (0.0, 0.0, 2.0))" \
    --config.problem.ion_charges="(7.0, 7.0)" \
    --config.problem.nelec="(7, 7)" \
    --config.model.ferminet.full_det="True" \
```
will train the full single-determinant FermiNet on the nitrogen molecule at dissociating bond length 4.0 for 2e5 epochs on 2000 walkers. By default the SPRING optimizer will be used,
which for now only works on a single device. It is possible to run on multiple GPUs with other optimizers, for example by adding the following:
```
--config.distribute=True \\
--config.vmc.optimizer_type=kfac
```

You can also reload and evaluate or continue training from previous checkpoints via the "`--reload.`" prefix. The options can be seen in [`train.default_config.get_default_reload_config()`](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/train/default_config.py#L47). The reloading will only occur if `--reload.logdir` is set.

The `vmc-statistics` command calls `train.runners.vmc_statistics`. This simple script is designed to be compatible with the output of an evaluation run with `vmc-molecule`, but can accept any path to a file which contains local energies (a file with nchains x nepochs energies). It computes and saves a json file containing the average energy, the sample variance, the estimated integrated autocorrelation, and the estimated standard error. The options can be viewed simply via `vmc-statistics -h`.


## SPRING optimizer

The preprint describing the SPRING optimizer can be found at https://arxiv.org/abs/2401.10190 and can be cited via:
```
@misc{goldshlager2024kaczmarzinspired,
      title={A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions}, 
      author={Gil Goldshlager and Nilin Abrahamsen and Lin Lin},
      year={2024},
      eprint={2401.10190},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```
VMCNet can be used straightforwardly to reproduce the results in the preprint. For example to run the preliminary optimization phase
for the carbon atom use the following command:
```
vmc-molecule \
--config.logdir=/path/to/logs \
--config.save_to_current_datetime_subfolder=False \
--config.subfolder_name=C_preliminary \
--config.problem.ion_pos='((0.,0.,0.),)' \
--config.problem.ion_charges='(6.,)' \
--config.problem.nelec='(4,2)' \
--config.vmc.optimizer_type=kfac \
--config.vmc.nchains=1000 \
--config.vmc.nepochs=1000 \
--config.eval.nepochs=0 \
--config.model.ferminet.ndeterminants=16 \
--config.model.ferminet.full_det=True \
--config.vmc.checkpoint_every=1000
```
To then run the main optimization phase with SPRING use:
```
vmc-molecule \
--reload.logdir=/path/to/logs/C_preliminary \
--reload.checkpoint_relative_file_path=checkpoints/1000.npz \
--reload.new_optimizer_state=True \
--reload.append=False \
--config.logdir=/path/to/logs/C_SPRING \
--config.vmc.nchains=1000 \
--config.vmc.nepochs=100000 \
--config.eval.nchains=2000 \
--config.eval.nepochs=20000 \
--config.vmc.optimizer_type=spring \
--config.vmc.optimizer.spring.learning_rate=0.02
```
SPRING is also effective when run from the start; the preliminary optimization phase was only included to make comparisons between optimizers more fair.
Other optimizers and hyperparameters can be configured following the options in vmcnet/train/default_config.py.


## Cite

The paper which originally introduced this repository can be found at https://doi.org/10.1016/j.jcp.2022.111765 and can be cited via:
```
@article{lin2023explicitly,
  title={Explicitly antisymmetrized neural network layers for variational Monte Carlo simulation},
  author={Lin, Jeffmin and Goldshlager, Gil and Lin, Lin},
  journal={Journal of Computational Physics},
  volume={474},
  pages={111765},
  year={2023},
  publisher={Elsevier}
}
```
To cite version `0.2.0` of this GitHub repository directly you can use the following:

```
@software{vmcnet2024github,
  author = {Jeffmin Lin, Gil Goldshlager, Nilin Abrahamsen, and Lin Lin},
  title = {VMCNet: Framework for training first-quantized neural network wavefunctions using VMC, built on JAX},
  url = {http://github.com/jeffminlin/vmcnet},
  version = {0.2.0},
  year = {2024},
}
```

## Contributing

See [how to contribute](CONTRIBUTING.md).
