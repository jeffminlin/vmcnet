# vmcnet
Flexible, general-purpose VMC framework, built on JAX.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

A virtual environment is recommended. After cloning, use `pip install -e .` to install the package in editable/develop mode, or `pip install -e .["testing"]` if planning on running tests.

If running on a GPU, CUDA needs to be set up properly to work with JAX, and you will need to install the correct `jaxlib` wheel. See, e.g., https://github.com/google/jax#pip-installation.