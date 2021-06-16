"""Setup vmcnet for pip install."""
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "flax",
    "jax",
    "jaxlib",
    "kfac_ferminet_alpha @ git+https://github.com/deepmind/deepmind_research#egg=kfac_ferminet_alpha&subdirectory=kfac_ferminet_alpha",  # noqa: E501
    "numpy",
]

EXTRA_PACKAGES = {"testing": ["black", "chex", "flake8", "pytest"]}

setup(
    name="vmcnet",
    version="0.1",
    description="Flexible, general-purpose VMC framework, built on JAX.",
    url="https://github.com/jeffminlin/vmcnet",
    packages=find_packages(exclude=["tests"]),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
)
