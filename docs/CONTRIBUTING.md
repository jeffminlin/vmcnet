# Contributing

We welcome contributions in the form of code, discussion, GitHub issues, or other avenues! If you wish to contribute a new feature or would like to address an outstanding todo/bug with the code, it can be very helpful to share your intentions via an [issue](https://github.com/jeffminlin/vmcnet/issues) to discuss the design plan and implementation details.

If contributing code, basic knowledge of git is assumed. Please start by forking the repository, and install your fork locally using Python 3.9, making sure to include the test suite (`pip install -e .[testing]`).

1. Add the main repository https://github.com/jeffminlin/vmcnet as an upstream remote to keep your fork synced as you develop (via fetch and rebase).
2. Create a branch on your fork for your development, ideally with a useful name that reflects your feature/bug-fix/etc.
3. After you implement your proposed code changes in your development branch, please run the code linters and formatters via `black .` and `lint.sh` (or have them run automatically by your editor -- we haven't yet set up git pre-commit hooks).
4. Please also run our tests via `pytest --cov=vmcnet --run_very_slow` and contribute to the test suite to help maintain the overall code quality.
5. When ready, you can submit your changes for review by opening a pull request against the main repository.

We use `mypy`, `flake8`, and `pydocstyle` to check code for style, and our linting configuration has been set up to follow the `black` formatter. We expect new contributions to include descriptive docstrings, and to be type hinted reasonably consistently. A code review and passing tests (via GitHub CI) will be required before your pull request can be merged.