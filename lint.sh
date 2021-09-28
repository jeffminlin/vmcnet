#!/bin/bash
echo "Running mypy to check for type errors"
mypy vmcnet tests
echo "Running flake8 to check for code style errors"
flake8 vmcnet tests
echo "Running pydocstyle to check for docstring style errors"
pydocstyle ./vmcnet ./tests
