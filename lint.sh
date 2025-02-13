#!/bin/bash
echo "Running mypy to check for type errors"
mypy vmcnet tests
echo "Running flake8 to check for code and documentation style errors"
flake8 vmcnet tests --select D --extend-ignore D401
echo "Running the black formatter to check for formatting errors"
black --check --diff --color vmcnet tests
