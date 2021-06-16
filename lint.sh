#!/bin/bash
echo "Running mypy to check for type errors"
mypy .
echo "Running flake8 to check for code style errors"
flake8 --max-line-length=88 --select=C,E,F,W,B,B950 --ignore=E203,W503 --per-file-ignores="__init__.py:F401"
echo "Running pydocstyle to check for docstring style errors"
pydocstyle --match='(.)*.py' --add-ignore=D401
