#!/bin/bash
echo "Running mypy to check for type errors"
mypy vmcnet tests setup.py
echo "Running flake8 to check for code style errors"
flake8 --filename='./vmcnet/*.py','./tests/*.py','./setup.py' --max-line-length=88 --select=C,E,F,W,B,B950 --ignore=E203,W503 --per-file-ignores="__init__.py:F401"
echo "Running pydocstyle to check for docstring style errors"
pydocstyle ./vmcnet ./tests ./setup.py --match='(.)*\.py' --add-ignore=D401
