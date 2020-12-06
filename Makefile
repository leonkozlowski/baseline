.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

lint: ## check style with flake8
	flake8 baseline

test: ## run tests quickly with the default Python
	pytest

test-verbose: ## run tests on every Python version with tox
	pytest -vv --cov=baseline --cov-report term-missing

black: ## format with black
	black --line-length 78 baseline

isort: ## format and sort ipmorts
	isort --multi-line 3 --trailing-comma .

mypy: ## line with mypy
	mypy app
