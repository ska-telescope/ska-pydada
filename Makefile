## The following should be standard includes
# include core makefile targets for release management
include .make/base.mk

# include python make support
include .make/python.mk

# include your own private variables for custom deployment configuration
-include PrivateRules.mak

POETRY_VERSION ?=1.8.2
POETRY_CONFIG_VIRTUALENVS_CREATE=false

PYTHON_LINE_LENGTH = 110
PYTHON_LINT_TARGET = src/ tests/
PYTHON_SWITCHES_FOR_FLAKE8 := --extend-ignore=BLK,T --enable=DAR104 --ignore=E203,FS003,W503,N802 --max-complexity=10 \
    --rst-roles=py:attr,py:class,py:const,py:exc,py:func,py:meth,py:mod \
		--rst-directives=deprecated,uml

ifeq ($(findstring notebook, $(MAKECMDGOALS)),notebook)
	PYTHON_LINT_TARGET=$(NOTEBOOK_LINT_TARGET)
	PYTHON_SWITCHES_FOR_FLAKE8 := --extend-ignore=BLK,T --enable=DAR104 --ignore=E203,FS003,W503,N802,D100,D103 --max-complexity=10
endif

PYTHON_SWITCHES_FOR_ISORT := --skip-glob="*/__init__.py" --py=310
PYTHON_SWITCHES_FOR_PYLINT = --disable=W,C,R
PYTHON_SWITCHES_FOR_AUTOFLAKE ?= --in-place --remove-unused-variables --remove-all-unused-imports --recursive --ignore-init-module-imports

PYTHON_VARS_AFTER_PYTEST := $(PYTHON_VARS_AFTER_PYTEST) --cov-config=$(PWD)/.coveragerc --disable-pytest-warnings -rP

mypy:
	$(PYTHON_RUNNER) mypy --config-file mypy.ini $(PYTHON_LINT_TARGET)

flake8:
	$(PYTHON_RUNNER) flake8 --show-source --statistics $(PYTHON_SWITCHES_FOR_FLAKE8) $(PYTHON_LINT_TARGET)

python-post-format:
	$(PYTHON_RUNNER) autoflake $(PYTHON_SWITCHES_FOR_AUTOFLAKE) $(PYTHON_LINT_TARGET)

notebook-post-format:
	$(PYTHON_RUNNER) nbqa autoflake $(PYTHON_SWITCHES_FOR_AUTOFLAKE) $(PYTHON_LINT_TARGET)

notebook-format: notebook-pre-format notebook-do-format notebook-post-format

python-post-lint: mypy

notebook-post-lint:
	$(PYTHON_RUNNER) nbqa mypy --config-file=mypy.ini $(PYTHON_LINT_TARGET)

.PHONY: python-post-format, notebook-format, notebook-post-format, python-post-lint, mypy, flake8
