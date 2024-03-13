## The following should be standard includes
# include core makefile targets for release management
include .make/base.mk

# include python make support
include .make/python.mk

# include your own private variables for custom deployment configuration
-include PrivateRules.mak

PYTHON_SWITCHES_FOR_ISORT := --skip-glob="*/__init__.py" --py=310
PYTHON_SWITCHES_FOR_PYLINT = --disable=W,C,R --ignored-modules="ska_pst.grpc"
PYTHON_SWITCHES_FOR_AUTOFLAKE ?= --in-place --remove-unused-variables --remove-all-unused-imports --recursive --ignore-init-module-imports