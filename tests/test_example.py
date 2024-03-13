#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the ska_python_skeleton module."""
import pytest

from ska_pydada import SKAPyDada

# from ska.skeleton import SKA, function_example


# TODO: Replace all the following examples
# with tests for the ska_python_skeleton package code
def test_something():
    """Assert with no defined return value."""
    assert True


def test_with_error():
    """Assert raising error."""
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise ValueError


# Fixture example
@pytest.fixture
def an_object():
    """Define fixture for subsequent test."""
    return {}


def test_ska_python_skeleton(an_object):
    """Assert fixture return value."""
    assert an_object == {}


def test_package():
    """Assert the ska_python_skeleton package code."""
    # assert function_example() is None
    foo = SKAPyDada()
    assert foo.example_2() == 2
    assert foo.example() is None
