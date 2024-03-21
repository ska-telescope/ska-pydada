# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file fixtures for the unit tests."""

from __future__ import annotations

import pathlib
from tempfile import NamedTemporaryFile
from typing import IO, Any, Generator

import pytest


@pytest.fixture
def temp_dada_file() -> Generator[IO[Any], None, None]:
    """Get a temp DADA file."""
    yield NamedTemporaryFile()


@pytest.fixture
def data_path() -> pathlib.Path:
    """Get the unit test data path."""
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture
def header_file(data_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the header.txt file."""
    return data_path / "header.txt"
