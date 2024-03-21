# -*- coding: utf-8 -*-
#
# This file is part of the SKA PST project
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE for more info.

"""Pytest conftest.py to set up fixtures."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="session")
def logging_level(pytestconfig: pytest.Config) -> int:
    """Get the logging level to use for tests."""
    logging_level = pytestconfig.getoption("loglevel")
    return logging.getLevelName(logging_level)


@pytest.fixture(scope="session")
def logger(logging_level: int) -> logging.Logger:
    """Get logger used within tests."""
    logger = logging.getLogger("TESTLOGGER")
    logger.setLevel(logging_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] [%(threadName)s] [%(filename)s:%(lineno)s] %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options for PyTest command line."""
    parser.addoption(
        "--loglevel",
        action="store",
        default="INFO",
        type=str,
        help="set internal logging level",
    )
