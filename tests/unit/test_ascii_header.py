# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file contains unit tests for ska_pydada.AsciiHeader."""

from __future__ import annotations

import pathlib
import random
from typing import IO, Any

import pytest

from ska_pydada import AsciiHeader
from ska_pydada.ascii_header import MAX_ASCII_HEADER_SIZE


def test_ascii_heaer_load_from_file(header_file: pathlib.Path) -> None:
    """Test loading an AsciiHeader from a file."""
    header = AsciiHeader.from_file(header_file)

    assert len(header) == 9

    assert header.header_size == 16384
    assert header.get_int("HDR_SIZE") == 16384
    assert header.get_float("HDR_VERSION") == 1.0
    assert header.get_int("NCHAN") == 432
    assert header.get_int("NBIT") == 32
    assert header.get_int("NDIM") == 2
    assert header.get_int("NPOL") == 2
    assert header.get_int("RESOLUTION") == 1327104
    assert header.get_value("UTC_START") == "2017-08-01-15:53:29"
    assert header.get_value("TEST_SPACE") == "First"


def test_ascii_header_set_value() -> None:
    """Test setting and getting a value from an AsciiHeader."""
    header = AsciiHeader()

    # assert default header size
    assert header.header_size == 4096

    assert "FOO" not in header
    header.set_value("FOO", "bar")
    assert "FOO" in header
    assert header["FOO"] == "bar"


def test_ascii_header_from_bytes(header_file: pathlib.Path) -> None:
    """Test loading an AsciiHeader from bytes file."""
    data = header_file.read_bytes()

    header = AsciiHeader.from_bytes(data)

    assert len(header) == 9

    assert header.header_size == 16384
    assert header.get_int("HDR_SIZE") == 16384
    assert header.get_float("HDR_VERSION") == 1.0
    assert header.get_int("NCHAN") == 432
    assert header.get_int("NBIT") == 32
    assert header.get_int("NDIM") == 2
    assert header.get_int("NPOL") == 2
    assert header.get_int("RESOLUTION") == 1327104
    assert header.get_value("UTC_START") == "2017-08-01-15:53:29"


def test_ascii_header_from_str(header_file: pathlib.Path) -> None:
    """Test loading an AsciiHeader from a string."""
    data = header_file.read_text()

    data += "# Adding a comment - next line is just whitespace\n"
    data += "                \n"

    header = AsciiHeader.from_str(data)

    assert len(header) == 9

    assert header.header_size == 16384
    assert header.get_int("HDR_SIZE") == 16384
    assert header.get_float("HDR_VERSION") == 1.0
    assert header.get_int("NCHAN") == 432
    assert header.get_int("NBIT") == 32
    assert header.get_int("NDIM") == 2
    assert header.get_int("NPOL") == 2
    assert header.get_int("RESOLUTION") == 1327104
    assert header.get_value("UTC_START") == "2017-08-01-15:53:29"


def test_ascii_header_set_header_size() -> None:
    """Test updating the header size of an AsciiHeader."""
    header = AsciiHeader()

    assert header.header_size == 4096
    header.header_size = 16384

    assert int(header["HDR_SIZE"]) == 16384


def test_ascii_heaer_set_invalid_header_size() -> None:
    """Test that there is a minimum header size of AsciiHeader."""
    header = AsciiHeader()

    new_header_size = random.randint(1, 4095)

    with pytest.raises(AssertionError) as exc_info:
        header.header_size = new_header_size

    exc_value = exc_info.value
    assert str(exc_value) == (f"expected header_size={new_header_size} to be at least 4096")

    # test constructor
    with pytest.raises(AssertionError) as exc_info:
        header = AsciiHeader(header_size=new_header_size)

    exc_value = exc_info.value
    assert str(exc_value) == (f"expected header_size={new_header_size} to be at least 4096")


def test_ascii_header_resolution_property() -> None:
    """Test that resolution property uses property or is calculated."""
    header = AsciiHeader()

    # default resolution is 1 byte
    assert header.resolution == 1

    header.set_value("RESOLUTION", 1234)

    assert header.resolution == 1234

    del header["RESOLUTION"]

    # set config
    header["NCHAN"] = 432
    header["NBIT"] = 32
    header["NDIM"] = 2
    header["NPOL"] = 2
    header["UDP_NSAMP"] = 24

    assert header.resolution == 165888, f"expected {header.resolution=} to be 165888"


def test_ascii_header_to_bytes() -> None:
    """Test serialising AsciiHeader to bytes."""
    header_size = random.randint(4096, 4 * 4096)
    header = AsciiHeader(header_size=header_size)
    header["VERY_LONG_KEY_NAME_OVER_20_CHARS"] = 42

    header_str = str(header)
    expected_str = f"""HDR_SIZE            {header_size}
VERY_LONG_KEY_NAME_OVER_20_CHARS 42
"""
    assert header_str == expected_str

    header_bytes = header.to_bytes()
    assert len(header_bytes) == header_size

    expected_str_bytes = expected_str.encode()
    assert header_bytes[: len(expected_str_bytes)] == expected_str_bytes

    null_padding = bytes(header_size - len(expected_str_bytes))
    assert header_bytes[len(expected_str_bytes) :] == null_padding


def test_ascii_header_from_file_where_size_is_too_large(temp_file: IO[Any]) -> None:
    """Test that an exception is raised if file was too large."""
    bytes_dada = bytes(MAX_ASCII_HEADER_SIZE + 1)
    with open(temp_file.name, "wb") as fd:
        fd.write(bytes_dada)

    with pytest.raises(AssertionError) as exc_info:
        AsciiHeader.from_file(temp_file.name)

    assert str(exc_info.value) == (
        f"file {temp_file.name} in {MAX_ASCII_HEADER_SIZE + 1} bytes which is "
        f"greater than max size of {MAX_ASCII_HEADER_SIZE}"
    )


def test_ascii_header_only_accepts_key_that_are_strings() -> None:
    """Assert only str keys are accepted."""
    header = AsciiHeader()

    with pytest.raises(AssertionError) as exc_info:
        header[123] = "foobar"

    assert str(exc_info.value) == "AsciiHeader only accepts str keys."
