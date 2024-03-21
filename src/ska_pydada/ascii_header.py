# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE for more info.

"""Module class to write PSR DADA files."""

from __future__ import annotations

__all__ = [
    "AsciiHeader",
]

import pathlib
from collections import OrderedDict
from typing import Any

from .common import (
    BITS_PER_BYTE,
    DEFAULT_HEADER_SIZE,
    DEFAULT_KEY_PADDING,
    HEADER_SIZE_KEY,
    NBIT,
    NCHAN,
    NDIM,
    NPOL,
    RESOLUTION,
    UDP_NSAMP,
)


class AsciiHeader(OrderedDict):
    """A utility class to abstract over a DADA header.

    This class extends an ordered dictionary to allow inserting and
    retrieving values. Values are stored as strings.
    """

    def __init__(
        self: AsciiHeader,
        header_size: int = DEFAULT_HEADER_SIZE,
        **kwargs: Any,
    ) -> None:
        """Initialise an instance of an ``AsciiHeader``.

        :param header_size: the size of output header, defaults to DEFAULT_HEADER_SIZE
        :type header_size: int, optional
        :raises AssertionError: if ``header_size`` is less than 4096
        """
        assert (
            header_size >= DEFAULT_HEADER_SIZE
        ), f"expected {header_size=} to be at least {DEFAULT_HEADER_SIZE}"
        if HEADER_SIZE_KEY not in kwargs:
            kwargs[HEADER_SIZE_KEY] = str(header_size)
        super().__init__(**kwargs)

    @staticmethod
    def from_file(file: pathlib.Path | str) -> AsciiHeader:
        """Load a header from a file.

        The input file must but a text file, such as a config
        file. This will not handle a DADA file that has binary
        data.

        :param file: the path to the file to load.
        :type file: pathlib.Path | str
        :return: the file parsed as an ``AsciiHeader``
        :rtype: AsciiHeader
        """
        with open(file, "r") as fd:
            data = fd.read()
            return AsciiHeader.from_str(data)

    @staticmethod
    def from_bytes(data: bytes) -> AsciiHeader:
        """Get an instance of an ``AsciiHeader`` from supplied bytes.

        This converts the data to a string using ``decode`` and then
        calls the :py:meth:`AsciiHeader.from_str`.

        :param data: the data to parse.
        :type data: bytes
        :return: the bytes parsed as an ``AsciiHeader``
        :rtype: AsciiHeader
        """
        return AsciiHeader.from_str(data.decode())

    @staticmethod
    def from_str(data: str) -> AsciiHeader:
        """Get an instance of an ``AsciiHeader`` from supplied string.

        :param data: the data to parse.
        :type data: str
        :return: the string parsed as an ``AsciiHeader``
        :rtype: AsciiHeader
        """
        header: dict = {}
        for line in data.splitlines():
            line = line.replace("\0", " ").strip()

            # ignore a comment
            if line.startswith("#"):
                continue

            if len(line) == 0:
                continue

            [key, value] = line.lstrip().split(" ", maxsplit=1)
            assert len(key) > 0, f"Expected header key of line '{line}' to not be empty"
            header[key] = value.lstrip()

        return AsciiHeader(**header)

    def __setitem__(self: AsciiHeader, __key: Any, __value: Any) -> None:
        """Set an item on the header.

        This overrides the ``__setitem__`` method on the parent by
        ensuring all the values are converted to a string.

        :param __key: the key of the record to set
        :type __key: Any
        :param __value: the value of the record to set
        :type __value: Any
        """
        super().__setitem__(__key, str(__value))

    @property
    def header_size(self: AsciiHeader) -> int:
        """The size of header in bytes.

        This represents the number of bytes that a serialised header would
        be if a part of a DADA file. The header would be NULL filled
        if the output length is less than this value.

        :return: the header size, in bytes.
        :rtype: int
        """
        return int(self[HEADER_SIZE_KEY])

    @header_size.setter
    def header_size(self: AsciiHeader, header_size: int) -> None:
        assert (
            header_size >= DEFAULT_HEADER_SIZE
        ), f"expected {header_size=} to be at least {DEFAULT_HEADER_SIZE}"
        self[HEADER_SIZE_KEY] = str(header_size)

    @property
    def resolution(self: AsciiHeader) -> int:
        """Get the calculated resolution based on values in the header.

        This is the number of bytes in a stride of data.

        If the ``RESOLUTION`` key exists in the header than that value is used,
        else this is determined by ``NDIM``, ``NBIT``, ``NPOL``, ``NCHAN`` and
        ``UDP_NSAMP``. If not all the values exist then a value of ``1`` is returned.

        :return: the number of bytes for a stride of data.
        :rtype: int
        """
        try:
            return self.get_int(RESOLUTION)
        except KeyError:
            pass

        try:
            ndim = self.get_int(NDIM)
            nbit = self.get_int(NBIT)
            npol = self.get_int(NPOL)
            nchan = self.get_int(NCHAN)
            udp_nsamp = self.get_int(UDP_NSAMP)

            return nchan * nbit * ndim * npol * udp_nsamp // BITS_PER_BYTE
        except KeyError:
            return 1

    def set_value(self: AsciiHeader, key: str, value: Any) -> None:
        """Set a value in the header.

        :param key: the key of the header record to set.
        :type key: str
        :param value: the value of the record.
        :type value: Any
        """
        self[key] = value

    def get_value(self: AsciiHeader, key: str) -> str:
        """Get the value of a header record given a key.

        :param key: the key of the record to get.
        :type key: str
        :return: the value of the record as a string value.
        :rtype: str
        :raises KeyError: if key does not exist
        """
        return str(self[key])

    def get_int(self: AsciiHeader, key: str) -> int:
        """Get the header value as an integer.

        :param key: the key of the record to get.
        :type key: str
        :return: the value of the record as a string value.
        :rtype: str
        :raises KeyError: if key does not exist
        :raises ValueError: if value cannot be converted to an integer.
        """
        value = self[key]
        return int(value)

    def get_float(self: AsciiHeader, key: str) -> float:
        """Get the header value as a float.

        :param key: the key of the record to get.
        :type key: str
        :return: the value of the record as a string value.
        :rtype: str
        :raises KeyError: if key does not exist
        :raises ValueError: if value cannot be converted to a float.
        """
        value = self[key]
        return float(value)

    def __str__(self: AsciiHeader) -> str:
        r"""Convert the header to a string.

        Each entry in the header is formated so the header key is
        right padded with spaces to at least 20 characters. The value
        is then appended to that line and a line feed character ``\n``
        is added.

        :return: the header as a string.
        :rtype: str
        """
        output = ""
        for k, v in self.items():
            key_length = len(k)
            min_width = max(DEFAULT_KEY_PADDING, key_length + 1)
            output += f"{k: <{min_width}}{v}\n"

        return output

    def to_bytes(self: AsciiHeader) -> bytes:
        """Convert the header to bytes.

        :return: the header converted to bytes but padded with NULL
            chars to be :py:attr:`header_size` in bytes long.
        :rtype: bytes
        """
        output = str(self)
        output_bytes = bytearray(self.header_size)
        output_bytes[: len(output)] = output.encode()
        return bytes(output_bytes)
