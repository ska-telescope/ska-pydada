# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE for more info.

"""Module with common values used in PyDADA."""

DEFAULT_HEADER_SIZE: int = 4096
"""
The default size of a DADA header.

This is also the minimum size of the header based on the DADA file format.
"""

DEFAULT_KEY_PADDING: int = 20
"""This is the default length of a DADA header key including trailing spaces."""

SIZE_OF_FLOAT32: int = 4
"""Size of a 32-bit float in bytes."""

SIZE_OF_COMPLEX64: int = 8
"""
Size of a complex value in bytes.

Each dimension is a 32-bit floating point number.
"""

BITS_PER_BYTE: int = 8
"""Number of bits per byte."""

MEGABYTE: int = 1024 * 1024
"""Number of bytes in a megabyte."""

# 4MB
DEFAULT_DATA_CHUNK_SIZE: int = 4 * MEGABYTE
"""Default size of a chunck of data to read."""

# DADA Header keys
HEADER_SIZE_KEY: str = "HDR_SIZE"
"""The header key used for header size."""

RESOLUTION: str = "RESOLUTION"
"""The header key used for knowing the stride/resolution of data."""

NDIM: str = "NDIM"
"""
The header key used to define number of dimensions of the data.

If 1 then data is real valued, if 2 then the data is complex valued.
"""

NBIT: str = "NBIT"
"""
The header key used to define the number of bits per dimension.

The total number of bits per value is ``NDIM * NBIT``. As an example
for floating point complex then this value is ``NBIT=32`` but the total
bits per value is ``64`` as ``NDIM`` is ``2``.
"""

NPOL: str = "NPOL"
"""The header key used to define the number of polarisations in the data."""

NCHAN: str = "NCHAN"
"""The header key used to define the number of frequency channels in the data."""

UDP_NSAMP: str = "UDP_NSAMP"
"""The header key used to define the number of time samples in a data heap, defines the RESOLUTION size."""
