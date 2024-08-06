# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE for more info.

"""Module class to read and write PSR DADA files."""

from __future__ import annotations

__all__ = [
    "SkaUnpacker",
    "Unpacker",
    "UnpackOptions",
    "SKA_DIGI_SCALE_MEAN",
]

import dataclasses
from typing import Any, Dict, Protocol, Tuple

import numpy as np
import numpy.typing as npt

NDIM_REAL = 1
"""
Data are real valued.

There is only 1 dimension.
"""

NDIM_COMPLEX = 2
"""
Data are complex valued.

There are 2 dimensions per value.
"""

NBIT_8 = 8
"""Data are 8 bit signed integers."""

NBIT_16 = 16
"""Data are 16 bit signed integers."""

NBIT_FLOAT = -32
"""Data are 32 bit floating point numbers."""


@dataclasses.dataclass(kw_only=True, frozen=True)
class UnpackOptions:
    """A data class that defines the options to use during unpacking of data."""

    nbit: int
    """
    The number of bits per sample value.

    Note that this value is for 1 dimension of the data.  The total
    number of bits for a sample value is ``nbits * ndim``.
    """

    nchan: int
    """The number of channels per time sample."""

    npol: int
    """The number of polarisations per channel."""

    ndim: int
    """
    The number of dimensions for the sample value.

    If ``ndim=1`` then the data are real valued and if ``ndim=2`` then the data are complexed valued.
    """

    addition_args: dict = dataclasses.field(default_factory=dict)
    """Additional arguments that may specific to a given unpacker."""

    def __getattr__(self: UnpackOptions, key: str) -> Any:
        """Get an attribute given the key.

        :param key: the attribute key to get
        :type key: str
        :return: the attribute value, if it exists
        :rtype: Any
        :raises AttributeError: if attribute is not found
        """
        if key in self.addition_args:
            return self.addition_args[key]

        raise AttributeError(f"'UnpackOptions' object has no attribute '{key}'")


class Unpacker(Protocol):
    """A protocol that class should implement to unpack bytes of data.

    The input data is expected to be in time, frequency and polarisation (TFP) ordering.
    The output data is also in TFP.
    """

    def unpack(self, *, data: bytes, options: UnpackOptions) -> np.ndarray:
        """
        Unpack the data that is assumed to be a buffer of unsigned int8 (i.e. bytes).

        Implementations of this protocol can be used to convert from a byte array of data to the
        unpackers

        The input data is expected to be in time, frequency and polarisation (TFP) ordering.
        The output data is also in TFP.

        :param data: the input data array of bytes
        :type data: bytes
        :param options: the unpack options. This may be specific to the unpacker that may
            need additional options.
        :type options: UnpackOptions
        :return: the unpacked array of data in TFP ordering.
        :rtype: np.ndarray
        """


SKA_DIGI_SCALE_MEAN: Dict[int, Tuple[float, float]] = {
    1: (0.5, 0.5),
    2: (1.03, 0.0),
    4: (3.14, 0.0),
    8: (10.1, 0.0),
    16: (1106.4, 0.0),
    -32: (1.0, 0.0),
}
"""The scales and means applied to data in the SKA Generic Voltage Digitiser."""


class SkaUnpacker:
    """
    An unpacker that handles SKA packed data.

    This unpacker should be used to unpack data that comes from SKA generated files,
    such as the the flow through mode data that has been digitised.
    """

    def _rescale_and_reshape(
        self, *, data: np.ndarray, nchan: int, npol: int, ndim: int, nbit: int
    ) -> np.ndarray:
        data = data.reshape((-1, nchan, npol, ndim))

        (scale, mean) = SKA_DIGI_SCALE_MEAN[nbit]
        data = (data - mean) / scale

        if ndim == 2:
            data = data.view(dtype=np.complex64)

        assert data.shape[-1] == 1, f"expected {data.shape=} to be 1"
        # reduce the axis
        data = np.squeeze(data, axis=-1)

        return data

    def _unpack_simple(
        self, raw_data: bytes, nchan: int, npol: int, ndim: int, nbit: int, dtype: npt.DTypeLike
    ) -> np.ndarray:
        # always cast to float, as Numpy doesn't have a non-float based complex number.
        data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
        return self._rescale_and_reshape(data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit)

    def _unpack_bytes(self, raw_data: bytes, nchan: int, npol: int, ndim: int, nbit: int) -> np.ndarray:
        assert nbit in {1, 2, 4}, f"expected {nbit=} to be either 1, 2 or 4"
        bit_mask = np.int8(pow(2, nbit) - 1)
        msb = np.uint8(1 << (nbit - 1))

        values_per_byte = 8 // nbit
        values_per_sample = nchan * npol * ndim

        assert (len(raw_data) * values_per_byte) % values_per_sample == 0, (
            f"Expected number of samples={len(raw_data) * values_per_byte} to be divisible "
            f"by nchan*npol*ndim={nchan*npol*ndim}"
        )
        ndat = len(raw_data) * values_per_byte // values_per_sample

        data = np.zeros(shape=(len(raw_data) * values_per_byte), dtype=np.float32)

        for idx in range(ndat * nchan * npol * ndim):
            in_value_idx = idx // values_per_byte
            value_shift = idx % values_per_byte
            in_value = np.uint8(raw_data[in_value_idx])

            bit_shifted_value = in_value >> (nbit * value_shift)
            value = int(bit_shifted_value & bit_mask)

            # This handles the twos complement of negative numbers when nbits < 8.
            # The msb is a mask for the most significant bit of NBIT. Doing a bitwise
            # AND operation we can see if the number should be negative.  If it is then cast
            # the bit_mask as an integer and flip bits and then do bitwise a OR operation on
            # the value.
            #
            # e.g NBIT = 2, msb = 0b10, bit_mask = 0b11
            # if value = 0b11, then output value should be -1, if 0b10 then value should be -2
            if nbit != 1 and (value & msb):
                value |= ~int(bit_mask)

            data[idx] = np.float32(value)

        return self._rescale_and_reshape(data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit)

    def unpack(self, *, data: bytes, options: UnpackOptions) -> np.ndarray:
        """Unpack SKA specific data given the unpack option.

        This returns the unpacked data as a 3 dimensional Numpy array with the following
        dimensions:

            * time
            * frequency
            * polarisation

        This may return real or complex values based on the ``NDIM`` value in the
        header. If ``NDIM`` is 1 then real floating point data is returned, if 2 then
        complex value data is returned.

        :param data: the input byte array of data.
        :type data: bytes
        :param options: the options for unpacking
        :type options: UnpackOptions
        :return: the unpacked output array of data.
        :rtype: np.ndarray
        """
        assert options is not None, "expected options not to be None"

        nbit = options.nbit
        nchan = options.nchan
        npol = options.npol
        ndim = options.ndim

        if nbit == NBIT_8:
            unpacked = self._unpack_simple(
                raw_data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit, dtype=np.int8
            )
        elif nbit == NBIT_16:
            unpacked = self._unpack_simple(
                raw_data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit, dtype=np.int16
            )
        elif nbit == NBIT_FLOAT:
            unpacked = self._unpack_simple(
                raw_data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit, dtype=np.float32
            )
        else:
            unpacked = self._unpack_bytes(raw_data=data, nchan=nchan, npol=npol, ndim=ndim, nbit=nbit)

        return unpacked
