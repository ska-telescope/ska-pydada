# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE for more info.

"""Module class to read and write PSR DADA files."""

from __future__ import annotations

__all__ = [
    "DadaFile",
]

import logging
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from .ascii_header import AsciiHeader
from .common import DEFAULT_DATA_CHUNK_SIZE, DEFAULT_HEADER_SIZE, SIZE_OF_COMPLEX64, SIZE_OF_FLOAT32


class DadaFile:
    """Class that can be used to read a PSR DADA file."""

    def __init__(
        self: DadaFile,
        header: AsciiHeader | None = None,
        raw_data: bytes | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        """Create instance of DADA file object.

        :param header: the :py:class:`AsciiHeader` to use to store metadata, defaults to None
        :type header: AsciiHeader | None, optional
        :param raw_data: the raw data to store, defaults to None.
        :type raw_data: bytes | None, optional
        :param logger: logger used for debugging purposes, defaults to None
        :type logger: logging.Logger | None, optional
        """
        self._header: AsciiHeader = header or AsciiHeader()
        self._raw_data: bytes = raw_data or b""
        self._file_data_ptr: int = 0
        self._logger = logger or logging.getLogger(__name__)
        self._file: pathlib.Path | None = None

    @staticmethod
    def load_from_file(
        file: pathlib.Path | str,
        chunk_size: int = DEFAULT_DATA_CHUNK_SIZE,
        logger: logging.Logger | None = None,
    ) -> DadaFile:
        """Load a DADA file and create an instance of a :py:class:`DadaFile`.

        :param file: a path to the file to load.
        :type file: pathlib.Path | str
        :param chunk_size: the maximum amount of data to load, defaults to DEFAULT_DATA_CHUNK_SIZE.
            If the file is more than the maximum amount then more data can be read by calling
            :py:meth:`load_next` on the instance returned.
        :type chunk_size: int, optional
        :param logger: the logger to use for debugging, defaults to None
        :type logger: logging.Logger | None, optional
        :return: an instance of a :py:class:`DadaFile` in which the header and data can be read.
        :rtype: DadaFile
        """
        file = pathlib.Path(file)
        assert file.exists() and file.is_file()

        with open(file, "rb") as f:
            # we now have bytes
            assert f.seekable(), "expected file to be seekable to allow reading of file."

            header = AsciiHeader.from_bytes(f.read(DEFAULT_HEADER_SIZE))

            if header.header_size != DEFAULT_HEADER_SIZE:
                f.seek(0)
                header = AsciiHeader.from_bytes(f.read(header.header_size))

        dada_file = DadaFile(header=header, logger=logger)
        dada_file._file = file
        dada_file.load_next(chunk_size=chunk_size)
        return dada_file

    def dump(
        self: DadaFile,
        file: pathlib.Path | str,
    ) -> None:
        """Dump the data to an external file.

        This method takes a path to file location to write to.
        This method will overwrite an existing file if it exists.

        :param file: the path to the file to write to.
        :type file: pathlib.Path | str
        """
        file = pathlib.Path(file)
        if file.exists():
            self._logger.warning(f"{str(file)} already exists, overwriting it")

        with open(file, "wb") as fd:
            header_bytes = self.header.to_bytes()
            fd.write(header_bytes)
            fd.write(self.raw_data)
            fd.flush()

    def load_next(self: DadaFile, *, chunk_size: int = DEFAULT_DATA_CHUNK_SIZE) -> int:
        """Load the next chunk of data.

        This will load the next chunk of data as a multiple of the ``RESOLUTION``
        of the data, which comes from :py:attr:`AsciiHeader.resolution`. The amount
        of data that can be loaded can be set by passing through a ``chunk_size``
        parameter, the default value is 4MB of dada.

        :param chunk_size: the amount of data to load, defaults to DEFAULT_DATA_CHUNK_SIZE.
            This method will round up to the nearest ``RESOLUTION`` or to the end
            of the file depending if there is not enough data left to read.
        :type chunk_size: int, optional
        :return: the amount of data loaded.
        :rtype: int
        """
        assert self._file is not None, "can only load more data if created from a file"
        file_size = self._file.stat().st_size

        if chunk_size < 0:
            chunk_size = file_size - self.header_size

        self._logger.debug(f"requested to load {chunk_size} bytes of data")
        resolution = self.resolution
        chunk_size = max(resolution, chunk_size)

        if chunk_size % resolution != 0:
            # round up the read size
            chunk_size = (chunk_size // resolution + 1) * resolution

        read_offset = self.header_size + self._file_data_ptr
        chunk_size = min(chunk_size, file_size - read_offset)

        if chunk_size <= 0:
            # already have read the whole file
            return 0

        self._logger.debug(f"loading {chunk_size} bytes of data")

        with open(self._file, "rb") as f:
            f.seek(read_offset)
            self._raw_data = f.read(chunk_size)
            self._file_data_ptr += chunk_size

        return chunk_size

    def est_num_chunks(self: DadaFile, chunk_size: int = DEFAULT_DATA_CHUNK_SIZE) -> int:
        """Get an estimate of number of data chunks given the ``chunk_size``.

        This method calculates the estimate number of chucks of data the whole
        file has given the ``chunk_size`` parameter.  This does not take into
        account that the :py:meth:`load_next` method rounds this value up
        to the nearest ``RESOLUTION``.

        :param chunk_size: the size of a chunk in bytes, defaults to DEFAULT_DATA_CHUNK_SIZE
        :type chunk_size: int, optional
        :return: the estimated number of chunks of data.
        :rtype: int
        """
        assert self._file is not None
        return int(np.ceil(self.data_size / chunk_size))

    @property
    def header(self: DadaFile) -> AsciiHeader:
        """Get the header for the DADA file.

        :return: the header of the file.
        :rtype: AsciiHeader
        """
        return self._header

    @property
    def header_size(self: DadaFile) -> int:
        """Get the size of the header, in bytes.

        :return: the size of the header in bytes.
        :rtype: int
        """
        return self._header.header_size

    @header_size.setter
    def header_size(self: DadaFile, header_size: int) -> None:
        r"""Set the header size of the output file.

        This must be at least 4096 bytes as the DADA spec expects that.
        If the string of the output header is shorter than this the
        header will be filled with NULL characters (i.e. ``\x00``).

        :param header_size: the size to set for the output header.
        :type header_size: int
        """
        self._header.header_size = header_size

    @property
    def data_size(self: DadaFile) -> int:
        """Get the overall size of the data block of the DADA file.

        This value is equal the total file size minus the size of the
        header.  If this instance was not loaded from a file (i.e.
        currently creating a file before dumping to the file system)
        then this value returns the size of the raw data that has been
        added to the instance.

        :return: the size of the data block with the output file in bytes.
        :rtype: int
        """
        if self._file is not None:
            return self._file.stat().st_size - self.header_size
        return len(self._raw_data)

    @property
    def raw_data(self: DadaFile) -> bytes:
        """Get the currently loaded data as a byte array.

        :return: the currently loaded data as a byte array.
        :rtype: bytes
        """
        return self._raw_data

    @property
    def resolution(self: DadaFile) -> int:
        """Get the calculated resolution of the file.

        See :py:attr:`AsciiHeader.resolution` for details.

        :return: the resolution of the data.
        :rtype: int
        """
        return self.header.resolution

    def set_data(self: DadaFile, data: np.ndarray) -> None:
        """Set the data of the file using a Numpy array.

        This does not persist the data. A call to :py:meth:`dump`
        is required to store the data.

        Note that data is serialised to bytes using native endianness.

        :param data: a Numpy array of the data to store. This can
            be in any shape or have any data type that can be converted
            to numerical data as bytes.
        :type data: np.ndarray
        """
        self._raw_data = data.tobytes()

    def data(
        self: DadaFile,
        shape: np._ShapeType | None = None,
        dtype: npt.DTypeLike = np.uint8,
    ) -> np.ndarray:
        """Get the data as a numpy array.

        This will return the raw byte data as a Numpy array
        with a data type of ``dtype``.

        If the ``shape`` parameter is specified then the array
        will be reshaped using row major (i.e. 'C' format).
        If no shape is provided a 1-dimensional array is returned
        and the client will need to perform the reshaping themselves.

        :param shape: the required shape of the output array, defaults to None.
            If no shape provided then a 1D array is returned.
        :type shape: np._ShapeType | None, optional
        :param dtype: the data type to have the raw bytes converted to,
            defaults to np.uint8.
        :type dtype: npt.DTypeLike, optional
        :return: the raw data converted to a Numpy array with a given type and shape.
        :rtype: np.ndarray
        """
        data = np.frombuffer(self.raw_data, dtype=dtype)
        if shape is not None:
            data = data.reshape(shape)

        return data

    def data_bytes(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the raw data as a Numpy byte array.

        This gets the raw data bytes and converts it to a Numpy array with an
        optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a Numpy byte array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.uint8)

    def data_i8(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the data as a signed 8-bit integer Numpy array.

        This parses the raw data as signed 8-bit integers and returns it as a Numpy
        array with an optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a signed 8-bit integer Numpy array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.int8)

    def data_i16(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the data as a signed 16-bit integer Numpy array.

        This parses the raw data as signed 16-bit integers and returns it as a Numpy
        array with an optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a signed 16-bit integer Numpy array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.int16)

    def data_i32(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the data as a signed 32-bit integer Numpy array.

        This parses the raw data as signed 32-bit integers and returns it as a Numpy
        array with an optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a signed 32-bit integer Numpy array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.int32)

    def data_f32(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the data as a 32-bit floating point Numpy array.

        This parses the raw data as 32-bit floating point numbers and returns it as
        a Numpy array with an optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a 32-bit floating point Numpy array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.float32)

    def data_c64(self: DadaFile, shape: np._ShapeType | None = None) -> np.ndarray:
        """Get the data as a 64-bit complex valued Numpy array.

        Numpy's ``complex64`` is stored as 2 32-bit floating point numbers, this is
        why this is ``c64`` as 64 bits are used to represent the number.

        This parses the raw data as 64-bit complex numbers and returns it as
        a Numpy array with an optional shape.

        :param shape: the desired output shape, defaults to None.
            If not set this will return a 1D array of the full size.
        :type shape: np._ShapeType | None, optional
        :return: the raw data as a 64-bit complex value Numpy array.
        :rtype: np.ndarray
        """
        return self.data(shape=shape, dtype=np.complex64)

    def as_time_freq_pol(self: DadaFile) -> np.ndarray:
        """Get the data as time, frequency and polarisation 3 dimensional array.

        This returns the raw data as a 3 dimensional Numpy array with the following
        dimensions:

            * time
            * frequency
            * polarisation

        The ``NCHAN`` header value defines the number of frequency channels.
        The ``NPOL`` parameter defines the number of polarisations.

        This may return real or complex values based on the ``NDIM`` value in the
        header. If ``NDIM`` is 1 then real floating point data is returned, if 2 then
        complex value data is returned. In both cases ``NBIT`` is assumed to be 32.
        For all other values of ``NDIM`` then an assertion error is raised.

        The number of time samples is defined as a free dimension the the shape of the
        data. This uses Numpy's standard of passing ``-1`` as the size of the dimension
        and lets Numpy determine the shape.

        :return: the raw data converted into a TFP Numpy array.
        :rtype: np.ndarray
        """
        npol = self.header.get_int("NPOL")
        nchan = self.header.get_int("NCHAN")
        ndim = self.header.get_int("NDIM")

        assert ndim in {1, 2}, f"currently on supports real or complex valued data. ndim={ndim}"

        if ndim == 2:
            assert self.data_size % (npol * nchan * SIZE_OF_COMPLEX64) == 0, (
                f"expected data size to be a multiple of {npol * nchan * SIZE_OF_COMPLEX64}"
                f" but was {self.data_size}"
            )
            return self.data_c64(shape=(-1, nchan, npol))
        else:
            assert self.data_size % (npol * nchan * SIZE_OF_FLOAT32) == 0, (
                f"expected data size to be a multiple of {npol * nchan * SIZE_OF_FLOAT32}"
                f" but was {self.data_size}"
            )
            return self.data_f32(shape=(-1, nchan, npol))

    def set_header_value(self: DadaFile, key: str, value: Any) -> None:
        """Set a header value to a given value.

        See :py:meth:`AsciiHeader.set_value` for more details.

        :param key: the key of the header to set.
        :type key: str
        :param value: the value to set the header record to.
        :type value: Any
        """
        self.header.set_value(key, value)

    def get_header_value(self: DadaFile, key: str) -> str:
        """Get a header value as a string value.

        :param key: the header key to get the value of.
        :type key: str
        :return: the value as a string
        :rtype: str
        :raises KeyError: if key doesn't exist
        """
        return self.header.get_value(key)

    def get_header_int(self: DadaFile, key: str) -> int:
        """Get the header value as an integer value.

        :param key: the header key to get the value of.
        :type key: str
        :return: the value as an integer
        :rtype: int
        :raises KeyError: if key doesn't exist
        :raises ValueError: if value cannot be converted to an integer.
        """
        return self.header.get_int(key)

    def get_header_float(self: DadaFile, key: str) -> float:
        """Get the header value as a float value.

        :param key: the header key to get the value of.
        :type key: str
        :return: the value as a float
        :rtype: float
        :raises KeyError: if key doesn't exist
        :raises ValueError: if value cannot be converted to a float.
        """
        return self.header.get_float(key)
