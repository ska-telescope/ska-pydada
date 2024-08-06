# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file contains unit tests for ska_pydada.DadaFile."""

from __future__ import annotations

import pathlib
from typing import IO, Any, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from ska_pydada import AsciiHeader, DadaFile
from ska_pydada.common import DEFAULT_DATA_CHUNK_SIZE
from ska_pydada.unpacker import UnpackOptions


def test_dada_file_create() -> None:
    """Test creating of a DADA file."""
    dada_file = DadaFile()

    header = dada_file.header
    assert header is not None
    assert dada_file.header_size == 4096

    dada_file.header_size = 30000
    assert header.header_size == 30000
    header.header_size = 12000
    assert dada_file.header_size == 12000

    assert dada_file.data_size == 0

    data_array = np.random.random(4000).astype(dtype=np.float32)
    dada_file.set_data(data_array)
    assert dada_file.data_size == 16000

    raw_data = dada_file.raw_data
    assert raw_data == data_array.tobytes()

    assert dada_file.resolution == 1
    dada_file.set_header_value("RESOLUTION", 123456)
    assert dada_file.get_header_int("RESOLUTION") == 123456
    assert dada_file.resolution == 123456

    dada_file.set_header_value("CAT_DOG", 1.234)
    assert dada_file.get_header_float("CAT_DOG") == 1.234

    dada_file.set_header_value("UTC_START", "2017-08-01-15:53:29")
    assert dada_file.get_header_value("UTC_START") == "2017-08-01-15:53:29"


def test_dada_file_load_from_file(temp_file: IO[Any], header_file: pathlib.Path) -> None:
    """Test load from file."""
    header = AsciiHeader.from_file(header_file)
    raw_data = np.random.rand(400).astype(dtype=np.float32).tobytes()

    dada_file = DadaFile(
        header=header,
        raw_data=raw_data,
    )
    dada_file.dump(file=temp_file.name)

    loaded_dada_file = DadaFile.load_from_file(temp_file.name)

    assert header == loaded_dada_file.header
    assert dada_file.raw_data == loaded_dada_file.raw_data

    assert loaded_dada_file.data_size == len(raw_data)


def test_dada_file_load_large_file(temp_file: IO[Any], header_file: pathlib.Path) -> None:
    """Test loading a large file greater than 4MB."""
    header = AsciiHeader.from_file(header_file)

    # generate raw data floats
    num_floats = 20 * header.resolution
    raw_data = np.random.rand(num_floats).astype(dtype=np.float32).tobytes()
    assert len(raw_data) >= DEFAULT_DATA_CHUNK_SIZE

    dada_file = DadaFile(header=header, raw_data=raw_data)
    dada_file.dump(file=temp_file.name)

    new_dada_file = DadaFile.load_from_file(file=temp_file.name)
    assert len(new_dada_file.raw_data) >= DEFAULT_DATA_CHUNK_SIZE
    assert len(new_dada_file.raw_data) % header.resolution == 0
    assert new_dada_file.data_size == len(raw_data)
    assert new_dada_file.est_num_chunks() == int(np.ceil(len(raw_data) / DEFAULT_DATA_CHUNK_SIZE))

    # reset the data pointer to 0
    new_dada_file._file_data_ptr = 0

    count = 0
    while True:
        num_bytes_read = new_dada_file.load_next()
        if num_bytes_read == 0:
            break

        assert num_bytes_read % header.resolution == 0
        count += 1

    assert new_dada_file.est_num_chunks() >= count


def test_dada_file_as_time_freq_pol_float32(temp_file: IO[Any], header_file: pathlib.Path) -> None:
    """Test DadaFile converting time freq pol for real valued data."""
    header = AsciiHeader.from_file(header_file)

    nchan = header.get_int("NCHAN")
    npol = header.get_int("NPOL")
    header.set_value("NDIM", 1)

    ntime = np.random.randint(10, 100)
    tfp_data = np.random.rand(ntime, nchan, npol).astype(dtype=np.float32)

    raw_data = tfp_data.tobytes()

    dada_file = DadaFile(header=header, raw_data=raw_data)
    dada_file.dump(file=temp_file.name)

    dada_file = DadaFile.load_from_file(file=temp_file.name)
    assert dada_file.raw_data == raw_data

    actual_tfp_data = dada_file.as_time_freq_pol()
    assert actual_tfp_data.dtype == np.float32
    assert actual_tfp_data.shape == (ntime, nchan, npol)
    np.testing.assert_allclose(actual_tfp_data, tfp_data)


def test_dada_file_as_time_freq_pol_complex64(temp_file: IO[Any], header_file: pathlib.Path) -> None:
    """Test DadaFile converting time freq pol for complex valued data."""
    header = AsciiHeader.from_file(header_file)

    nchan = header.get_int("NCHAN")
    npol = header.get_int("NPOL")
    header.set_value("NDIM", 2)

    ntime = np.random.randint(10, 100)
    tfp_data = np.random.rand(ntime, nchan, 2 * npol).astype(dtype=np.float32)
    tfp_data = tfp_data.view(np.complex64)

    raw_data = tfp_data.tobytes()

    dada_file = DadaFile(header=header, raw_data=raw_data)
    dada_file.dump(file=temp_file.name)

    dada_file = DadaFile.load_from_file(file=temp_file.name)
    assert dada_file.raw_data == raw_data

    actual_tfp_data = dada_file.as_time_freq_pol()
    assert actual_tfp_data.dtype == np.complex64
    assert actual_tfp_data.shape == (ntime, nchan, npol)
    np.testing.assert_allclose(actual_tfp_data, tfp_data)


DTYPE_SIZE = {
    np.int8: 1,
    np.int16: 2,
    np.int32: 4,
    np.float32: 4,
    np.complex64: 8,
}


@pytest.mark.parametrize(
    "dtype,data_fn,shape",
    [
        (np.uint8, "data_bytes", None),
        (np.uint8, "data_bytes", (128, 432, 2)),
        (np.int8, "data_i8", None),
        (np.int8, "data_i8", (128, 432, 2)),
        (np.int16, "data_i16", None),
        (np.int16, "data_i16", (64, 432, 2)),
        (np.int32, "data_i32", None),
        (np.int32, "data_i32", (32, 432, 2)),
        (np.float32, "data_f32", None),
        (np.float32, "data_f32", (32, 432, 2)),
        (np.complex64, "data_c64", None),
        (np.complex64, "data_c64", (16, 432, 2)),
    ],
)
def test_dada_file_data_formats(
    dtype: npt.DTypeLike, data_fn: str, shape: tuple[int, int, int] | None
) -> None:
    """Test DadaFile getting data in different formats."""
    ntime = 32
    nchan = 432
    npol = 2

    # create some random bytes, length is 4 * ntime * nchan * npol bytes
    raw_data = np.random.rand(ntime, nchan, npol).astype(np.float32).tobytes()

    dada_file = DadaFile(raw_data=raw_data)

    data_out: np.ndarray = getattr(dada_file, data_fn)(shape=shape)
    assert data_out.dtype == dtype

    expected_out = np.frombuffer(raw_data, dtype=dtype)
    if shape is not None:
        assert data_out.shape == shape
        expected_out = expected_out.reshape(shape)

    np.testing.assert_allclose(data_out, expected_out)


def test_dada_file_unpack_tfp() -> None:
    """Test that unpack calls the unpacker with given options."""
    file_data = np.random.randn(1000).astype(np.float32)

    file = DadaFile()
    file.set_data(file_data)

    options = MagicMock()
    options.ndim = 2
    unpacker = MagicMock()

    file.unpack_tfp(unpacker=unpacker, options=options)

    cast(MagicMock, unpacker.unpack).assert_called_once_with(data=file_data.tobytes(), options=options)


def test_dada_file_unpack_tfp_with_invalid_ndim() -> None:
    """Test that unpack_tfp calls the unpacker with given options."""
    file_data = np.random.randn(1000).astype(np.float32)

    file = DadaFile()
    file.set_data(file_data)

    ndim = np.random.randint(3, 100)
    options = MagicMock()
    options.ndim = ndim
    unpacker = MagicMock()

    with pytest.raises(
        AssertionError,
        match=f"unpack_tfp currently supports only real or complex valued data. options.ndim={ndim}",
    ):
        file.unpack_tfp(unpacker=unpacker, options=options)


def test_dada_file_unpack_tfp_with_no_options() -> None:
    """Test that unpack_tfp provides correct default unpack options from AsciiHeader."""
    nbit = 8
    npol = 1
    nchan = 3
    ndim = 2

    file_data = np.random.randn(1000).astype(np.float32)

    file = DadaFile()
    file.set_header_value("NBIT", nbit)
    file.set_header_value("NDIM", ndim)
    file.set_header_value("NCHAN", nchan)
    file.set_header_value("NPOL", npol)
    file.set_data(file_data)

    options = UnpackOptions(nbit=nbit, nchan=nchan, npol=npol, ndim=ndim)
    unpacker = MagicMock()

    file.unpack_tfp(unpacker=unpacker)

    cast(MagicMock, unpacker.unpack).assert_called_once_with(data=file_data.tobytes(), options=options)
