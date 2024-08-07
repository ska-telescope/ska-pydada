# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file contains unit tests for the SkaUnpacker in ska_pydada.unpacker.py."""


import numpy as np
import pytest

from ska_pydada import SKA_DIGI_SCALE_MEAN, SkaUnpacker, UnpackOptions
from ska_pydada.common import BITS_PER_BYTE

# fmt: off
SOURCE_DATA = np.array(
    [
        -12.77228, 12.6336634, -5.0050000, -1.4842639, -2.5428553,
        0.7414923, -2.1585503, 2.3708253, -0.40148783, -7.3821115,
        1.2647325, -2.0321102, 0.7521466, -0.5648981, 2.9565287,
        4.120244, 0.47453195, -3.2339506, 1.2360413, -1.4815402,
        -1.1341237, -4.9306426, -1.9723871, -3.630312, -2.237967,
        2.8785143, -6.6134176, 1.2712177, -2.2587833, 2.0703325,
        0.95531327, 1.2340846
    ],
    dtype=np.float32,
)
"""Random source data."""

NBIT_1_DATA = np.array([-94, -44, 5, -22], dtype=np.int8)

NBIT_2_DATA = np.array([-90, 102, -104, 93, -104, -85, 102, 86], dtype=np.int8)

NBIT_4_DATA = np.array([
    120, -72, 40, 121, -113, -92, -30, 119, -127, -76, -116,
    -118, 121, 72, 121, 67
], dtype=np.int8)

NBIT_8_DATA = np.array([
    -128, 127, -51, -15, -26, 7, -22, 24, -4, -75, 13, -21, 8,
    -6, 30, 42, 5, -33, 12, -15, -11, -50, -20, -37, -23, 29,
    -67, 13, -23, 21, 10, 12,
], dtype=np.int8)

NBIT_16_DATA = np.array([
    -14131, 13978, -5538, -1642, -2813, 820, -2388, 2623, -444,
    -8168, 1399, -2248, 832, -625, 3271, 4559, 525, -3578, 1368,
    -1639, -1255, -5455, -2182, -4017, -2476, 3185, -7317, 1406,
    -2499, 2291, 1057, 1365,
], dtype=np.int16)

NBIT_FLOAT_DATA = SOURCE_DATA
# fmt: on


def _pack_data(data: np.ndarray, nbit: int) -> np.ndarray:
    if nbit == -32:
        return data
    elif nbit == 1:
        int_data = np.ones(len(data), dtype=np.int8)
        int_data[np.signbit(data)] = 0
    else:
        (scale, mean) = SKA_DIGI_SCALE_MEAN[nbit]
        scaled_data = np.around(scale * data + mean).astype(np.int16)
        max_value = pow(2, nbit - 1) - 1
        min_value = -(max_value + 1)
        int_data = np.clip(scaled_data, min_value, max_value)

    if nbit == 8:
        return int_data.astype(np.int8)
    elif nbit == 16:
        return int_data

    # now just handling 1, 2 or 4 bit to be packed.  Assuming TFP
    mask = np.int8(pow(2, nbit) - 1)
    samples_per_byte = 8 // nbit
    out_data = np.zeros(len(data) // samples_per_byte, dtype=np.int8)

    for idx in range(len(data)):
        value: np.int8 = int_data[idx]
        out_idx = idx // samples_per_byte
        byte_sample_idx = idx % samples_per_byte
        out_data[out_idx] |= (value & mask) << (byte_sample_idx * nbit)

    return out_data


def _assert_statistics(
    population_mean: float,
    population_var: float,
    samples: np.ndarray,
    tolerance: float = 6.0,
) -> None:
    """Assert that sample mean and var are within a given tolerance of population stats."""
    N = len(samples)
    S = population_var
    mu = population_mean
    # This is the 4th moment of a gaussian distribution
    mu_4 = 3.0 * S**2
    E = np.mean(samples)
    V = np.var(samples, ddof=1)

    # expected variance in E
    var_e = S / N
    sigma_e = np.sqrt(var_e)

    # expected variance in V
    var_v = (mu_4 - (N - 3) / (N - 1) * S**2) / N
    sigma_v = np.sqrt(var_v)

    n_sigma_e = np.fabs(E - mu) / sigma_e
    n_sigma_v = np.fabs(V - S) / sigma_v

    assert n_sigma_e <= tolerance and n_sigma_v <= tolerance, (
        f"Expected sample mean ({E:0.6f}) and variance ({V:0.3f}) to be within {tolerance:0.3f} sigma"
        f" of {mu:0.6f} and {S:0.3f} respectively. n_sigma_e={n_sigma_e:0.3f}, "
        f"n_sigma_v={n_sigma_v:0.3f}"
    )


@pytest.mark.parametrize(
    "nbit,input_data",
    [
        (1, NBIT_1_DATA),
        (2, NBIT_2_DATA),
        (4, NBIT_4_DATA),
        (8, NBIT_8_DATA),
        (16, NBIT_16_DATA),
        (-32, NBIT_FLOAT_DATA),
    ],
)
def test_ska_unpacker_unpack_known_data(nbit: int, input_data: np.ndarray) -> None:
    """Test unpacking on known data."""
    ndim = 2
    nchan = 2
    npol = 2

    (scale, mean) = SKA_DIGI_SCALE_MEAN[nbit]
    if nbit == 1:
        source_data = np.ones(len(SOURCE_DATA), dtype=np.float32)
        source_data[np.signbit(SOURCE_DATA)] = 0.0
    elif nbit == -32:
        source_data = np.copy(SOURCE_DATA)
    else:
        source_data = np.copy(SOURCE_DATA)
        source_data = np.around(scale * SOURCE_DATA + mean).astype(np.float32)

        max_value = pow(2, nbit - 1) - 1
        min_value = -(max_value + 1)
        source_data = np.clip(source_data, min_value, max_value)

    expected_output_data = (source_data - mean) / scale

    options = UnpackOptions(nbit=nbit, nchan=nchan, npol=npol, ndim=ndim)
    unpacker = SkaUnpacker()

    output = unpacker.unpack(data=input_data.tobytes(), options=options)

    expected_output_data_cf64 = (
        expected_output_data.reshape((-1, nchan, npol, ndim)).view(np.complex64).squeeze(-1)
    )

    np.testing.assert_allclose(output, expected_output_data_cf64)


@pytest.mark.parametrize(
    "nbit,nchan,npol,ndim",
    [
        (1, 32, 2, 2),
        (1, 64, 1, 2),
        (1, 64, 2, 1),
        (1, 128, 1, 1),
        # NOTE - 2bit packing and unpacking fails assertions
        # (2, 32, 2, 2),
        # (2, 64, 1, 2),
        # (2, 64, 2, 1),
        # (2, 128, 1, 1),
        (4, 32, 2, 2),
        (4, 64, 1, 2),
        (4, 64, 2, 1),
        (4, 128, 1, 1),
        (8, 32, 2, 2),
        (8, 64, 1, 2),
        (8, 64, 2, 1),
        (8, 128, 1, 1),
        (16, 32, 2, 2),
        (16, 64, 1, 2),
        (16, 64, 2, 1),
        (16, 128, 1, 1),
        (-32, 32, 2, 2),
        (-32, 64, 1, 2),
        (-32, 64, 2, 1),
        (-32, 128, 1, 1),
    ],
)
def test_ska_unpacker_unpack_random_data(nbit: int, nchan: int, npol: int, ndim: int) -> None:
    """Test that the expected output has the expected mean and variance from random data."""
    source_data = np.random.randn(100 * nchan * npol * ndim).astype(np.float32)
    packed_data = _pack_data(data=source_data, nbit=nbit)

    options = UnpackOptions(nbit=nbit, ndim=ndim, nchan=nchan, npol=npol)
    unpacker = SkaUnpacker()

    unpacked_data = unpacker.unpack(data=packed_data.tobytes(), options=options).flatten()

    if ndim == 2:
        unpacked_data = unpacked_data.view(np.float32)

    _assert_statistics(population_mean=0.0, population_var=1.0, samples=unpacked_data)


def test_unpack_options_with_additional_arg() -> None:
    """Test that additional arguments can be accessed as attributes on a UnpackOptions instance."""
    additional_ags = {
        "luke": "vader",
        "arthur": 42,
    }

    options = UnpackOptions(nbit=1, nchan=2, npol=2, ndim=2, addition_args=additional_ags)

    assert options.luke == "vader"
    assert options.arthur == 42

    with pytest.raises(AttributeError, match="'UnpackOptions' object has no attribute 'bob'"):
        options.bob


@pytest.mark.parametrize(
    "nbit",
    [
        1,
        2,
        4,
        8,
        16,
        -32,
    ],
)
def test_unpack_when_data_not_resolution(nbit: int) -> None:
    """Test that unpack can handle data length not a multiple of the resolution."""
    nchan = 91
    npol = 3
    ndim = 2
    nbytes = 100001

    source_data = np.random.randint(0, 255, size=nbytes).astype(np.uint8)

    options = UnpackOptions(nbit=nbit, ndim=ndim, nchan=nchan, npol=npol)
    unpacker = SkaUnpacker()

    num_resolutions = 1
    resolution_bits = nchan * npol * ndim * abs(nbit)
    while (num_resolutions * resolution_bits) % BITS_PER_BYTE > 0:
        num_resolutions <<= 1

    effective_resolution = (num_resolutions * resolution_bits) // BITS_PER_BYTE
    expected_ndat = (nbytes // effective_resolution) * num_resolutions

    unpacked_data = unpacker.unpack(data=source_data.tobytes(), options=options)
    expected_shape = (expected_ndat, nchan, npol)

    assert unpacked_data.shape == expected_shape, f"expected {unpacked_data.shape=} to equal {expected_shape}"
