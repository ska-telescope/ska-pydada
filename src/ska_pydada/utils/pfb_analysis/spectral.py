# -*- coding: utf-8 -*-
#
# This file is part of the SKA PYDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""Utility submodule for checking polyphase filter-bank (PFB) spectral fidelity."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import List, Sequence, Tuple

import numpy as np
from scipy.fft import fft, fftshift

from ska_pydada import DadaFile

from .common import POWER_NEG_100_DB, power_as_db

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SpectralFidelityToneResult:
    """
    A data class representing the result of checking a tone against the expected fidelity.

    All the power data reported in this class is from the frequency domain after applying a
    fast Fourier transform (FFT) on the input data.
    """

    time_sample_start_idx: int
    """
    The starting time bin index used for checking this tone.

    This is ``test tone number * T_TEST``
    """

    frequency_bin_idx: int
    """The frequency bin index of where the tone is found in the spectrum."""

    expected_frequency_bin_idx: int
    """The expected frequency bin index of where the tone is found in the spectrum."""

    valid_frequency_bin: bool
    """An indicator of whether the ``frequency_bin_idx`` equals ``expected_frequency_bin_idx``."""

    signal_power: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the relative signal power used to calculate results."""

    signal_power_db: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the relative signal power, in decibels, used to calculate results."""

    max_spectral_confusion_result: bool
    """An indicator of whether all of the ``signal_power_db`` is less than expected maximum."""

    max_spectral_confusion_result_idx: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of indices in the ``signal_power_db`` that exceeds the expected maximum."""

    total_spectral_confusion_power_result: bool
    """An indicator of whether the ``total_spectral_confusion_power_db`` value is greater than expected."""

    total_spectral_confusion_power_db: float
    """The total amount of spurious power, in decibels."""

    overall_result: bool
    """
    An indicator of the overall result for the tone.

    If the value is ``True`` then the test for the tone passes. If the value is ``False`` then
    the test failed.

    This is equivalent of the following:

    .. code-block:: python

        valid_frequency_bin and max_spectral_confusion_result and total_spectral_confusion_power_result

    """


@dataclasses.dataclass(kw_only=True, frozen=True)
class SpectralFidelityResult:
    """A dataclass to aggregate the overall result of checking the spectral fidelity of the PFB."""

    tsamp: float
    """
    The time, in microseconds, per sample.

    This comes from the ``TSAMP`` header within a DADA file.
    """

    overall_result: bool
    """
    The overall result of the analysis of all tones.

    If any result for any tone is ``False`` then this value is ``False``, that is all tones
    must be valid for the overall result to be valid.
    """

    tone_results: List[SpectralFidelityToneResult]
    """A list of results for each individual tone found in the file."""


def analyse_pfb_spectral_fidelity(
    file: str | pathlib.Path,
    t_test: int,
    t_ifft: int,
    expected_tones: Sequence[int | Tuple[int, int]],
    max_power_db: float = -60.0,
    max_total_spectral_confusion_power_db: float = -50.0,
) -> SpectralFidelityResult:
    """Analyse PFB output for spectral fidelity.

    This method is used to analyse a DADA file that has data stored in
    time, frequency, polarisation format (TFP).

    Current limitations of this method are:

        * Assumes that NCHAN = 1
        * Assumes that NPOL = 1

    The ``expected_tones`` can either be a list of integers which is the frequency
    channel bin the tone should be in or a list of tuples where the first value is
    the test sample block index and the second value is the frequency channel bin the
    tone should be in.

    The first form is simple and if there are multiple values this function will
    assume that the index in the list is the index of test case (using zero offset).

    The second form is used if trying to get a specific test that is not the first (``index = 0``)

    :param file: the location of the file to load and analyse.
    :type file: str | pathlib.Path
    :param t_test: the number of time samples that the test covers.
        This needs to be greater than ``t_ifft`` to avoid Gibbs effects in the expected pulses.
    :type t_test: int
    :param t_ifft: the number of elements in the inverse Fast Fourier Transform (FFT) performed during
        PFB inversion, which is equal to the number of elements in the forward FFTs performed
        on each test sample block.
    :type t_ifft: int
    :param expected_tones: a sequence of zero offset indices that the expected pulse is at,
        defaults to None.
    :type expected_tones: Sequence[int | Tuple[int, int]]
    :return: the results of analysing the PFB inversion fidelity.
    :rtype: SpectralFidelityResult
    """
    assert len(expected_tones) > 0, "expected at least 1 tone to analyse"

    file = pathlib.Path(file)
    assert file.exists() and file.is_file(), f"expected {file} to exist and be a file not a directory"

    dada_file = DadaFile.load_from_file(file=file, chunk_size=-1)

    tfp_voltage_data = dada_file.as_time_freq_pol()

    # at the moment we only handle NCHAN = 1, NPOL = 1
    tfp_voltage_data = tfp_voltage_data.flatten()

    if isinstance(expected_tones[0], int):
        _expected_tones: List[Tuple[int, int]] = list(enumerate(expected_tones))  # type: ignore
    else:
        _expected_tones: List[Tuple[int, int]] = expected_tones  # type: ignore

    def calc_results(test_idx: int, expected_frequency_bin_idx: int) -> SpectralFidelityToneResult:
        start_idx = test_idx * t_test
        end_idx = start_idx + t_ifft

        assert end_idx <= len(tfp_voltage_data), (
            f"expected test {test_idx} end index to be less than or equal the length of data"
            f" {len(tfp_voltage_data)}"
        )

        sample_data = tfp_voltage_data[start_idx:end_idx]
        fft_data = fftshift(fft(sample_data, n=t_ifft))

        fft_power = np.abs(fft_data) ** 2.0
        frequency_bin_idx = int(np.argmax(fft_power))

        logger.debug(f"{frequency_bin_idx=}")

        valid_frequency_bin = frequency_bin_idx == expected_frequency_bin_idx
        logger.debug(f"{frequency_bin_idx=}, {expected_frequency_bin_idx=}, {valid_frequency_bin=}")

        relative_fft_power = fft_power / fft_power[frequency_bin_idx]
        relative_fft_power[relative_fft_power < POWER_NEG_100_DB] = POWER_NEG_100_DB
        fft_db = power_as_db(relative_fft_power)

        expected_max_power = max_power_db * np.ones_like(relative_fft_power)
        expected_max_power[expected_frequency_bin_idx] = 1.0
        spectral_confusion_idx = np.where(fft_db > expected_max_power)[0]
        max_spectral_confusion_result = len(spectral_confusion_idx) == 0
        logger.debug(f"Max Spurious Power Idx (db): {spectral_confusion_idx}")

        total_spectral_confusion_power = (
            np.sum(relative_fft_power) - relative_fft_power[expected_frequency_bin_idx]
        )
        total_spectral_confusion_power_db = float(power_as_db(total_spectral_confusion_power))
        total_spectral_confusion_power_result = (
            total_spectral_confusion_power_db <= max_total_spectral_confusion_power_db
        )
        logger.debug(f"Total Spurious Power (db): {total_spectral_confusion_power_db}")

        overall_result = (
            valid_frequency_bin and max_spectral_confusion_result and total_spectral_confusion_power_result
        )

        return SpectralFidelityToneResult(
            time_sample_start_idx=start_idx,
            frequency_bin_idx=frequency_bin_idx,
            expected_frequency_bin_idx=expected_frequency_bin_idx,
            valid_frequency_bin=valid_frequency_bin,
            signal_power=relative_fft_power,
            signal_power_db=fft_db,
            max_spectral_confusion_result_idx=spectral_confusion_idx,
            max_spectral_confusion_result=max_spectral_confusion_result,
            total_spectral_confusion_power_db=total_spectral_confusion_power_db,
            total_spectral_confusion_power_result=total_spectral_confusion_power_result,
            overall_result=overall_result,
        )

    tone_results = [calc_results(test_idx, k_idx) for (test_idx, k_idx) in _expected_tones]
    overall_result = True
    for r in tone_results:
        overall_result &= r.overall_result

    tsamp = dada_file.get_header_float("TSAMP")
    return SpectralFidelityResult(
        tsamp=tsamp,
        overall_result=overall_result,
        tone_results=tone_results,
    )
