# -*- coding: utf-8 -*-
#
# This file is part of the SKA PYDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""Utility submodule for checking polyphase filter-bank (PFB) temporal fidelity."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import List, Tuple

import numpy as np

from ska_pydada import DadaFile

from .common import POWER_NEG_100_DB, power_as_db


@dataclasses.dataclass(kw_only=True, frozen=True)
class TemporalFidelityImpulseResult:
    """A data class representing the result of checking an impulse against the expected fidelity."""

    impulse_idx: int
    """The timestamp index of where the impulse is in the file."""

    expected_impulse_idx: int
    """The expected timestamp index of where the impulse is in the file."""

    valid_impulse_position: bool
    """An indicator of whether the ``impulse_idx`` equals ``expected_impulse_idx``."""

    signal_power: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the relative signal power used to calculated results."""

    signal_power_db: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the relative signal power, in decibels, used to calculated results."""

    expected_max_power: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the maximum relative signal power based on parameters provided."""

    expected_max_power_db: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of the maximum relative signal power, in decibels, based on parameters provided."""

    max_power_result: bool
    """An indicator of whether all of the ``signal_power_db`` is <= to ``expected_max_power_db``."""

    max_power_result_idx: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of indices in the ``signal_power_db`` that exceeds the ``expected_max_power_db``."""

    total_spurious_power_result: bool
    """An indicator of whether the ``total_spurious_power_db`` value is greater than expected."""

    total_spurious_power_db: float
    """The total amount of spurious power, in decibels."""

    overall_result: bool
    """
    An indicator of the overall result for the impulse.

    This is equivalent of the following:

    .. code-block:: python

        valid_impulse_position and max_power_result and total_spurious_power_result

    """

    data_mask: np.ndarray = dataclasses.field(repr=False)
    """A Numpy array of indices for which the power for this impulse is checked against."""

    spurious_power_mask: np.ndarray = dataclasses.field(repr=False)
    """
    A Numpy array of indicies for the signal that is outside the main envelope.

    This is used for calulating the spurious power of the impulse.
    """


@dataclasses.dataclass(kw_only=True, frozen=True)
class TemporalFidelityResult:
    """A dataclass to aggregate the overall result of checking the temporal fidelity of the PFB."""

    tsamp: float
    """
    The time, in microseconds, per sample.

    This comes from the ``TSAMP`` header within a DADA file.
    """

    overall_result: bool
    """
    The overall result of the analysis of all impulses.

    If any result for any impulse is ``False`` then this value is ``False``, that is all impulses
    must be valid for the overall result to be valid.
    """

    impulse_results: List[TemporalFidelityImpulseResult]
    """A list of results for each individual impulse found in the file."""


def generate_expected_max_power(
    *,
    env_slope_db: float,
    env_halfwidth_us: float,
    nifft: int,
    impulse_idx: int,
    tsamp: float,
    nsamp: int,
    max_db_outside_env: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
    """Generate the expected maximum relative power to use in temporal PFB fidelity.

    The method generates a Numpy array that contains the maximium power, in dB, that
    will be used in temporal analysis.  It also returns an index array that is used
    to select the time samples that are within the bounds to be analyised as well
    as an index array of time samples that should be analyised but are not within
    the envelope (i.e. the value in the max power array where ``dB == max_db_outside_env``)

    :param env_slope_db: the slope, in dB / µsec, that is allowed around the impulse.
    :type env_slope_db: float
    :param env_halfwidth_us: the temporal halfwidth around the impulse, in microseconds.
    :type env_halfwidth_us: float
    :param nifft: the number of elements used in the inverse fast Fourier transform.
    :type nifft: int
    :param impulse_idx: the index within the data of where the impulse should be.
    :type impulse_idx: int
    :param tsamp: the time, in microseconds, per sample.
    :type tsamp: float
    :param nsamp: the total number of samples within the data file.
    :type nsamp: int
    :param max_db_outside_env: the maximum power, in decibels, allowed outside of the
        envelope.
    :type max_db_outside_env: float
    :return: a tuple of 3 Numpy arrays.  The first is the maximum expected power,
        the second is an index array to select records, and the 3rd is an index mask
        for records that should be analysed but not within the envelope.
    :rtype: Tuple[np.array, np.array, np.array]
    """
    idx = np.arange(0, nsamp)
    idx_dist_from_impulse = np.abs(idx - impulse_idx)
    delta_t = idx_dist_from_impulse * tsamp
    envelope_mask = np.where(delta_t <= env_halfwidth_us)

    power_db = max_db_outside_env * np.ones(shape=nsamp, dtype=np.float32)
    nifft_mask = np.where(idx_dist_from_impulse <= nifft / 2)
    outside_env_mask = np.logical_and(idx_dist_from_impulse <= nifft / 2, delta_t > env_halfwidth_us)

    power_db[envelope_mask] = env_slope_db * delta_t[envelope_mask]

    power = np.power(10.0, power_db / 10.0)
    return power, nifft_mask, outside_env_mask


def analyse_pfb_temporal_fidelity(
    file: str | pathlib.Path,
    # Ideally this should in in HEADER of output file
    env_slope_db: float,
    env_halfwidth_us: float,
    max_db_outside_env: float,
    max_spurious_power_db: float,
    nifft: int,
    num_impulses: int | None = None,
    expected_impulses: List[int] | None = None,
) -> TemporalFidelityResult:
    """Analyse PFB output for temporal fidelity.

    This method is used to analyse a DADA file that has data stored in
    temporal, frequency, polarisation format (TFP).

    Current limitations of this method are:

        * Assumes that NCHAN = 1
        * Assumes that NPOL = 1

    :param file: the location of the file to load and analyse.
    :type file: str | pathlib.Path
    :param env_slope_db: the slope, in dB / µsec, that is allowed around the impulse.
    :type env_slope_db: float
    :param env_halfwidth_us: the temporal halfwidth around the impulse, in microseconds.
    :type env_halfwidth_us: float
    :param max_db_outside_env: the maximum amount of relative power, in decibels, allowed
        outside of the allowed envelope
    :type max_db_outside_env: float
    :param max_spurious_power_db: the maximum total spurious power integrated outside of the
        allowed envelope, in decibels.
    :type max_spurious_power_db: float
    :param nifft: the number of elements used in the inverse fast Fourier transform.
    :type nifft: int
    :param num_impulses: the number of impulses to analyse, defaults to None. If not
        set this value is the length of ``expected_impulses``. Either this and/or
        ``expected_impulses`` must be set.
    :type num_impulses: int | None, optional
    :param expected_impulses: a list of zero offset indices that the expected pulse is at,
        defaults to None.  If None then the method will find the ``num_impulses`` highest
        values of power. Either this and/or ``num_impulses`` must be set.
    :type expected_impulses: List[int] | None, optional
    :raises AssertionError: either ``num_impulses`` and/or ``expected_impulses`` are set
        and the values are consistent with each other. Also the effective value of
        ``num_impulses`` must be greater than 0.
    :return: the results of analysing the PFB inversion fidelity.
    :rtype: TemporalFidelityResult
    """
    if num_impulses is None:
        assert expected_impulses is not None, "either num_impulses and/or expected_impulses must be set"
        num_impulses = len(expected_impulses)
    elif expected_impulses is not None:
        assert (
            len(expected_impulses) == num_impulses
        ), f"expected {len(expected_impulses)=} to be equal to {num_impulses=}"

    assert num_impulses > 0, "expected at least 1 impulse to analyse"

    file = pathlib.Path(file)
    assert file.exists() and file.is_file(), f"expected {file} to exists and be a file not a directory"

    dada_file = DadaFile.load_from_file(file=file, chunk_size=-1)

    tfp_voltage_data = dada_file.as_time_freq_pol()

    # at the moment we only handle NCHAN = 1, NPOL = 1
    tfp_voltage_data = tfp_voltage_data.flatten()

    tfp_power = np.abs(tfp_voltage_data) ** 2.0

    # need to get the indexes of the impulse
    impulse_indices = np.argpartition(tfp_power, -num_impulses)[-num_impulses:]  # pylint: disable=E1130
    impulse_indices = np.sort(impulse_indices)

    if expected_impulses is None:
        expected_impulses = [*impulse_indices]

    tsamp = dada_file.get_header_float("TSAMP")

    def calc_results(impulse_idx: int, nth_pulse: int) -> TemporalFidelityImpulseResult:
        (expected_max_power, data_mask, spurious_power_mask) = generate_expected_max_power(
            env_slope_db=env_slope_db,
            env_halfwidth_us=env_halfwidth_us,
            nifft=nifft,
            impulse_idx=impulse_idx,
            tsamp=tsamp,
            nsamp=len(tfp_voltage_data),
            max_db_outside_env=max_db_outside_env,
        )

        expected_impulse_idx = expected_impulses[nth_pulse]
        valid_impulse_position = expected_impulse_idx == impulse_idx

        expected_max_power = expected_max_power[data_mask]
        expected_max_power_db = power_as_db(expected_max_power)

        signal_power_full = tfp_power / tfp_power[impulse_idx]
        signal_power = signal_power_full[data_mask]

        # ensure we don't end up with a Numpy warning of divide by zero
        signal_power[signal_power < POWER_NEG_100_DB] = POWER_NEG_100_DB
        signal_power_db = power_as_db(signal_power)

        # the IDX is a tuple
        max_power_result_idx = np.where(signal_power > expected_max_power)[0]

        # True if valid, False if not
        max_power_result = not np.any(max_power_result_idx)

        total_spurious_power = np.sum(signal_power_full[spurious_power_mask])
        total_spurious_power_db = float(power_as_db(total_spurious_power))
        total_spurious_power_result = total_spurious_power_db <= max_spurious_power_db

        return TemporalFidelityImpulseResult(
            impulse_idx=impulse_idx,
            expected_impulse_idx=expected_impulse_idx,
            valid_impulse_position=valid_impulse_position,
            signal_power=signal_power,
            signal_power_db=signal_power_db,
            expected_max_power=expected_max_power,
            expected_max_power_db=expected_max_power_db,
            max_power_result=max_power_result,
            max_power_result_idx=max_power_result_idx,
            total_spurious_power_result=total_spurious_power_result,
            total_spurious_power_db=total_spurious_power_db,
            overall_result=total_spurious_power_result and max_power_result and valid_impulse_position,
            data_mask=data_mask[0],
            spurious_power_mask=spurious_power_mask,
        )

    impulse_results = [
        calc_results(impulse_idx=impulse_idx, nth_pulse=i) for (i, impulse_idx) in enumerate(impulse_indices)
    ]
    overall_result = True
    for r in impulse_results:
        overall_result &= r.overall_result

    return TemporalFidelityResult(
        tsamp=tsamp,
        overall_result=overall_result,
        impulse_results=impulse_results,
    )
