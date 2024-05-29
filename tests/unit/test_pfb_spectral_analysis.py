# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file contains unit tests for ska_pydada.utils.pfb_analysis.py."""

import pathlib
from typing import List

import pytest

from ska_pydada import DadaFile
from ska_pydada.utils.pfb_analysis import analyse_pfb_spectral_fidelity


@pytest.fixture
def expected_impulses() -> List[int]:
    """Get a list of channel indices of where impulses are expected."""
    return [82944, 165240, 82296]


@pytest.fixture
def spectral_fidelity_pass_file(data_path: pathlib.Path) -> pathlib.Path:
    """Get the test file that should pass spectral fidelity analysis."""
    return data_path / "dada" / "inverted_spectral_pass.dada"


@pytest.fixture
def spectral_fidelity_fail_file(data_path: pathlib.Path) -> pathlib.Path:
    """Get the test file that should fail spectral fidelity analysis."""
    return data_path / "dada" / "inverted_spectral_fail.dada"


def test_analyse_pfb_spectral_fidelity_assertions(spectral_fidelity_pass_file: pathlib.Path) -> None:
    """Test the assertions when calling analyse_pfb_spectral_fidelity."""
    with pytest.raises(AssertionError, match="expected at least 1 impulse to analyse"):
        analyse_pfb_spectral_fidelity(
            "anyfile.dada",
            t_test=290304,
            t_ifft=165888,
            max_power_db=-60.0,
            max_total_spectral_confusion_power_db=-50.0,
            expected_impulses=[],
        )

    with pytest.raises(AssertionError, match="expected anyfile.dada to exists and be a file not a directory"):
        analyse_pfb_spectral_fidelity(
            "anyfile.dada",
            t_test=290304,
            t_ifft=165888,
            max_power_db=-60.0,
            max_total_spectral_confusion_power_db=-50.0,
            expected_impulses=[82944],
        )

    with pytest.raises(
        AssertionError, match="expected test 5 end index to be less than or equal the length of data"
    ):
        analyse_pfb_spectral_fidelity(
            spectral_fidelity_pass_file,
            t_test=290304,
            t_ifft=165888,
            max_power_db=-60.0,
            max_total_spectral_confusion_power_db=-50.0,
            expected_impulses=[(5, 82944)],
        )


@pytest.mark.skip("Currently PFB inversion is failing spectral fidelity")
def test_analyse_pfb_spectral_fidelity_passes(
    spectral_fidelity_fail_file: pathlib.Path,
    expected_impulses: List[int],
) -> None:
    """Test that a valid PFB inversion passes analysis."""
    result = analyse_pfb_spectral_fidelity(
        file=spectral_fidelity_fail_file,
        t_test=290304,
        t_ifft=165888,
        max_power_db=-60.0,
        max_total_spectral_confusion_power_db=-50.0,
        expected_impulses=expected_impulses,
    )

    dada_file = DadaFile.load_from_file(spectral_fidelity_fail_file)

    assert result.overall_result, "expected overall result to be True"

    assert dada_file.get_header_float("TSAMP") == result.tsamp
    assert len(result.impulse_results) == len(expected_impulses)

    for expected_frequency_bin_idx, r in zip(expected_impulses, result.impulse_results):
        assert r.expected_frequency_bin_idx == expected_frequency_bin_idx
        assert r.frequency_bin_idx == expected_frequency_bin_idx
        assert r.valid_frequency_bin
        assert r.max_spectral_confusion_result
        assert len(r.max_spectral_confusion_result_idx) == 0
        assert r.total_spectral_confusion_power_result
        assert r.total_spectral_confusion_power_db <= -50.0


def test_analyse_pfb_spectral_fidelity_fails(
    spectral_fidelity_fail_file: pathlib.Path,
    expected_impulses: List[int],
) -> None:
    """Test that a valid PFB inversion passes analysis."""
    result = analyse_pfb_spectral_fidelity(
        file=spectral_fidelity_fail_file,
        t_test=290304,
        t_ifft=165888,
        max_power_db=-60.0,
        max_total_spectral_confusion_power_db=-50.0,
        expected_impulses=expected_impulses,
    )

    dada_file = DadaFile.load_from_file(spectral_fidelity_fail_file)

    assert not result.overall_result, "expected file to fail spectral fidelity analysis"

    assert dada_file.get_header_float("TSAMP") == result.tsamp
    assert len(result.impulse_results) == len(expected_impulses)

    for expected_frequency_bin_idx, r in zip(expected_impulses, result.impulse_results):
        assert r.expected_frequency_bin_idx == expected_frequency_bin_idx
        assert r.frequency_bin_idx == expected_frequency_bin_idx
        assert r.valid_frequency_bin
        assert not r.max_spectral_confusion_result
        assert len(r.max_spectral_confusion_result_idx) > 0
        assert not r.total_spectral_confusion_power_result
        assert r.total_spectral_confusion_power_db > -50.0
