# -*- coding: utf-8 -*-
#
# This file is part of the SKA PyDADA project.
#
# Distributed under the terms of the BSD 3-clause new license.
# See LICENSE.txt for more info.

"""This file contains unit tests for ska_pydada.utils.pfb_analysis.py."""

import pathlib

import pytest

from ska_pydada import DadaFile
from ska_pydada.utils.pfb_analysis import analyse_pfb_temporal_fidelity


@pytest.fixture
def temporal_fidelity_pass_file(data_path: pathlib.Path) -> pathlib.Path:
    """Get the test file that should pass temporal fidelity analysis."""
    return data_path / "temporal_fidelity_pass.dada"


@pytest.fixture
def temporal_fidelity_fail(data_path: pathlib.Path) -> pathlib.Path:
    """Get the test file that should fail temporal fidelity analysis."""
    return data_path / "temporal_fidelity_fail.dada"


def test_analyse_pfb_temporal_fidelity_assertions() -> None:
    """Test the assertions when calling analyse_pfb_temporal_fidelity."""
    with pytest.raises(AssertionError, match="either num_impulses and/or expected_impulses must be set"):
        analyse_pfb_temporal_fidelity(
            "anyfile.dada",
            env_slope_db=-4.0,
            env_halfwidth_us=15.0,
            nifft=196608,
            max_db_outside_env=-60.0,
            max_spurious_power_db=-50.0,
        )

    with pytest.raises(
        AssertionError, match=r"expected len\(expected_impulses\)=2 to be equal to num_impulses=1"
    ):
        analyse_pfb_temporal_fidelity(
            "anyfile.dada",
            env_slope_db=-4.0,
            env_halfwidth_us=15.0,
            nifft=196608,
            max_db_outside_env=-60.0,
            max_spurious_power_db=-50.0,
            num_impulses=1,
            expected_impulses=[1, 2],
        )

    with pytest.raises(AssertionError, match="expected at least 1 impulse to analyse"):
        analyse_pfb_temporal_fidelity(
            "anyfile.dada",
            env_slope_db=-4.0,
            env_halfwidth_us=15.0,
            max_db_outside_env=-60.0,
            max_spurious_power_db=-50.0,
            nifft=196608,
            num_impulses=0,
        )

    with pytest.raises(AssertionError, match="expected at least 1 impulse to analyse"):
        analyse_pfb_temporal_fidelity(
            "anyfile.dada",
            env_slope_db=-4.0,
            env_halfwidth_us=15.0,
            nifft=196608,
            max_db_outside_env=-60.0,
            max_spurious_power_db=-50.0,
            expected_impulses=[],
        )

    with pytest.raises(AssertionError, match="expected anyfile.dada to exists and be a file not a directory"):
        analyse_pfb_temporal_fidelity(
            "anyfile.dada",
            env_slope_db=-4.0,
            env_halfwidth_us=15.0,
            nifft=196608,
            max_db_outside_env=-60.0,
            max_spurious_power_db=-50.0,
            num_impulses=1,
            expected_impulses=[1],
        )


def test_analyse_pfb_temporal_fidelity_passes(temporal_fidelity_pass_file: pathlib.Path) -> None:
    """Test that a valid PFB inversion passes analysis."""
    expected_impulses = [1536, 173664]

    result = analyse_pfb_temporal_fidelity(
        temporal_fidelity_pass_file,
        env_slope_db=-4.0,
        env_halfwidth_us=15.0,
        nifft=196608,
        max_db_outside_env=-60.0,
        max_spurious_power_db=-50.0,
        num_impulses=2,
    )

    dada_file = DadaFile.load_from_file(temporal_fidelity_pass_file)

    assert result.overall_result, "expected overall result to be True"
    assert dada_file.get_header_float("TSAMP") == result.tsamp
    assert len(result.impulse_results) == 2

    for expected_impulse_idx, r in zip(expected_impulses, result.impulse_results):
        assert r.expected_impulse_idx == expected_impulse_idx
        assert r.impulse_idx == expected_impulse_idx
        assert r.valid_impulse_position
        assert r.max_power_result
        assert len(r.max_power_result_idx) == 0
        assert r.total_spurious_power_result
        assert r.total_spurious_power_db <= -50.0


def test_analyse_pfb_temporal_fidelity_fails_when_expected_impulses_are_incorrect(
    temporal_fidelity_pass_file: pathlib.Path,
) -> None:
    """Test that a valid PFB inversion passes analysis."""
    expected_impulses = [1537, 173663]
    actual_impulses = [1536, 173664]

    result = analyse_pfb_temporal_fidelity(
        temporal_fidelity_pass_file,
        env_slope_db=-4.0,
        env_halfwidth_us=15.0,
        nifft=196608,
        max_db_outside_env=-60.0,
        max_spurious_power_db=-50.0,
        num_impulses=2,
        expected_impulses=[1537, 173663],
    )

    dada_file = DadaFile.load_from_file(temporal_fidelity_pass_file)

    assert not result.overall_result
    assert dada_file.get_header_float("TSAMP") == result.tsamp
    assert len(result.impulse_results) == 2

    for expected_impulse_idx, actual_impulse_idx, r in zip(
        expected_impulses, actual_impulses, result.impulse_results
    ):
        assert r.expected_impulse_idx == expected_impulse_idx
        assert r.impulse_idx == actual_impulse_idx
        assert not r.valid_impulse_position


def test_analyse_pfb_temporal_fidelity_fails(temporal_fidelity_fail: pathlib.Path) -> None:
    """Test that a valid PFB inversion fails analysis for incorrect PFB."""
    expected_impulses = [1280, 173408]

    result = analyse_pfb_temporal_fidelity(
        temporal_fidelity_fail,
        env_slope_db=-4.0,
        env_halfwidth_us=15.0,
        max_db_outside_env=-60.0,
        max_spurious_power_db=-50.0,
        nifft=196608,
        num_impulses=2,
    )

    dada_file = DadaFile.load_from_file(temporal_fidelity_fail)

    assert not result.overall_result
    assert dada_file.get_header_float("TSAMP") == result.tsamp
    assert len(result.impulse_results) == 2

    for expected_impulse_idx, r in zip(expected_impulses, result.impulse_results):
        assert r.expected_impulse_idx == expected_impulse_idx
        assert r.impulse_idx == expected_impulse_idx
        assert r.valid_impulse_position
        assert not r.max_power_result
        assert len(r.max_power_result_idx) > 0
        assert not r.total_spurious_power_result
        assert r.total_spurious_power_db > -50.0
