"""
Units tests for `scores.probability.pit`.
"""

import numpy as np
import pytest
import xarray as xr

from scores.probability.pit import (
    _check_args_with_pits,
    _pit_no_ptmass,
    _pit_ptmass,
    pit,
    pit_histogram_values,
    pit_scores,
)
from tests.probabilty import pit_test_data as ptd


def test__pit_ptmass():
    """Tests `_pit_ptmass`."""
    result = _pit_ptmass(ptd.DA_PIT_VALUES, ptd.ARRAY_PIT_THRESH)
    xr.testing.assert_allclose(result, ptd.EXP_PIT_PTMASSS)


def test__pit_no_ptmass():
    """Tests `_pit_no_ptmass`."""
    result = _pit_no_ptmass(ptd.DA_PIT_VALUES, ptd.ARRAY_PIT_THRESH)
    xr.testing.assert_allclose(result, ptd.EXP_PIT_NOPTMASS)


def test_pit():
    """Tests `pit`."""
    result = pit(
        ptd.DA_FCST_CDF,
        ptd.DA_OBS,
        "x",
        possible_pointmass=2,
        included_pit_thresholds=ptd.ARRAY_PIT_THRESH,
        pit_precision=0.05,
    )

    xr.testing.assert_allclose(result, ptd.EXP_PIT)


@pytest.mark.parametrize(
    ("fcst_cdf", "obs", "fcst_threshold_dim", "pit_precision", "error_msg_snippet"),
    [
        (
            ptd.DA_FCST_CDF,
            ptd.DA_OBS,
            "y",
            0,
            "`fcst_threshold_dim` is not a dimension of `fcst_cdf`",
        ),
        (
            ptd.DA_FCST_CDF2,
            ptd.DA_OBS,
            "x",
            0,
            "values of `fcst_cdf` must be in the closed",
        ),
        (
            ptd.DA_FCST_CDF3,
            ptd.DA_OBS,
            "x",
            0,
            "coordinates along `fcst_threshold_dim` of `fcst_cdf` must be increasing",
        ),
        (ptd.DA_FCST_CDF, ptd.DA_OBS2, "x", 0, "'x' is a dimension of `obs`"),
        (
            ptd.DA_FCST_CDF,
            ptd.DA_OBS3,
            "x",
            0,
            "Dimensions of `obs` must be a subset of dimensions of `fcst_cdf`",
        ),
        (ptd.DA_FCST_CDF, ptd.DA_OBS, "x", -1.0, "`pit_precision` must be nonnegative"),
    ],
)
def test_pit_raises(fcst_cdf, obs, fcst_threshold_dim, pit_precision, error_msg_snippet):
    """Tests that `pit` raises as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        pit(fcst_cdf, obs, fcst_threshold_dim, pit_precision=pit_precision)


@pytest.mark.parametrize(
    ("pit_cdf", "n_bins", "expected"),
    [
        (ptd.DA_PIT_CDF1, 4, ptd.EXP_PIT_HIST1A),
        (ptd.DA_PIT_CDF1, 2, ptd.EXP_PIT_HIST1B),
        (ptd.DA_PIT_CDF2, 4, ptd.EXP_PIT_HIST2B),
    ],
)
def test_pit_histogram_values(pit_cdf, n_bins, expected):
    """Tests `pit_histogram_values` with a variety of imputs."""
    result = pit_histogram_values(pit_cdf, pit_threshold_dim="pit_thresh", n_bins=n_bins)
    xr.testing.assert_allclose(result, expected)


def test_pit_histogram_values2():
    """
    Tests `pit_histogram_values`. Uses `xr.testing.assert_allclose`
    fails due to tiny floating point arithmetic differences in calculated coordinates.
    """
    result = pit_histogram_values(ptd.DA_PIT_CDF2, pit_threshold_dim="pit_thresh", n_bins=5, dims=["station"])
    np.testing.assert_allclose(result, ptd.EXP_PIT_HIST2A)


@pytest.mark.parametrize(
    ("pit_cdf", "pit_threshold_dim", "dims", "error_msg_snippet"),
    [
        (ptd.DA_PIT_CDF2, "x", None, "`pit_threshold_dim` is not a dimension"),
        (
            ptd.DA_PIT_CDF2,
            "pit_thresh",
            ["pit_thresh"],
            "`pit_threshold_dim` is in `dims`",
        ),
        (ptd.DA_PIT_CDF2, "pit_thresh", ["zzz"], "`dims` is not a subset of"),
        (ptd.DA_FCST_CDF2, "x", None, "values of `pit_cdf` must be in the closed"),
        (
            ptd.DA_PIT_CDF3,
            "pit_thresh",
            None,
            "`pit_cdf` is not 0 when `pit_cdf\\[pit_threshold_dim\\] < 0`",
        ),
        (
            ptd.DA_PIT_CDF4,
            "pit_thresh",
            None,
            "`pit_cdf` is not 1 when `pit_cdf\\[pit_threshold_dim\\] > 1`",
        ),
    ],
)
def test__check_args_with_pits(pit_cdf, pit_threshold_dim, dims, error_msg_snippet):
    """Tests that `_check_args_with_pits` raises as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        _check_args_with_pits(pit_cdf, pit_threshold_dim, dims)


def test_pit_histogram_raises():
    """
    Tests that `pit_histogram_values` raises as expected. Note that all but
    one raises tests for `pit_histogram_values` are performed by `test__check_args_with_pits`.
    """
    with pytest.raises(ValueError, match="`n_bins` must be at least 1"):
        pit_histogram_values(ptd.DA_PIT_CDF2, pit_threshold_dim="pit_thresh", n_bins=-10, dims=None)


@pytest.mark.parametrize(
    ("pit_cdf", "dims", "expected"),
    [
        (ptd.DA_PIT_SCORE1, None, ptd.EXP_PIT_SCORE1A),
        (ptd.DA_PIT_SCORE1, ["station", "date"], ptd.EXP_PIT_SCORE1B),
        (ptd.DA_PIT_SCORE2, ["station"], ptd.EXP_PIT_SCORE2),
    ],
)
def test_pit_scores(pit_cdf, dims, expected):
    """Tests `pit_scores`"""
    result = pit_scores(pit_cdf, "pit_thresh", dims)
    xr.testing.assert_allclose(result, expected)
