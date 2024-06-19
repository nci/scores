"""
This module contains unit tests for scores.processing.cdf
"""

import numpy as np
import pytest
import xarray as xr

from scores.processing.cdf import (
    add_thresholds,
    cdf_envelope,
    decreasing_cdfs,
    fill_cdf,
    integrate_square_piecewise_linear,
    observed_cdf,
    propagate_nan,
    round_values,
)
from tests.probabilty import cdf_test_data, crps_test_data
from tests.processing.cdf import functions_test_data as ftd


def test_round_values_exception():
    """Test rounding throws the right exceptions"""
    with pytest.raises(ValueError):
        round_values(xr.DataArray(), -1, final_round_decpl=5)


@pytest.mark.parametrize(
    ("array", "rounding_precision", "expected"),
    [
        (ftd.DA_ROUND, 0, ftd.EXP_ROUND1),
        (ftd.DA_ROUND, 0.2, ftd.EXP_ROUND2),
        (ftd.DA_ROUND, 5, ftd.EXP_ROUND3),
    ],
)
def test_round_values(array, rounding_precision, expected):
    """Tests `round_values` with a variety of inputs."""
    output_as = round_values(array, rounding_precision)
    xr.testing.assert_allclose(output_as, expected)


def test_propagate_nan_error():
    """Test propagating throws the right exceptions"""
    faulty_array = xr.Dataset({"lertitude": [1, 2, np.nan, 4], "longitude": [20, 21, 22, 23, 24]})
    with pytest.raises(ValueError):
        propagate_nan(faulty_array, "latitude")


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        ("x", cdf_test_data.EXP_PROPNAN_X),
        ("y", cdf_test_data.EXP_PROPNAN_Y),
    ],
)
def test_propagate_nan(dim, expected):
    """Tests `propagate_nan` with a variety of inputs."""
    result = propagate_nan(cdf_test_data.DA_PROPNAN, dim)
    xr.testing.assert_allclose(result, expected)


def test_observed_cdf_errors():
    """Test `osbserved_cdf_errors` with a variety of inputs."""
    obs = xr.DataArray.from_dict({"dims": "temp", "data": [np.nan, np.nan, np.nan, np.nan]})
    threshold_dim = "irrelevant for test"
    threshold_values = None
    badprecision = -1

    # Bad precision raises a value error
    with pytest.raises(ValueError):
        observed_cdf(obs, threshold_dim, threshold_values=threshold_values, precision=badprecision)

    # Null obs and a null threshold value raises a value error
    with pytest.raises(ValueError):
        observed_cdf(obs, threshold_dim, threshold_values=threshold_values)


@pytest.mark.parametrize(
    ("function_values", "expected"),
    [
        (crps_test_data.DA_ISPL1, crps_test_data.EXP_ISPL1),
        (crps_test_data.DA_ISPL2, crps_test_data.EXP_ISPL2),
        (crps_test_data.DA_ISPL3, crps_test_data.EXP_ISPL3),
    ],
)
def test_integrate_square_piecewise_linear(function_values, expected):
    """Tests `integrate_square_piecewise_linear` with a variety of inputs."""
    result = integrate_square_piecewise_linear(function_values, "x")
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("method", "min_nonnan", "expected"),
    [
        ("linear", 2, cdf_test_data.EXP_FILL_CDF1A),
        ("step", 1, cdf_test_data.EXP_FILL_CDF1B),
        ("step", 2, cdf_test_data.EXP_FILL_CDF1C),
        ("forward", 1, cdf_test_data.EXP_FILL_CDF1D),
        ("backward", 1, cdf_test_data.EXP_FILL_CDF1E),
    ],
)
def test_fill_cdf(method, min_nonnan, expected):
    """Tests `fill_cdf` with a variety of inputs."""
    output = fill_cdf(cdf_test_data.DA_FILL_CDF1, "x", method, min_nonnan)
    xr.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    (
        "cdf",
        "threshold_dim",
        "method",
        "min_nonnan",
        "error_class",
        "error_msg_snippet",
    ),
    [
        (  # cdf values outside [0,1]
            cdf_test_data.DA_CDF_FROM_QUANTS4,
            "q_level",
            "linear",
            2,
            ValueError,
            "Input CDF has some values less than 0 or greater than 1.",
        ),
        (
            cdf_test_data.DA_FILL_CDF1,
            "y",
            "linear",
            2,
            ValueError,
            "'y' is not a dimension of `cdf`",
        ),
        (
            cdf_test_data.DA_FILL_CDF1,
            "x",
            "linear",
            1,
            ValueError,
            "`min_nonnan` must be at least 2 when `method='linear'`",
        ),
        (
            cdf_test_data.DA_FILL_CDF1,
            "x",
            "step",
            0,
            ValueError,
            "`min_nonnan` must be at least 1 when `method='step'`",
        ),
        (
            cdf_test_data.DA_FILL_CDF1,
            "x",
            "quad",
            0,
            ValueError,
            "`method` must be 'linear', 'step', 'forward' or 'backward'",
        ),
    ],
)
# pylint: disable=too-many-arguments
def test_fill_cdf_raises(cdf, threshold_dim, method, min_nonnan, error_class, error_msg_snippet):
    """`fill_cdf` raises an exception as expected."""
    with pytest.raises(error_class, match=error_msg_snippet):
        fill_cdf(
            cdf,
            threshold_dim,
            method,
            min_nonnan,
        )


@pytest.mark.parametrize(
    ("cdf"),
    [
        (cdf_test_data.DA_CDF_ENVELOPE1),
        (cdf_test_data.DA_CDF_ENVELOPE2),
    ],
)
def test_cdf_envelope(cdf):
    """Tests `cdf_envelope` with a variety of inputs."""
    result = cdf_envelope(cdf, "x")
    xr.testing.assert_allclose(result, cdf_test_data.EXP_CDF_ENVELOPE1)


def test_cdf_envelope_raises():
    """Tests that `cdf_envelope` raises the correct error."""
    with pytest.raises(ValueError, match="'y' is not a dimension of `cdf`"):
        cdf_envelope(cdf_test_data.DA_CDF_ENVELOPE1, "y")


@pytest.mark.parametrize(
    ("fill_method", "expected"),
    [
        ("linear", ftd.EXP_ADD_THRESHOLDS1),
        ("none", ftd.EXP_ADD_THRESHOLDS2),
    ],
)
def test_add_thresholds(fill_method, expected):
    """Tests `add_thresholds` with a variety of inputs."""
    result = add_thresholds(ftd.DA_ADD_THRESHOLDS, "x", [0.5, 0.2, 0.75, np.nan], fill_method)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("cdf", "tolerance", "expected"),
    [
        (ftd.DA_DECREASING_CDFS1, 0, ftd.EXP_DECREASING_CDFS1A),
        (ftd.DA_DECREASING_CDFS1, 0.3, ftd.EXP_DECREASING_CDFS1B),
    ],
)
def test_decreasing_cdfs(cdf, tolerance, expected):
    """Tests `decreasing_cdfs` with a variety of inputs."""
    result = decreasing_cdfs(cdf, "x", tolerance)
    xr.testing.assert_equal(result, expected)
