"""
This module contains unit tests for scores.probability.functions
"""
import numpy as np
import pytest
import xarray as xr

import scores.probability.functions
from tests import assertions
from tests.probabilty import cdf_test_data, crps_test_data


def test_round_values_exception():
    """Test rounding throws the right exceptions"""
    with pytest.raises(ValueError):
        scores.probability.functions.round_values(xr.DataArray(), -1, 5)


def test_propagate_nan_error():
    """Test propagating throws the right exceptions"""
    faulty_array = xr.Dataset({"lertitude": [1, 2, np.NaN, 4], "longitude": [20, 21, 22, 23, 24]})
    with pytest.raises(ValueError):
        scores.probability.functions.propagate_nan(faulty_array, "latitude")


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        ("x", cdf_test_data.EXP_PROPNAN_X),
        ("y", cdf_test_data.EXP_PROPNAN_Y),
    ],
)
def test_propagate_nan(dim, expected):
    """Tests `propagate_nan` with a variety of inputs."""
    result = scores.probability.functions.propagate_nan(cdf_test_data.DA_PROPNAN, dim)
    assertions.assert_dataarray_equal(result, expected, decimals=7)


def test_observed_cdf_errors():
    """Test `osbserved_cdf_errors` with a variety of inputs."""
    obs = xr.DataArray.from_dict({"dims": "temp", "data": [np.NaN, np.NaN, np.NaN, np.NaN]})
    threshold_dim = "irrelevant for test"
    threshold_values = None
    badprecision = -1

    # Bad precision raises a value error
    with pytest.raises(ValueError):
        scores.probability.functions.observed_cdf(obs, threshold_dim, threshold_values, precision=badprecision)

    # Null obs and a null threshold value raises a value error
    with pytest.raises(ValueError):
        scores.probability.functions.observed_cdf(obs, threshold_dim, threshold_values)


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
    result = scores.probability.functions.integrate_square_piecewise_linear(function_values, "x")
    assertions.assert_dataarray_equal(result, expected, decimals=7)


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
    output = scores.probability.functions.fill_cdf(cdf_test_data.DA_FILL_CDF1, "x", method, min_nonnan)
    assertions.assert_dataarray_equal(output, expected, decimals=7)


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
        scores.probability.functions.fill_cdf(
            cdf,
            threshold_dim,
            method,
            min_nonnan,
        )


@pytest.mark.parametrize(
    (
        "cdf",
        "threshold_dim",
        "tolerance",
        "error_msg_snippet",
    ),
    [
        (
            cdf_test_data.DA_DECREASING_CDFS1,
            "y",
            0,
            "'y' is not a dimension of `cdf`",
        ),
        (
            cdf_test_data.DA_DECREASING_CDFS1,
            "x",
            -1,
            "`tolerance` must be nonnegative.",
        ),
        (
            cdf_test_data.DA_CDF_ENVELOPE2,
            "y",
            0,
            "'y' is not a dimension of `cdf`",
        ),
        (
            cdf_test_data.DA_CDF_ENVELOPE1,
            "x",
            0,
            "CDFs should have no NaNs or be all NaN along `threshold_dim`",
        ),
        (
            cdf_test_data.DA_NAN_DECREASING_CDFS2,
            "x",
            0,
            "Coordinates along 'x' dimension should be increasing.",
        ),
    ],
)
def test_check_nan_decreasing_inputs(cdf, threshold_dim, tolerance, error_msg_snippet):
    """Tests that `_check_nan_decreasing_inputs` raises an exception as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        scores.probability.checks.check_nan_decreasing_inputs(cdf, threshold_dim, tolerance)


@pytest.mark.parametrize(
    ("cdf"),
    [
        (cdf_test_data.DA_CDF_ENVELOPE1),
        (cdf_test_data.DA_CDF_ENVELOPE2),
    ],
)
def test_cdf_envelope(cdf):
    """Tests `cdf_envelope` with a variety of inputs."""
    result = scores.probability.functions.cdf_envelope(cdf, "x")
    assertions.assert_dataarray_equal(result, cdf_test_data.EXP_CDF_ENVELOPE1, decimals=7)


def test_cdf_envelope_raises():
    """Tests that `cdf_envelope` raises the correct error."""
    with pytest.raises(ValueError, match="'y' is not a dimension of `cdf`"):
        scores.probability.functions.cdf_envelope(cdf_test_data.DA_CDF_ENVELOPE1, "y")
