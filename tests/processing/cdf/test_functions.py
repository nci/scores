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
from scores.processing.cdf.cdf_functions import (
    _var_from_cdf,
    check_cdf,
    check_cdf_support,
    expectedvalue_from_cdf,
    variance_from_cdf,
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
    faulty_array = xr.Dataset({"lertitude": [1, 2, np.NaN, 4], "longitude": [20, 21, 22, 23, 24]})
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
    obs = xr.DataArray.from_dict({"dims": "temp", "data": [np.NaN, np.NaN, np.NaN, np.NaN]})
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


@pytest.mark.parametrize(
    ("cdf", "threshold_dim", "dims", "error_msg_snippet"),
    [
        (
            cdf_test_data.DA_OBSERVED_CDF,
            "crazy",
            None,
            "`thresh_dim` is not a dimension of `da_cdf`",
        ),
        (
            cdf_test_data.DA_OBSERVED_CDF,
            "date",
            ["date", "station"],
            "`thresh_dim` is in `my_dims`",
        ),
        (
            cdf_test_data.DA_OBSERVED_CDF,
            "station",
            ["date", "bird"],
            "`my_dims` is not a subset of dimensions of `da_cdf`",
        ),
        (
            cdf_test_data.DA_WITHIN_BOUNDS3,
            "x",
            None,
            "values of `da_cdf` must be in the closed interval",
        ),
        (
            cdf_test_data.DA_NAN_DECREASING_CDFS2,
            "x",
            None,
            "coordinates along `thresh_dim` of `da_cdf` must be increasing",
        ),
    ],
)
def test_check_cdf(cdf, threshold_dim, dims, error_msg_snippet):
    """Tests that `check_cdf` raises an exceptions as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        check_cdf(cdf, threshold_dim, dims, "da_cdf", "thresh_dim", "my_dims")


@pytest.mark.parametrize(
    ("cdf", "lower_supp", "upper_supp"),
    [
        (cdf_test_data.DA_SUPPORT1, -np.inf, np.inf),
        (cdf_test_data.DA_SUPPORT1, 0, 1),
        (cdf_test_data.DA_SUPPORT2, 1, 2),
        (cdf_test_data.DA_SUPPORT3, 1, 2),
    ],
)
def test_check_cdf_support(cdf, lower_supp, upper_supp):
    """Tests that `check_cdf_support` does not raise as expected."""
    check_cdf_support(cdf, "x", lower_supp, upper_supp, "my_cdf", "my_dim")


@pytest.mark.parametrize(
    ("cdf", "lower_supp", "upper_supp", "error_msg_snippet"),
    [
        (cdf_test_data.DA_SUPPORT1, 3, 1, "`upper_supp < lower_supp`"),
        (cdf_test_data.DA_SUPPORT1, np.inf, 9, "`upper_supp < lower_supp`"),
        (cdf_test_data.DA_SUPPORT1, 2, 3, "`my_cdf` is not 0 when `my_cdf\\[my_dim\\] < 2`"),
        (cdf_test_data.DA_SUPPORT1, -2, 0.9, "`my_cdf` is not 1 when `my_cdf\\[my_dim\\] > 0.9`"),
    ],
)
def test_check_cdf_support_raises(cdf, lower_supp, upper_supp, error_msg_snippet):
    """Tests that `check_cdf_support` raises as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        check_cdf_support(cdf, "x", lower_supp, upper_supp, "my_cdf", "my_dim")


def test_expectedvalue_from_cdf():
    """Tests that `expectedvalue_from_cdf` gives correct output."""
    result = expectedvalue_from_cdf(cdf_test_data.DA_CDF_EXPECTEDVALUE, "x")
    xr.testing.assert_allclose(result, cdf_test_data.EXP_EXPECTEDVALUE)


def test_expectedvalue_from_cdf_raises():
    """Tests that `expectedvalue_from_cdf` raises if `nonnegative_support = False`."""
    with pytest.raises(ValueError, match="This function currently only handles"):
        expectedvalue_from_cdf(cdf_test_data.DA_CDF_EXPECTEDVALUE, "x", nonnegative_support=False)


def test___var_from_cdf():
    """Tests that `_var_from_cdf` gives correct output."""
    result = _var_from_cdf(cdf_test_data.DA_FUNCVALS, "x")
    xr.testing.assert_allclose(result, cdf_test_data.EXP_VARCDF)


def test_variance_from_cdf():
    """Tests that 'variance_from_cdf' gives correct output."""
    result = variance_from_cdf(cdf_test_data.DA_CDF_VARIANCE, "x")
    xr.testing.assert_allclose(result, cdf_test_data.EXP_VARIANCE)
