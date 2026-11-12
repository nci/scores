"""
Tests for `scores.isoreg_impl`.
"""

import numpy as np
import pytest
import xarray as xr
from numpy import nan

from scores.processing.isoreg_impl import (
    _bootstrap_ir,
    _confidence_band,
    _contiguous_ir,
    _contiguous_quantile_ir,
    _do_ir,
    _iso_arg_checks,
    _nanquantile,
    _tidy_ir_inputs,
    _xr_to_np,
    isotonic_fit,
)
from tests.processing import isoreg_test_data as itd


@pytest.mark.parametrize(
    ("fcst", "obs", "weight", "expected"),
    [
        (itd.FCST_XRTONP, itd.OBS_XRTONP, None, itd.EXP_XRTONP1),
        (itd.FCST_XRTONP, itd.OBS_XRTONP, itd.WT_XRTONP, itd.EXP_XRTONP2),
    ],
)
def test__xr_to_np(fcst, obs, weight, expected):
    """Tests that `_xr_to_np` gives results as expected."""
    result = _xr_to_np(fcst, obs, weight)
    for i in range(3):
        np.testing.assert_array_equal(result[i], expected[i])


@pytest.mark.parametrize(
    ("fcst", "obs", "weight", "error_msg_snippet"),
    [
        (
            itd.FCST_XRTONP,
            itd.OBS_XRTONP2,
            None,
            "`fcst` and `obs` must have same dimensions.",
        ),
        (
            itd.FCST_XRTONP,
            itd.OBS_XRTONP,
            itd.OBS_XRTONP2,
            "`fcst` and `weight` must have same dimensions.",
        ),
    ],
)
def test__xr_to_np_raises(fcst, obs, weight, error_msg_snippet):
    """Tests that `_xr_to_np` raises as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        _xr_to_np(fcst, obs, weight)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "weight",
        "functional",
        "quantile_level",
        "solver",
        "bootstraps",
        "confidence_level",
        "error_msg_snippet",
    ),
    [
        (
            np.array([1, 2]),
            np.array([1]),
            None,
            "mean",
            None,
            None,
            None,
            0.9,
            "`fcst` and `obs` must have same shape.",
        ),
        (
            np.array([1, 2, "string"]),
            np.array([1, 2, 3]),
            None,
            "mean",
            None,
            None,
            None,
            0.9,
            "`fcst` must be an array of floats or integers.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, "category"]),
            None,
            "mean",
            None,
            None,
            None,
            0.9,
            "`obs` must be an array of floats or integers.",
        ),
        (
            np.array([1, 2]),
            np.array([10, 20]),
            np.array([1]),
            "mean",
            None,
            None,
            None,
            0.9,
            "`fcst` and `weight` must have same shape.",
        ),
        (
            xr.DataArray(data=[1, 2.1, 3]),
            xr.DataArray(data=[1, 2, nan]),
            xr.DataArray(data=[1, 2, "category"]),
            "mean",
            None,
            None,
            None,
            0.9,
            "`weight` must be an array of floats or integers, or else `None`.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            np.array([1, -1, nan]),
            "mean",
            None,
            None,
            None,
            0.9,
            "`weight` values must be either positive or NaN.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            np.array([1, 1, nan]),
            "median",
            None,
            None,
            None,
            0.9,
            "`functional` must be one of 'mean', 'quantile' or `None`.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            "quantile",
            1.0,
            None,
            None,
            0.9,
            "`quantile_level` must be strictly between 0 and 1.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            np.array([1, 1, nan]),
            "quantile",
            0.3,
            None,
            None,
            0.9,
            "Weighted quantile isotonic regression has not been implemented.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            None,
            0.5,
            None,
            None,
            0.9,
            "`functional` and `solver` cannot both be `None`.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            "quantile",
            0.5,
            np.quantile,
            None,
            0.9,
            "One of `functional` or `solver` must be `None`.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            "quantile",
            0.5,
            None,
            13.6,
            0.9,
            "`bootstraps` must be a positive integer.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            "quantile",
            0.5,
            None,
            -2,
            0.9,
            "`bootstraps` must be a positive integer.",
        ),
        (
            np.array([1, 2, 3]),
            np.array([10, 20, 30]),
            None,
            "quantile",
            0.5,
            None,
            5000,
            0.0,
            "`confidence_level` must be strictly between 0 and 1.",
        ),
    ],
)
def test__iso_arg_checks(  # pylint: disable=too-many-locals, too-many-arguments
    fcst,
    obs,
    weight,
    functional,
    quantile_level,
    solver,
    bootstraps,
    confidence_level,
    error_msg_snippet,
):
    """Tests that `_iso_arg_checks` raises as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        _iso_arg_checks(
            fcst=fcst,
            obs=obs,
            weight=weight,
            functional=functional,
            quantile_level=quantile_level,
            solver=solver,
            bootstraps=bootstraps,
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    ("fcst", "obs", "weights", "expected"),
    [
        (itd.FCST_TIDY1, itd.OBS_TIDY1, None, itd.EXP_TIDY1),
        (itd.FCST_TIDY1, itd.OBS_TIDY1, itd.WEIGHT_TIDY1, itd.EXP_TIDY2),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            None,
            (np.array([0, 1]), np.array([0.0, 1.0]), None),
        ),  # test that obs are converted to float
    ],
)
def test__tidy_ir_inputs(fcst, obs, weights, expected):
    """Tests that `_tidy_ir_inputs` gives results as expected."""
    result = _tidy_ir_inputs(fcst, obs, weight=weights)
    for i in range(3):
        np.testing.assert_array_equal(result[i], expected[i])


def test__tidy_ir_inputs_raises():
    """Tests that _tidy_ir_inputs raises as expected."""
    with pytest.raises(ValueError, match="pairs remaining after NaNs removed."):
        _tidy_ir_inputs(np.array([0.0, nan, 4.1, 3]), np.array([nan, 0.0, nan, nan]))


def _secmom_solver(y):
    """
    Solver for 'second moment about 0' isotonic regression.
    This is used to test the case where the solver of a singleton is not the
    value of the singleton.
    """
    return np.average(y**2)


@pytest.mark.parametrize(
    ("functional", "solver", "functional_preserves_singletons", "expected"),
    [
        ("mean", None, True, np.array([2.0, 2.0, 2.0, 6.0])),
        ("quantile", None, True, np.array([1.0, 1.0, 1.0, 6.0])),
        ("none", np.max, True, np.array([5.0, 5.0, 5.0, 6.0])),
        (None, _secmom_solver, False, np.array([26 / 3, 26 / 3, 26 / 3, 36])),
    ],
)
def test__do_ir(functional, solver, functional_preserves_singletons, expected):
    """
    Tests that `_do_ir` gives results as expected.
    Simultaneously supplies simple confirmation tests for `_contiguous_mean_ir`,
    `_contiguous_quantile_ir` and `_contiguous_ir`.
    """
    result = _do_ir(
        np.array([5.0, 1, 0, 6]),
        weight=None,
        functional=functional,
        quantile_level=0.5,
        solver=solver,
        functional_preserves_singletons=functional_preserves_singletons,
    )
    np.testing.assert_array_equal(result, expected)


def test__contiguous_quantile_ir():
    """
    Tests that `_contiguous_quantile_ir` gives results as expected.
    Using default `np.quantile` method ('linear') would fail this test,
    but it passes when `method="averaged_inverted_cdf"` is used.
    """
    y = np.array([1.0, 0.0, 0, 0, 0, 3, 4, 6, 5, 5, 9])
    alpha = 0.8
    expected = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0, 4.0, 6.0, 6.0, 6.0, 9.0])
    result = _contiguous_quantile_ir(y, alpha)
    np.testing.assert_array_equal(result, expected)


def _wmean_solver(y, weight):
    """solver for mean isotonic regression"""
    return np.average(y, weights=weight)


@pytest.mark.parametrize(
    ("y", "solver", "functional_preserves_singletons", "weight", "expected"),
    [
        (itd.Y1, np.median, True, None, itd.EXP_IR_MEDIAN),
        (itd.Y1, np.mean, True, None, itd.EXP_IR_MEAN),
        (itd.Y1, _wmean_solver, True, itd.W1, itd.EXP_IR_WMEAN),
        (itd.Y1, _wmean_solver, False, itd.W1, itd.EXP_IR_WMEAN),  # True or False give same result
        (np.ndarray(0), np.mean, True, None, 0),
        (itd.Y1, _secmom_solver, False, None, itd.EXP_IR_SECMOM),
        (itd.Y2, _secmom_solver, False, None, itd.EXP_IR_SECMOM2),
    ],
)
def test__contiguous_ir(y, solver, functional_preserves_singletons, weight, expected):
    """Tests that `_contiguous_ir` gives results as expected."""
    result = _contiguous_ir(y, solver, functional_preserves_singletons, weight=weight)
    np.testing.assert_array_equal(result, expected)


def test__contiguous_ir_raises():
    """Tests that `_contiguous_ir` raises as expected."""
    with pytest.raises(ValueError, match="`y` and `weight` must have same length."):
        _contiguous_ir(np.array([1.0, 2, 3]), np.mean, True, weight=np.array([1, 1]))


@pytest.mark.parametrize(
    ("weight", "functional", "q_level", "solver", "bootstrap", "expected"),
    [
        (itd.BS_WT, None, None, _wmean_solver, 3, itd.BS_EXP1),
        (None, "mean", None, None, 3, itd.BS_EXP2),
        (None, "quantile", 0.5, None, 1, itd.BS_EXP3),
    ],
)
def test__bootstrap_ir(  # pylint: disable=too-many-locals, too-many-arguments
    weight, functional, q_level, solver, bootstrap, expected
):
    """Tests that `_contiguous_ir` gives results as expected."""
    np.random.seed(seed=1)
    result = _bootstrap_ir(
        fcst=itd.BS_FCST,
        obs=itd.BS_OBS,
        weight=weight,
        functional=functional,
        quantile_level=q_level,
        solver=solver,
        bootstraps=bootstrap,
    )
    np.testing.assert_array_equal(result, expected)


def test__confidence_band():
    """Tests that `_confidence_band` gives results as expected."""
    result = _confidence_band(itd.CB_BOOT_INPUT, 0.5, 4)
    for i in range(2):
        np.testing.assert_array_equal(result[i], itd.EXP_CB[i])


def test__confidence_band_nan():
    """_confidence_band returns expected objects with all NaN input"""
    result = _confidence_band(np.where(False, itd.CB_BOOT_INPUT, np.nan), 0.5, 4)

    expected = np.array([np.nan] * 7)
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[1], expected)


def test__nanquantile():
    """_nanquantile returns expected results."""
    for quant in [0.75, 0.25]:
        result = _nanquantile(itd.CB_BOOT_INPUT, quant)
        np.testing.assert_array_equal(result, itd.EXP_CB2[quant])


@pytest.mark.parametrize(
    ("fcst", "obs", "bootstraps", "report_bootstrap_results", "expected"),
    [
        (itd.FCST_XR, itd.OBS_XR, None, None, itd.EXP_IF1),
        (itd.FCST_ARRAY, itd.OBS_ARRAY, None, None, itd.EXP_IF1),
        (itd.FCST_ARRAY, itd.OBS_ARRAY, 3, False, itd.EXP_IF2),
        (itd.FCST_ARRAY, itd.OBS_ARRAY, 3, True, itd.EXP_IF3),
    ],
)
def test_isotonic_fit(fcst, obs, bootstraps, report_bootstrap_results, expected):
    """Tests that `isotonic_fit` gives results as expected."""
    np.random.seed(seed=1)
    result = isotonic_fit(
        fcst,
        obs,
        functional=None,
        solver=np.mean,
        bootstraps=bootstraps,
        confidence_level=0.5,
        min_non_nan=3,
        report_bootstrap_results=report_bootstrap_results,
    )
    # check arrays are equal
    array_keys = [
        "fcst_sorted",
        "fcst_counts",
        "regression_values",
        "confidence_band_lower_values",
        "confidence_band_upper_values",
    ]
    if report_bootstrap_results:
        array_keys.append("bootstrap_results")
    for key in array_keys:
        np.testing.assert_array_equal(result[key], expected[key])

    assert result["confidence_band_levels"] == expected["confidence_band_levels"]

    for key in [
        "regression_func",
        "confidence_band_lower_func",
        "confidence_band_upper_func",
    ]:
        np.testing.assert_array_equal(result[key](itd.TEST_POINTS), expected[key](itd.TEST_POINTS))


def test_isotonic_fit2():
    """
    Tests that `isotonic_fit` handles integer obs inputs correctly.
    If integers aren't converted to float, the regression values are [0.0, 0.0],
    which is incorrect.
    """
    result = isotonic_fit(np.array([0, 1]), np.array([1, 0]), solver=np.mean, functional=None)
    regression_result = result["regression_values"]
    regression_expected = np.array([0.5, 0.5])
    np.testing.assert_array_equal(regression_result, regression_expected)


def test_istonic_fit3():
    """
    Tests that isotonic fit handles `functional_preserves_singletons=False` correctly,
    including with bootstrapped data.
    """
    np.random.seed(seed=3)
    result = isotonic_fit(
        np.array([0, 1, 2]),
        np.array([-1.0, 0, 2]),
        functional=None,
        solver=_secmom_solver,
        functional_preserves_singletons=False,
        bootstraps=2,
        report_bootstrap_results=True,
    )

    # first test regression values
    regression_result = result["regression_values"]
    regression_expected = np.array([1 / 2, 1 / 2, 4])
    np.testing.assert_array_equal(regression_result, regression_expected)

    # Now test bootstrap output.
    # `selection` is [2, 0, 1], [0, 0, 0] when the seed is 3.
    # `fcst_sample` is [2, 0, 1], [0, 0, 0]
    # `obs_sample` is [2, -1, 0], [-1, -1, -1]
    # Now need to sort the data so that fcst increasing, obs match new fcst.
    # sorted fcst is [0, 1, 2], [0, 0, 0]
    # sorted obs is [-1, 0, 2], [-1, -1, -1]
    # Regressed values at sorted `fcst_sample`: [1/2, 1/2, 4], [1, nan, nan];
    # note that functional_preserves_singletons=False will give an incorrect result of
    # [-1, 0, 2] for the first row.
    bootstrap_result = result["bootstrap_results"]
    bootstrap_expected = np.array([[1 / 2, 1 / 2, 4], [1, nan, nan]])
    np.testing.assert_array_equal(bootstrap_result, bootstrap_expected)
