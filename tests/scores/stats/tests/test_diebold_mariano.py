"""
This module contains unit tests for scores.stats.tests.diebold_mariano_impl
"""
import numpy as np
import pytest
import xarray as xr

from scores.stats.tests.diebold_mariano_impl import (
    _dm_gamma_hat_k,
    _dm_test_statistic,
    _dm_v_hat,
    _hg_func,
    _hg_method_stat,
    _hln_method_stat,
    diebold_mariano,
)


@pytest.mark.parametrize(
    (
        "da_timeseries",
        "ts_dim",
        "h_coord",
        "method",
        "confidence_level",
        "statistic_distribution",
        "error_msg",
    ),
    [
        (
            xr.DataArray(data=[1, 2], dims=["x"], coords={"x": [0, 1]}),
            "x",
            "h",
            "KEV",
            -0.4,
            "t",
            "`method` must be one of",
        ),
        (
            xr.DataArray(data=[1, 2], dims=["x"], coords={"x": [0, 1]}),
            "x",
            "h",
            "HG",
            -0.4,
            "chi_sq",
            "`statistic_distribution` must be one of",
        ),
        (
            xr.DataArray(data=[1, 2], dims=["x"], coords={"x": [0, 1]}),
            "x",
            "h",
            "HLN",
            -0.4,
            "t",
            "`confidence_level` must be strictly between 0 and 1.",
        ),
        (
            xr.DataArray(data=[1, 2], dims=["x"], coords={"x": [0, 1]}),
            "x",
            "h",
            "HLN",
            1.0,
            "t",
            "`confidence_level` must be strictly between 0 and 1.",
        ),
        (
            xr.DataArray(data=[1, 2], dims=["x"], coords={"x": [0, 1]}),
            "x",
            "h",
            "HLN",
            0.95,
            "t",
            "`da_timeseries` must have exactly two dimensions.",
        ),
        (
            xr.DataArray(data=[[1], [2]], dims=["x", "y"], coords={"x": [0, 1], "y": [1]}),
            "z",
            "y",
            "HLN",
            0.95,
            "t",
            "`ts_dim` 'z' must be a dimension of `da_timeseries`.",
        ),
        (
            xr.DataArray(
                data=[[1, 2]],
                dims=["x", "y"],
                coords={"x": [0], "y": [0, 1], "h": ("x", [1])},
            ),
            "x",
            "h1",
            "HLN",
            0.5,
            "t",
            "`h_coord` must be among the coordinates of `da_timeseries`.",
        ),
        (
            xr.DataArray(
                data=[[1, 2]],
                dims=["x", "y"],
                coords={"x": [0], "y": [0, 1], "h": ("x", [1.5])},
            ),
            "x",
            "h",
            "HLN",
            0.3,
            "t",
            " must be an integer.",
        ),
        (
            xr.DataArray(
                data=[[1, 2]],
                dims=["x", "y"],
                coords={"x": [0], "y": [0, 1], "h": ("x", [np.nan])},
            ),
            "x",
            "h",
            "HLN",
            0.9,
            "t",
            " must be an integer.",
        ),
        (
            xr.DataArray(
                data=[[1, 2]],
                dims=["x", "y"],
                coords={"x": [0], "y": [0, 1], "h": ("x", [-2])},
            ),
            "x",
            "h",
            "HLN",
            0.8,
            "t",
            " must be positive.",
        ),
        (
            xr.DataArray(
                data=[[1, 2, 3, 4], [1, 2, np.nan, np.nan]],
                dims=["x", "y"],
                coords={"x": [0, 1], "y": [0, 1, 2, 3], "h": ("x", [3, 3])},
            ),
            "x",
            "h",
            "HLN",
            0.7,
            "t",
            "must be less than the length of the corresponding timeseries",
        ),
        (
            xr.DataArray(
                data=[[1, 2, 3, 4], [1, 2, np.nan, np.nan]],
                dims=["x", "y"],
                coords={"x": [0, 1], "y": [0, 1, 2, 3], "h": ("x", [4, 1])},
            ),
            "x",
            "h",
            "HLN",
            0.6,
            "t",
            "must be less than the length of the corresponding timeseries",
        ),
    ],
)
def test_diebold_mariano_raises(
    da_timeseries,
    ts_dim,
    h_coord,
    method,
    confidence_level,
    statistic_distribution,
    error_msg,
):
    """Tests that diebold_mariano raises a ValueError as expected."""
    with pytest.raises(ValueError, match=error_msg):
        diebold_mariano(
            da_timeseries,
            ts_dim,
            h_coord,
            method,
            confidence_level,
            statistic_distribution,
        )


def test__hg_func():
    """Tests that _hg_func returns as expected."""
    pars = [2, 0.5]
    lag = np.array([0, 1, 2])
    acv = np.array([1, 2, -1])
    expected = 4 * np.exp(np.array([0, -6, -12])) - acv
    result = _hg_func(pars, lag, acv)
    np.testing.assert_allclose(result, expected)


DM_DIFFS = np.array([1.0, 2, 3, 4])
DM_DIFFS2 = np.array([0.0, -1, 0, 2, -1, 0, 1, -4, -1, -1, 2, 1, 0, 2, -1])
DM_DIFFS3 = np.array([0.0, -1, 0, 1, -1, 1, -2, -2, -1, -1])


@pytest.mark.parametrize(
    ("diffs", "h", "expected"),
    [
        (DM_DIFFS2, 14, -0.169192),  # cross-checked with R verification
        (DM_DIFFS2, 1, -0.169192),
        (DM_DIFFS3, 10, -1.860521),  # cross-checked with R verification
    ],
)
def test__hg_method_stat1(diffs, h, expected):
    """
    Tests that _hg_method_stat returns result as expected when least_squares
    routine is successful.

    The results of the first test were cross-checked with the R package `verification`.
    The following R code reproduces the results:
        library(verification)
        obs <- rep(0, 15)
        fcst1 <- c(1, 2, -1, 4, -2, 3, 4, 1, 0, 0, -2, 3, 4, -5, -4)
        fcst1 <- c(1, 3, 1, 2, -3, 3, 3, 5, 1, -1, 0, 2, 4, -3, -5)
        test <- predcomp.test(obs, fcst1, fcst2, test = "HG")
        summary(test)
    """
    result = _hg_method_stat(diffs, h)
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, 1.25),
        (2, -1.5),
    ],
)
def test__dm_gamma_hat_k(k, expected):
    """Tests that _dm_gamma_hat_k returns values as expected."""
    result = _dm_gamma_hat_k(DM_DIFFS, 2.5, 4, k)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("diffs", "h", "expected"),
    [
        (DM_DIFFS, 3, (5.0 + 2 * 1.25 + 2 * (-1.5)) / 16),
        (np.array([1, -1, 1, -1]), 2, np.nan),  # original result = 0, so changed to NaN
    ],
)
def test__dm_v_hat(diffs, h, expected):
    """Tests that _dm_v_hat returns values as expected."""

    np.testing.assert_allclose(_dm_v_hat(diffs, np.mean(diffs), len(diffs), h), expected)


# DM test stat when timeseries is [1, 2, 3, 4] and h = 2
DM_TEST_STAT_EXP1 = ((3 / 8) ** 0.5) * 2.5 * (0.46875 ** (-0.5))


@pytest.mark.parametrize(
    ("diffs", "h", "expected"),
    [
        (DM_DIFFS, 2, DM_TEST_STAT_EXP1),
        (np.array([1, -1, 1, -1]), 2, np.nan),  # v_hat = 0, so output NaN
    ],
)
def test__hln_method_stat(diffs, h, expected):
    """Tests that _hln_method_stat returns as expected."""
    result = _hln_method_stat(diffs, h)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("diffs", "h", "method", "expected"),
    [
        (DM_DIFFS, 2, "HLN", DM_TEST_STAT_EXP1),
        (np.array([np.nan, 1, 2, 3, 4.0, np.nan]), 2, "HLN", DM_TEST_STAT_EXP1),
        (np.array([1.0, 1, 1, 1]), 2, "HLN", np.nan),
        (np.array([0, 0, 0, 0]), 2, "HLN", np.nan),
        (DM_DIFFS2, 14, "HG", -0.1691921),
    ],
)
def test__dm_test_statistic(diffs, h, method, expected):
    """Tests that _dm_test_statistic returns values as expected."""
    result = _dm_test_statistic(diffs, h, method)
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("diff", "h", "method", "error_msg"),
    [
        (DM_DIFFS, 3, "KEV", "`method` must be one of"),
        (DM_DIFFS, 100, "HLN", "The condition"),
        (np.array([np.nan, 1, np.nan]), 2, "HLN", "The condition"),
        (DM_DIFFS, 0, "HG", "The condition"),
    ],
)
def test__dm_test_statistic_raises(diff, h, method, error_msg):
    """Tests that _dm_test_statistic raises a ValueError as expected."""
    with pytest.raises(ValueError, match=error_msg):
        _dm_test_statistic(diff, h, method)


# DM test stat when timeseries is [2.0, 1, -3, -1, 0] and h = 3
DM_TEST_STAT_EXP2 = ((6 / 25) ** 0.5) * (-0.2) * (0.0864 ** (-0.5))

# expected outputs for dm_test_stats
DM_TEST_STATS_T_EXP = xr.Dataset(
    data_vars=dict(
        mean=(["lead_day"], [2.5, -0.2, 1.0]),
        dm_test_stat=(["lead_day"], [DM_TEST_STAT_EXP1, DM_TEST_STAT_EXP2, np.nan]),
        timeseries_len=(["lead_day"], [4, 5, 5]),
        confidence_gt_0=(
            ["lead_day"],
            [0.9443164226429581, 0.3778115634892615, np.nan],
        ),
        ci_upper=(["lead_day"], [5.131140307989639, 1.079108068801774, np.nan]),
        ci_lower=(
            ["lead_day"],
            [-0.13114030798963894, -1.4791080688017741, np.nan],
        ),
    ),
    coords={"lead_day": [1, 2, 3]},
)

DM_TEST_STATS_NORMAL_EXP = xr.Dataset(
    data_vars=dict(
        mean=(["lead_day"], [2.5, -0.2, 1.0]),
        dm_test_stat=(["lead_day"], [DM_TEST_STAT_EXP1, DM_TEST_STAT_EXP2, np.nan]),
        timeseries_len=(["lead_day"], [4, 5, 5]),
        confidence_gt_0=(
            ["lead_day"],
            [0.9873263406612659, 0.36944134018176367, np.nan],
        ),
        ci_upper=(["lead_day"], [4.339002261450286, 0.7869121761708835, np.nan]),
        ci_lower=(
            ["lead_day"],
            [0.6609977385497137, -1.1869121761708834, np.nan],
        ),
    ),
    coords={"lead_day": [1, 2, 3]},
)


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        ("t", DM_TEST_STATS_T_EXP),
        ("normal", DM_TEST_STATS_NORMAL_EXP),
    ],
)
def test_diebold_mariano(distribution, expected):
    """Tests that diebold_mariano gives results as expected."""
    da_timeseries = xr.DataArray(
        data=[[1, 2, 3.0, 4, np.nan], [2.0, 1, -3, -1, 0], [1.0, 1, 1, 1, 1]],
        dims=["lead_day", "valid_date"],
        coords={
            "lead_day": [1, 2, 3],
            "valid_date": ["a", "b", "c", "d", "e"],
            "h": ("lead_day", [2, 3, 4]),
        },
    )
    result = diebold_mariano(
        da_timeseries,
        "lead_day",
        "h",
        method="HLN",
        confidence_level=0.9,
        statistic_distribution=distribution,
    )
    xr.testing.assert_allclose(result, expected, atol=7)
