"""
Contains unit tests for scores.df.continuous
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import numpy as np
import pandas as pd
import polars as pl
import pytest

import scores.df as scores

PRECISION = 4

# Constructors for each backend under test. Keyed by name so failures identify the backend.
SERIES_CONSTRUCTORS = {"pandas": pd.Series, "polars": pl.Series}


@pytest.fixture(params=sorted(SERIES_CONSTRUCTORS), ids=sorted(SERIES_CONSTRUCTORS))
def series(request):
    """Returns a series constructor for each supported dataframe backend."""
    return SERIES_CONSTRUCTORS[request.param]


@pytest.fixture
def fcst(series):
    """Creates a forecast series for the backend under test."""
    return series([1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 3.0])


@pytest.fixture
def obs(series):
    """Creates an observation series for the backend under test."""
    return series([1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0])


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("mse", 1.0909),
        ("rmse", 1.0445),
        ("mae", 0.7273),
    ],
)
def test_metric_matches_expected(metric, expected, fcst, obs):
    """
    Test that each metric returns the expected value, identically across backends.
    """
    result = getattr(scores.continuous, metric)(fcst, obs)
    assert isinstance(result, float)
    assert round(result, PRECISION) == expected


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("mse", 1.4545),
        ("rmse", 1.206),
        ("mae", 0.9091),
    ],
)
def test_metric_against_scalar(metric, expected, fcst):
    """
    Test that a series can be compared against a single point value, as `scores.pandas`
    allows.
    """
    result = getattr(scores.continuous, metric)(fcst, 1.0)
    assert round(result, PRECISION) == expected


@pytest.mark.parametrize(
    "metric, expected",
    [
        # The angular difference between each pair is 20 degrees, not 340.
        ("mse", 400.0),
        ("rmse", 20.0),
        ("mae", 20.0),
    ],
)
def test_metric_is_angular(metric, expected, series):
    """
    Test that angular data wraps around 360 degrees rather than being treated linearly.
    """
    fcst_angular = series([10.0, 350.0, 0.0])
    obs_angular = series([350.0, 10.0, 340.0])
    result = getattr(scores.continuous, metric)(fcst_angular, obs_angular, is_angular=True)
    assert round(result, PRECISION) == expected


def test_is_angular_takes_smaller_explementary_angle(series):
    """
    Test the boundary of the angular calculation: a difference of exactly 180 degrees is
    preserved, while anything larger is reflected back below 180.
    """
    result = scores.continuous.mae(series([0.0, 0.0]), series([180.0, 270.0]), is_angular=True)
    assert round(result, PRECISION) == 135.0  # mean of 180 and 90


def test_missing_values_skipped_pandas():
    """
    Test that NaN values are skipped for pandas, matching `scores.pandas` behaviour.
    """
    fcst_nan = pd.Series([-1.0, 3.0, 1.0, 3.0, np.nan, 2.0])
    obs_nan = pd.Series([1.0, 1.0, 1.0, 2.0, 1.0, 2.0])
    assert round(scores.continuous.mse(fcst_nan, obs_nan), PRECISION) == 1.8


def test_missing_values_skipped_polars():
    """
    Test null handling for Polars, which distinguishes null from NaN. Null is skipped,
    whereas NaN propagates through the mean. This differs from pandas and is documented
    in the metric docstrings.
    """
    obs_null = pl.Series([1.0, 1.0, 1.0, 2.0, 1.0, 2.0])

    fcst_null = pl.Series([-1.0, 3.0, 1.0, 3.0, None, 2.0])
    assert round(scores.continuous.mse(fcst_null, obs_null), PRECISION) == 1.8

    fcst_nan = pl.Series([-1.0, 3.0, 1.0, 3.0, np.nan, 2.0])
    assert np.isnan(scores.continuous.mse(fcst_nan, obs_null))
