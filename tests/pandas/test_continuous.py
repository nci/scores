"""
Contains unit tests for scores.continuous.standard
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import numbers

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scores.pandas as scores
from scores.continuous.standard_impl import additive_bias

PRECISION = 4

# Mean Squared Error


def test_mse_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    expected = 1.0909
    result = scores.continuous.mse(fcst_pd_series, obs_pd_series)
    assert isinstance(result, float)
    assert round(result, 4) == expected


def test_mse_dataframe():
    """
    Test calculation works correctly on dataframe columns
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
    expected = 1.0909
    result = scores.continuous.mse(df["fcst"], df["obs"])
    assert isinstance(result, float)
    assert round(result, PRECISION) == expected


# Root Mean Squared Error


@pytest.fixture
def rmse_fcst_pandas():
    """Creates forecast Pandas series for test."""
    return pd.Series([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_fcst_nan_pandas():
    """Creates forecast Pandas series containing NaNs for test."""
    return pd.Series([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_obs_pandas():
    """Creates observation Pandas series for test."""
    return pd.Series([1, 1, 1, 2, 1, 2, 1, 1, -1, 3, 1])


@pytest.mark.parametrize(
    "forecast, observations, expected, request_kwargs",
    [
        ("rmse_fcst_pandas", "rmse_obs_pandas", 1.3484, {}),
        ("rmse_fcst_pandas", 1, 1.3484, {}),
        ("rmse_fcst_nan_pandas", "rmse_obs_pandas", 1.3784, {}),
    ],
    ids=[
        "pandas-series-1d",
        "pandas-to-point",
        "pandas-series-nan-1d",
    ],
)
def test_rmse_pandas_1d(forecast, observations, expected, request_kwargs, request):
    """
    Test RMSE for the following cases:
       * Calculates the correct value for a simple pandas 1d series
    """
    if isinstance(forecast, str):
        forecast = request.getfixturevalue(forecast)
    if isinstance(observations, str):
        observations = request.getfixturevalue(observations)
    result = scores.continuous.rmse(forecast, observations, **request_kwargs)
    if not isinstance(result, float):
        assert (result.round(PRECISION) == expected).all()
    else:
        assert np.round(result, PRECISION) == expected


# Mean Absolute Error


def test_mae_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    expected = 0.7273
    result = scores.continuous.mae(fcst_pd_series, obs_pd_series)
    assert isinstance(result, float)
    assert round(result, 4) == expected


def test_mae_dataframe():
    """
    Test calculation works correctly on dataframe columns
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
    expected = 0.7273
    result = scores.continuous.mae(df["fcst"], df["obs"])
    assert isinstance(result, float)
    assert round(result, PRECISION) == expected


# Additive Bias (Mean Error)


def test_additive_bias_pandas_series():
    fcst = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])

    expected = 0.5455

    result = scores.continuous.additive_bias(fcst=fcst, obs=obs)

    assert isinstance(result, numbers.Real)
    assert round(float(result), PRECISION) == expected


def test_additive_bias_raises_when_not_xarray_and_weights_given():
    fcst = np.array([1, 2, 3])
    obs = np.array([1, 2, 3])
    weights = xr.DataArray([1, 2, 3])

    with pytest.raises(ValueError):
        additive_bias(fcst=fcst, obs=obs, weights=weights)
