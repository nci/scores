"""
Contains unit tests for scores.continuous.standard
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import numpy as np
import pandas as pd
import pytest
import torch

import scores

PRECISION = 4

# Mean Squared Error
#
DA1_BIAS = torch.tensor(
    np.array([[1, 1, np.nan], [0, 0, 0], [0.5, -0.5, 0.5]]),
)

DA2_BIAS = torch.tensor(
    np.array([[2, 2, 6], [2, 10, 0], [-0.5, 0.5, -0.5]]),
)

BIAS_WEIGHTS = torch.tensor(
    np.array([[1, 1, 1], [3, 0, 0], [3, 0, 0]]),
)

EXP_BIAS2 = torch.tensor(np.array([-1.33333]))
EXP_BIAS3 = torch.tensor(np.array(-1.625))


def test_mse_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])

    fcst_tensor = torch.tensor([1.0, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_tensor = torch.tensor([1.0, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])

    expected = 1.0909
    pd_result = scores.continuous.mse(fcst_pd_series, obs_pd_series)
    assert isinstance(pd_result, float)
    assert round(pd_result, 4) == expected

    tensor_result = scores.continuous.mse(fcst_tensor, obs_tensor)
    assert tensor_result.dtype is torch.float
    assert torch.round(tensor_result, decimals=4) == torch.tensor(expected)


@pytest.mark.parametrize(
    ("fcst", "obs", "weights", "expected"),
    [
        # Check weighting works
        # (DA1_BIAS, DA2_BIAS, BIAS_WEIGHTS, EXP_BIAS2),
        (DA1_BIAS, DA2_BIAS, None, EXP_BIAS3),
    ],
)
def test_additive_bias(fcst, obs, weights, expected):
    """
    Tests continuous.additive_bias
    Also tests mean_error (which is an identical function)
    """

    fcst = fcst.rename(None)
    obs = obs.rename(None)

    if weights is None:
        weights = torch.ones(fcst.shape)

    weights = weights.rename(None)

    weights = weights * (~torch.isnan(fcst))  # mask out nans from fcst
    weights = weights * (~torch.isnan(obs))  # mask out nans from obs
    tensor_result = scores.loss.continuous.additive_bias(fcst, obs, weights=weights)

    tensor_result = tensor_result.rename(None)
    assert (torch.round(tensor_result, decimals=4) == torch.tensor(expected)).all()


# def test_mse_dataframe():
#     """
#     Test calculation works correctly on dataframe columns
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
#     expected = 1.0909
#     result = scores.continuous.mse(df["fcst"], df["obs"])
#     assert isinstance(result, float)
#     assert round(result, PRECISION) == expected


# # Root Mean Squared Error


# @pytest.fixture
# def rmse_fcst_pandas():
#     """Creates forecast Pandas series for test."""
#     return pd.Series([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


# @pytest.fixture
# def rmse_fcst_nan_pandas():
#     """Creates forecast Pandas series containing NaNs for test."""
#     return pd.Series([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


# @pytest.fixture
# def rmse_obs_pandas():
#     """Creates observation Pandas series for test."""
#     return pd.Series([1, 1, 1, 2, 1, 2, 1, 1, -1, 3, 1])


# @pytest.mark.parametrize(
#     "forecast, observations, expected, request_kwargs",
#     [
#         ("rmse_fcst_pandas", "rmse_obs_pandas", 1.3484, {}),
#         ("rmse_fcst_pandas", 1, 1.3484, {}),
#         ("rmse_fcst_nan_pandas", "rmse_obs_pandas", 1.3784, {}),
#     ],
#     ids=[
#         "pandas-series-1d",
#         "pandas-to-point",
#         "pandas-series-nan-1d",
#     ],
# )
# def test_rmse_pandas_1d(forecast, observations, expected, request_kwargs, request):
#     """
#     Test RMSE for the following cases:
#        * Calculates the correct value for a simple pandas 1d series
#     """
#     if isinstance(forecast, str):
#         forecast = request.getfixturevalue(forecast)
#     if isinstance(observations, str):
#         observations = request.getfixturevalue(observations)
#     result = scores.continuous.rmse(forecast, observations, **request_kwargs)
#     if not isinstance(result, float):
#         assert (result.round(PRECISION) == expected).all()
#     else:
#         assert np.round(result, PRECISION) == expected


# # Mean Absolute Error


# def test_mae_pandas_series():
#     """
#     Test calculation works correctly on pandas series
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     expected = 0.7273
#     result = scores.continuous.mae(fcst_pd_series, obs_pd_series)
#     assert isinstance(result, float)
#     assert round(result, 4) == expected


# def test_mae_dataframe():
#     """
#     Test calculation works correctly on dataframe columns
#     """

#     fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
#     obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
#     df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
#     expected = 0.7273
#     result = scores.continuous.mae(df["fcst"], df["obs"])
#     assert isinstance(result, float)
#     assert round(result, PRECISION) == expected
