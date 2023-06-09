"""
Contains unit tests for scores.continuous
"""

import numpy as np
import numpy.random
import pandas as pd
import pytest
import xarray as xr

import scores.continuous

PRECISION = 4

# Mean Squared Error


def test_mse_xarray_1d():
    """
    Test both value and expected datatype matches for xarray calculation
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_as_xarray_1d = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    result = scores.continuous.mse(fcst_as_xarray_1d, obs_as_xarray_1d)
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])

    expected = xr.DataArray(1.0909)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_mse_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    expected = 1.0909
    result = scores.continuous.mse(fcst_pd_series, obs_pd_series)
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
    assert round(result, PRECISION) == expected


def test_mse_xarray_to_point():
    """
    Test MSE calculates the correct value for a simple 1d sequence
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    result = scores.continuous.mse(fcst_as_xarray_1d, 1)
    expected = xr.DataArray(1.45454545)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_mse():
    """
    Test MSE calculates the correct value on a 2d array
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.mse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d)
    expected = xr.DataArray(142.33162)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_mse_with_dimensions():
    """
    Assert that MSE can correctly calculate MSE along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.mse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims="latitude")

    expected_values = [290.0929, 90.8107, 12.2224, 176.2005]
    expected_dimensions = ("longitude",)
    assert all(result.round(4) == expected_values)
    assert result.dims == expected_dimensions


# Root Mean Squared Error

def test_rmsd_xarray_1d():
    """
    Test both value and expected datatype matches for xarray calculation using rmsd link
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_as_xarray_1d = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    result = scores.continuous.rmsd(fcst_as_xarray_1d, obs_as_xarray_1d)

    expected = xr.DataArray(1.0445)
    assert result.round(PRECISION) == expected.round(PRECISION)

def test_rmse_xarray_1d():
    """
    Test both value and expected datatype matches for xarray calculation
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_as_xarray_1d = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    result = scores.continuous.rmse(fcst_as_xarray_1d, obs_as_xarray_1d)

    expected = xr.DataArray(1.0445)
    assert result.round(PRECISION) == expected.round(PRECISION)

def test_rmse_xarray_all():
    """
    Test both value and expected datatype matches for xarray calculation preserving all dims
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_as_xarray_1d = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    result = scores.continuous.rmse(fcst_as_xarray_1d, obs_as_xarray_1d, preserve_dims = 'all')

    expected_values = xr.DataArray([0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 2])
    assert all(result.round(4) == expected_values)

def test_rmse_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    expected = 1.0445
    result = scores.continuous.rmse(fcst_pd_series, obs_pd_series)
    assert round(result, 4) == expected


def test_rmse_dataframe():
    """
    Test calculation works correctly on dataframe columns
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    df = pd.DataFrame({"fcst": fcst_pd_series, "obs": obs_pd_series})
    expected = 1.0445
    result = scores.continuous.rmse(df["fcst"], df["obs"])
    assert round(result, PRECISION) == expected


def test_rmse_xarray_to_point():
    """
    Test RMSE calculates the correct value for a simple 1d sequence
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    result = scores.continuous.rmse(fcst_as_xarray_1d, 1)
    expected = xr.DataArray(1.206)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_rmse():
    """
    Test RMSE calculates the correct value on a 2d array
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.rmse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d)
    expected = xr.DataArray(11.9303)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_rmse_with_dimensions():
    """
    Assert that rmse can correctly calculate RMSE along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.rmse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims="latitude")

    expected_values = [17.0321,  9.5295,  3.4961, 13.2741]
    expected_dimensions = ("longitude",)

    assert all(result.round(4) == expected_values)
    assert result.dims == expected_dimensions

def test_xarray_rmse_dimension_handling_with_arrays():
    """
    Test RMSE is calculated correctly along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    reduce_lat = scores.continuous.rmse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims="latitude")

    preserve_lon = scores.continuous.rmse(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, preserve_dims="longitude")

    expected_values = [17.0321,  9.5295,  3.4961, 13.2741]
    expected_dimensions = ("longitude",)

    assert all(reduce_lat.round(4) == expected_values)
    assert all(preserve_lon.round(4) == expected_values)
    assert reduce_lat.dims == expected_dimensions
    assert preserve_lon.dims == expected_dimensions

# Mean Absolute Error


def test_mae_xarray_1d():
    """
    Test both value and expected datatype matches for xarray calculation
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_as_xarray_1d = xr.DataArray([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    result = scores.continuous.mae(fcst_as_xarray_1d, obs_as_xarray_1d)
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])

    expected = xr.DataArray(0.7273)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_mae_pandas_series():
    """
    Test calculation works correctly on pandas series
    """

    fcst_pd_series = pd.Series([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    obs_pd_series = pd.Series([1, 1, 1, 2, 1, 2, 1, 1, 1, 3, 1])
    expected = 0.7273
    result = scores.continuous.mae(fcst_pd_series, obs_pd_series)
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
    assert round(result, PRECISION) == expected


def test_mae_xarray_to_point():
    """
    Test MAE calculates the correct value for a simple sequence
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    result = scores.continuous.mae(fcst_as_xarray_1d, 1)
    expected = xr.DataArray(0.9091)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_mae():
    """
    Test MAE calculates the correct value on a 2d array
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d)
    expected = xr.DataArray(8.7688)
    assert result.round(PRECISION) == expected.round(PRECISION)


def test_2d_xarray_mae_with_dimensions():
    """
    Test MAE is calculated correctly along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    result = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims="latitude")

    expected_values = [13.2397, 9.0065, 2.9662, 9.8629]
    expected_dimensions = ("longitude",)
    assert all(result.round(4) == expected_values)
    assert result.dims == expected_dimensions


def test_xarray_dimension_handling_with_arrays():
    """
    Test MAE is calculated correctly along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    reduce_lat = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims="latitude")

    preserve_lon = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, preserve_dims="longitude")

    expected_values = [13.2397, 9.0065, 2.9662, 9.8629]
    expected_dimensions = ("longitude",)
    assert all(reduce_lat.round(4) == expected_values)
    assert all(preserve_lon.round(4) == expected_values)
    assert reduce_lat.dims == expected_dimensions
    assert preserve_lon.dims == expected_dimensions


def test_xarray_dimension_preservations_with_arrays():
    """
    Test MAE is calculated correctly along a specified dimension
    """
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    fcst_temperatures_xr_2d = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
    obs_temperatures_xr_2d = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])

    reduce_empty = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, reduce_dims=[])
    preserve_all = scores.continuous.mae(fcst_temperatures_xr_2d, obs_temperatures_xr_2d, preserve_dims="all")

    assert reduce_empty.dims == fcst_temperatures_xr_2d.dims  # Nothing should be reduced
    assert (reduce_empty == preserve_all).all()