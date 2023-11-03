"""
Contains unit tests for scores.continuous.standard
"""

import dask
import dask.array
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

    expected = xr.DataArray(1.0909)
    assert isinstance(result, xr.DataArray)
    assert result.round(PRECISION) == expected.round(PRECISION)


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


def test_mse_xarray_to_point():
    """
    Test MSE calculates the correct value for a simple 1d sequence
    Currently breaks type hinting but here for future pandas support
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    result = scores.continuous.mse(fcst_as_xarray_1d, 1)  # type: ignore
    expected = xr.DataArray(1.45454545)
    assert isinstance(result, xr.DataArray)
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
    assert isinstance(result, xr.DataArray)
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
    assert isinstance(result, xr.DataArray)
    assert all(result.round(4) == expected_values)  # type: ignore  # We don't want full xarray comparison, and static analysis is confused about types
    assert result.dims == expected_dimensions


# Root Mean Squared Error
@pytest.fixture
def rmse_fcst_xarray():
    return xr.DataArray([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_fcst_nan_xarray():
    return xr.DataArray([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_obs_xarray():
    return xr.DataArray([1, 1, 1, 2, 1, 2, 1, -1, 1, 3, 1])


@pytest.fixture
def rmse_fcst_pandas():
    return pd.Series([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_fcst_nan_pandas():
    return pd.Series([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_obs_pandas():
    return pd.Series([1, 1, 1, 2, 1, 2, 1, 1, -1, 3, 1])


@pytest.mark.parametrize(
    "forecast, observations, expected, request_kwargs",
    [
        ("rmse_fcst_xarray", "rmse_obs_xarray", xr.DataArray(1.3484), {}),
        (
            "rmse_fcst_xarray",
            "rmse_obs_xarray",
            xr.DataArray([2, 2, 0, 1, 1, 0, 1, 2, 0, 1, 2]),
            dict(preserve_dims="all"),
        ),
        ("rmse_fcst_nan_xarray", "rmse_obs_xarray", xr.DataArray(1.3784), {}),
        ("rmse_fcst_xarray", 1, xr.DataArray(1.3484), {}),
        ("rmse_fcst_nan_xarray", 1, xr.DataArray(1.3784), {}),
        ("rmse_fcst_pandas", "rmse_obs_pandas", 1.3484, {}),
        ("rmse_fcst_pandas", 1, 1.3484, {}),
        ("rmse_fcst_nan_pandas", "rmse_obs_pandas", 1.3784, {}),
    ],
    ids=[
        "simple-1d",
        "preserve-1d",
        "simple-1d-w-nan",
        "to-point",
        "to-point-w-nan",
        "pandas-series-1d",
        "pandas-to-point",
        "pandas-series-nan-1d",
    ],
)
def test_rmse_xarray_1d(forecast, observations, expected, request_kwargs, request):
    """
    Test RMSE for the following cases:
       * Calculates the correct value for a simple xarray 1d sequence
       * Calculates the correct value for an xarray 1d sequence preserving all
       * Calculates the correct value for a simple pandas 1d series
       * Calculates the correct value for an xarray 1d sequence comparing to a point
    """
    if isinstance(forecast, str):
        forecast = request.getfixturevalue(forecast)
    if isinstance(observations, str):
        observations = request.getfixturevalue(observations)
    result = scores.continuous.rmse(forecast, observations, **request_kwargs)
    assert (result.round(PRECISION) == expected).all()


@pytest.fixture
def rmse_2d_fcst_xarray():
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    fcst_temperatures_2d[0, 2, 1] = np.nan
    return xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])


@pytest.fixture
def rmse_2d_obs_xarray():
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    obs_temperatures_2d[0, 1, 2] = np.nan
    return xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])


@pytest.fixture
def rmse_2d_expected_xarray():
    lons = [30, 31, 32, 33]
    exp_temperatures_2d = [2.6813, 1.2275, 1.252, 2.6964]
    return xr.DataArray(exp_temperatures_2d, dims=["longitude"], coords=[lons])


@pytest.mark.parametrize(
    "forecast, observations, expected, request_kwargs, expected_dimensions",
    [
        ("rmse_2d_fcst_xarray", "rmse_2d_obs_xarray", xr.DataArray(2.1887), {}, ()),
        (
            "rmse_2d_fcst_xarray",
            "rmse_2d_obs_xarray",
            "rmse_2d_expected_xarray",
            dict(reduce_dims="latitude"),
            ("longitude",),
        ),
        (
            "rmse_2d_fcst_xarray",
            "rmse_2d_obs_xarray",
            "rmse_2d_expected_xarray",
            dict(preserve_dims="longitude"),
            ("longitude",),
        ),
    ],
    ids=["simple-2d", "reduce-2d", "preserve-2d"],
)
def test_rmse_xarray_2d_rand(forecast, observations, expected, request_kwargs, expected_dimensions, request):
    """
    Test RMSE for the following cases on 2d Data:
       * Calculates the correct value for a simple xarray 2d sequence
       * Calculates the correct value for an xarray 2d sequence reducing over set dim
       * Calculates the correct value for an xarray 2d sequence preserving set dim
    """
    if isinstance(forecast, str):
        forecast = request.getfixturevalue(forecast)
    if isinstance(observations, str):
        observations = request.getfixturevalue(observations)
    if isinstance(expected, str):
        expected = request.getfixturevalue(expected)

    result = scores.continuous.rmse(forecast, observations, **request_kwargs)
    xr.testing.assert_allclose(result.round(PRECISION), expected)
    assert result.dims == expected_dimensions


def create_xarray(data: list[list[float]]):
    npdata = np.array(data)
    lats = list(np.arange(npdata.shape[0]) + 50)
    lons = list(np.arange(npdata.shape[1]) + 30)
    return xr.DataArray(npdata, dims=["latitude", "longitude"], coords=[lats, lons])


@pytest.mark.parametrize(
    "forecast, observations, expected, request_kwargs,",
    [
        (create_xarray([[0, 0], [1, 1]]), create_xarray([[0, 0], [1, 1]]), xr.DataArray(0), {}),
        (create_xarray([[-1, 0], [1, 1]]), create_xarray([[-1, 0], [1, 1]]), xr.DataArray(0), {}),
        (create_xarray([[1, 0], [1, 1]]), create_xarray([[-1, 0], [1, 1]]), xr.DataArray(1), {}),
        (create_xarray([[-1, 0], [1, 1]]), create_xarray([[1, 0], [1, 1]]), xr.DataArray(1), {}),
        (create_xarray([[np.nan, 0], [1, 1]]), create_xarray([[1, 0], [1, 1]]), xr.DataArray(0), {}),
        (
            create_xarray([[np.nan, 1], [1, 1]]),
            create_xarray([[1, 0], [1, 1]]),
            xr.DataArray(np.sqrt(1 / 3).round(PRECISION)),
            {},
        ),
        (create_xarray([[1, 0], [1, 1]]), create_xarray([[np.nan, 0], [1, 1]]), xr.DataArray(0), {}),
        (
            create_xarray([[1, 0], [1, 1]]),
            create_xarray([[np.nan, 1], [1, 1]]),
            xr.DataArray(np.sqrt(1 / 3).round(PRECISION)),
            {},
        ),
    ],
    ids=[
        "perfect",
        "perfect-neg",
        "single",
        "single_neg",
        "perfect-nan",
        "single-nan",
        "perfect-nan-r",
        "single-nan-r",
    ],
)
def test_rmse_xarray_2d_defined(forecast, observations, expected, request_kwargs, request):
    """
    Test RMSE Values for defined edge cases

    """
    if isinstance(forecast, str):
        forecast = request.getfixturevalue(forecast)
    if isinstance(observations, str):
        observations = request.getfixturevalue(observations)
    if isinstance(expected, str):
        expected = request.getfixturevalue(expected)

    result = scores.continuous.rmse(forecast, observations, **request_kwargs)
    xr.testing.assert_allclose(result.round(PRECISION), expected)


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


def test_mae_xarray_to_point():
    """
    Test MAE calculates the correct value for a simple sequence
    Tests unhinted types but this is useful
    """
    fcst_as_xarray_1d = xr.DataArray([1, 3, 1, 3, 2, 2, 2, 1, 1, 2, 3])
    result = scores.continuous.mae(fcst_as_xarray_1d, 1)  # type: ignore
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
    assert all(result.round(4) == expected_values)  # type: ignore  # We don't want full xarray comparison, and static analysis is confused about types
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
    assert all(preserve_lon.round(4) == expected_values)  # type: ignore  # We don't want full xarray comparison, and static analysis is confused about types
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


FCST_CHUNKED = xr.DataArray(
    data=np.array([[1, 2], [3, 10]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
).chunk()
OBS_CHUNKED = xr.DataArray(
    data=np.array([[0, 0], [0, np.nan]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
).chunk()


def test_mse_with_dask():
    """
    Test that mse works with dask
    """
    result = scores.continuous.mse(FCST_CHUNKED, OBS_CHUNKED, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)  # type: ignore # Static analysis fails to recognise the type of 'result' correctly
    result = result.compute()  # type: ignore # Static analysis thinks this is a float, but it's a dask array
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[5, 4], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)


def test_mae_with_dask():
    """
    Test that mae works with dask
    """
    result = scores.continuous.mae(FCST_CHUNKED, OBS_CHUNKED, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)  # type: ignore
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[2, 2], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)


def test_rmse_with_dask():
    """
    Test that rmse works with dask
    """
    result = scores.continuous.rmse(FCST_CHUNKED, OBS_CHUNKED, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[np.sqrt(5), 2], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)
