"""
Contains unit tests for scores.continuous.standard
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=R0801

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore # pylint: disable=invalid-name  # pragma: no cover

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
    """Creates forecast Xarray for test."""
    return xr.DataArray([-1, 3, 1, 3, 0, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_fcst_nan_xarray():
    """Creates forecast Xarray containing NaNs for test."""
    return xr.DataArray([-1, 3, 1, 3, np.nan, 2, 2, 1, 1, 2, 3])


@pytest.fixture
def rmse_obs_xarray():
    """Creates observation Xarray for test."""
    return xr.DataArray([1, 1, 1, 2, 1, 2, 1, -1, 1, 3, 1])


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
        ("rmse_fcst_xarray", "rmse_obs_xarray", xr.DataArray(1.3484), {}),
        (
            "rmse_fcst_xarray",
            "rmse_obs_xarray",
            xr.DataArray([2, 2, 0, 1, 1, 0, 1, 2, 0, 1, 2]),
            {"preserve_dims": "all"},
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
    if not isinstance(result, float):
        assert (result.round(PRECISION) == expected).all()
    else:
        assert np.round(result, PRECISION) == expected


@pytest.fixture
def rmse_2d_fcst_xarray():
    """Creates 2D forecast Xarray for test."""
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
    fcst_temperatures_2d[0, 2, 1] = np.nan
    return xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])


@pytest.fixture
def rmse_2d_obs_xarray():
    """Creates 2D observation Xarray for test."""
    numpy.random.seed(0)
    lats = [50, 51, 52, 53]
    lons = [30, 31, 32, 33]
    obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
    obs_temperatures_2d[0, 1, 2] = np.nan
    return xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])


@pytest.fixture
def rmse_2d_expected_xarray():
    """Creates 2D forecast Xarray to be used as expected result for test."""
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
            {"reduce_dims": "latitude"},
            ("longitude",),
        ),
        (
            "rmse_2d_fcst_xarray",
            "rmse_2d_obs_xarray",
            "rmse_2d_expected_xarray",
            {"preserve_dims": "longitude"},
            ("longitude",),
        ),
    ],
    ids=["simple-2d", "reduce-2d", "preserve-2d"],
)
def test_rmse_xarray_2d_rand(  # pylint: disable=too-many-arguments
    forecast, observations, expected, request_kwargs, expected_dimensions, request
):
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
    """Creates an Xarray.DataArray to be used in the tests"""
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


# Dask tests
def test_mse_with_dask():
    """
    Test that mse works with dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst_chunked = xr.DataArray(
        data=np.array([[1, 2], [3, 10]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()
    obs_chunked = xr.DataArray(
        data=np.array([[0, 0], [0, np.nan]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()

    result = scores.continuous.mse(fcst_chunked, obs_chunked, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)  # type: ignore # Static analysis fails to recognise the type of 'result' correctly
    result = result.compute()  # type: ignore # Static analysis thinks this is a float, but it's a dask array
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[5, 4], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)


def test_mae_with_dask():
    """
    Test that mae works with dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst_chunked = xr.DataArray(
        data=np.array([[1, 2], [3, 10]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()
    obs_chunked = xr.DataArray(
        data=np.array([[0, 0], [0, np.nan]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()

    result = scores.continuous.mae(fcst_chunked, obs_chunked, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)  # type: ignore
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[2, 2], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)


def test_rmse_with_dask():
    """
    Test that rmse works with dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst_chunked = xr.DataArray(
        data=np.array([[1, 2], [3, 10]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()
    obs_chunked = xr.DataArray(
        data=np.array([[0, 0], [0, np.nan]]), dims=["dim1", "dim2"], coords={"dim1": [1, 2], "dim2": [1, 2]}
    ).chunk()

    result = scores.continuous.rmse(fcst_chunked, obs_chunked, reduce_dims="dim1")
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    expected = xr.DataArray(data=[np.sqrt(5), 2], dims=["dim2"], coords={"dim2": [1, 2]})
    xr.testing.assert_equal(result, expected)


# Angular / directional tests
DA1_ANGULAR = xr.DataArray([[10, 10], [90, 90]], coords=[[0, 1], [0, 1]], dims=["i", "j"])
DA2_ANGULAR = xr.DataArray([[350, 180], [270, 280]], coords=[[0, 1], [0, 1]], dims=["i", "j"])


def test_mse_angular():
    """Tests that `mse` returns the expected object with `is_angular` is True"""

    expected = xr.DataArray(
        [[20**2, 170**2], [180**2, 170**2]],
        coords=[[0, 1], [0, 1]],
        dims=["i", "j"],
        name="mean_squared_error",
    )

    result = scores.continuous.mse(DA1_ANGULAR, DA2_ANGULAR, preserve_dims=["i", "j"], is_angular=True)

    xr.testing.assert_equal(result, expected)


def test_mae_angular():
    """Tests that `mae` returns the expected object with `is_angular` is True"""

    expected = xr.DataArray(
        [[20, 170], [180, 170]],
        coords=[[0, 1], [0, 1]],
        dims=["i", "j"],
        name="mean_squared_error",
    )

    result = scores.continuous.mae(DA1_ANGULAR, DA2_ANGULAR, preserve_dims=["i", "j"], is_angular=True)

    xr.testing.assert_equal(result, expected)


def test_rmse_angular():
    """Tests that `rmse` returns the expected object with `is_angular` is True"""

    expected = xr.DataArray(
        [((20**2 + 170**2) / 2) ** 0.5, ((180**2 + 170**2) / 2) ** 0.5],
        coords={"i": [0, 1]},
        dims=["i"],
        name="mean_squared_error",
    )

    result = scores.continuous.rmse(DA1_ANGULAR, DA2_ANGULAR, preserve_dims=["i"], is_angular=True)

    xr.testing.assert_equal(result, expected)


DA1_CORR = xr.DataArray(
    np.array([[1, 2, 3], [0, 1, 0], [0.5, -0.5, 0.5], [3, 6, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA2_CORR = xr.DataArray(
    np.array([[2, 4, 6], [6, 5, 6], [3, 4, 5], [3, np.nan, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA3_CORR = xr.DataArray(
    np.array([[1, 2, 3], [3, 2.5, 3], [1.5, 2, 2.5], [1.5, np.nan, 1.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA4_CORR = xr.DataArray(
    np.array([[1, 3, 7], [2, 2, 8], [3, 1, 7]]),
    dims=("space", "time"),
    coords=[
        ("space", ["x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA5_CORR = xr.DataArray(
    np.array([1, 2, 3]),
    dims=("space"),
    coords=[("space", ["x", "y", "z"])],
)

EXP_CORR_KEEP_SPACE_DIM = xr.DataArray(
    np.array([1.0, -1.0, 0.0, np.nan]),
    dims=("space"),
    coords=[("space", ["w", "x", "y", "z"])],
)

EXP_CORR_REDUCE_ALL = xr.DataArray(1.0)

EXP_CORR_DIFF_SIZE = xr.DataArray(
    np.array([1.0, -1.0, 0.0]),
    dims=("time"),
    coords=[("time", [1, 2, 3])],
)


@pytest.mark.parametrize(
    ("da1", "da2", "reduce_dims", "preserve_dims", "expected"),
    [
        # Check reduce dim arg
        (DA1_CORR, DA2_CORR, None, "space", EXP_CORR_KEEP_SPACE_DIM),
        # Check preserve dim arg
        (DA1_CORR, DA2_CORR, "time", None, EXP_CORR_KEEP_SPACE_DIM),
        # Check reduce all
        (DA3_CORR, DA2_CORR, None, None, EXP_CORR_REDUCE_ALL),
        # Check different size arrays as input
        (DA4_CORR, DA5_CORR, "space", None, EXP_CORR_DIFF_SIZE),
    ],
)
def test_correlation(da1, da2, reduce_dims, preserve_dims, expected):
    """
    Tests continuous.correlation
    """
    result = scores.continuous.correlation(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    xr.testing.assert_allclose(result, expected)


def test_correlation_dask():
    """
    Tests continuous.correlation works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = scores.continuous.correlation(DA3_CORR.chunk(), DA2_CORR.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, EXP_CORR_REDUCE_ALL)


DA1_BIAS = xr.DataArray(
    np.array([[1, 1, np.nan], [0, 0, 0], [0.5, -0.5, 0.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y"]),
        ("time", [1, 2, 3]),
    ],
)

DA2_BIAS = xr.DataArray(
    np.array([[2, 2, 6], [2, 10, 0], [-0.5, 0.5, -0.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y"]),
        ("time", [1, 2, 3]),
    ],
)

DA3_BIAS = xr.DataArray(
    np.array([[2, 2, 6], [2, 10, 0], [2, 0.5, -0.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y"]),
        ("time", [1, 2, 3]),
    ],
)
BIAS_WEIGHTS = xr.DataArray(
    np.array([[1, 1, 1], [3, 0, 0], [3, 0, 0]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y"]),
        ("time", [1, 2, 3]),
    ],
)

EXP_BIAS1 = xr.DataArray(
    np.array([-1, -4, 1 / 3]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_BIAS2 = xr.DataArray(
    np.array([-1, -2, 1]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_BIAS3 = xr.DataArray(np.array(-1.625))

EXP_BIAS4 = xr.DataArray(
    np.array([1 / 2, 0, -1]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_BIAS5 = xr.DataArray(
    np.array([2, np.inf, -1]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_BIAS6 = xr.DataArray(
    np.array([1 / 2, 0, 1 / 4]),
    dims=("space"),
    coords=[
        ("space", ["w", "x", "y"]),
    ],
)
EXP_BIAS7 = xr.DataArray(np.array((2.5 / 8) / (15.5 / 8)))

DS_BIAS1 = xr.Dataset({"a": DA1_BIAS, "b": DA2_BIAS})
DS_BIAS2 = xr.Dataset({"a": DA2_BIAS, "b": DA1_BIAS})
EXP_DS_BIAS1 = xr.Dataset({"a": EXP_BIAS1, "b": -EXP_BIAS1})
EXP_DS_BIAS2 = xr.Dataset({"a": EXP_BIAS4, "b": EXP_BIAS5})


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        # Check reduce dim arg
        (DA1_BIAS, DA2_BIAS, None, "space", None, EXP_BIAS1),
        # Check weighting works
        (DA1_BIAS, DA2_BIAS, None, "space", BIAS_WEIGHTS, EXP_BIAS2),
        # Check preserve dim arg
        (DA1_BIAS, DA2_BIAS, "time", None, None, EXP_BIAS1),
        # Reduce all
        (DA1_BIAS, DA2_BIAS, None, None, None, EXP_BIAS3),
        # Test with Dataset
        (DS_BIAS1, DS_BIAS2, None, "space", None, EXP_DS_BIAS1),
    ],
)
def test_additive_bias(fcst, obs, reduce_dims, preserve_dims, weights, expected):
    """
    Tests continuous.additive_bias
    Also tests mean_error (which is an identical function)
    """
    result = scores.continuous.additive_bias(
        fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )
    result2 = scores.continuous.mean_error(
        fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )
    xr.testing.assert_equal(result, expected)
    xr.testing.assert_equal(result, result2)


def test_additive_bias_dask():
    """
    Tests that continuous.additive_bias works with Dask
    """
    fcst = DA1_BIAS.chunk()
    obs = DA2_BIAS.chunk()
    weights = BIAS_WEIGHTS.chunk()
    result = scores.continuous.additive_bias(fcst, obs, preserve_dims="space", weights=weights)
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_equal(result, EXP_BIAS2)


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        # Check reduce dim arg
        (DA1_BIAS, DA2_BIAS, None, "space", None, EXP_BIAS4),
        # Check divide by zero returns a np.inf
        (DA2_BIAS, DA1_BIAS, None, "space", None, EXP_BIAS5),
        # Check weighting works
        (DA1_BIAS, DA3_BIAS, None, "space", BIAS_WEIGHTS, EXP_BIAS6),
        # # Check preserve dim arg
        (DA1_BIAS, DA2_BIAS, "time", None, None, EXP_BIAS4),
        # Reduce all
        (DA1_BIAS, DA2_BIAS, None, None, None, EXP_BIAS7),
        # Test with Dataset
        (DS_BIAS1, DS_BIAS2, None, "space", None, EXP_DS_BIAS2),
    ],
)
def test_multiplicative_bias(fcst, obs, reduce_dims, preserve_dims, weights, expected):
    """
    Tests continuous.multiplicative_bias
    """
    result = scores.continuous.multiplicative_bias(
        fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )
    xr.testing.assert_equal(result, expected)


def test_multiplicative_bias_dask():
    """
    Tests that continuous.multiplicative_bias works with Dask
    """
    fcst = DA1_BIAS.chunk()
    obs = DA3_BIAS.chunk()
    weights = BIAS_WEIGHTS.chunk()
    result = scores.continuous.multiplicative_bias(fcst, obs, preserve_dims="space", weights=weights)
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_equal(result, EXP_BIAS6)
