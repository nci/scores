"""Tests for the quantile-quantile score implementation."""

import numpy as np
import pytest
import xarray as xr

from scores.plotdata import qq
from scores.utils import dask_available

HAS_DASK = dask_available()

if HAS_DASK:
    import dask.array as da
else:
    da = None

NP_INTERP_METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


@pytest.fixture
def sample_dataarray1():
    """Generate data for testing"""
    data = np.linspace(0, 99, num=100).reshape(10, 10)
    return xr.DataArray(data, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})


@pytest.fixture
def sample_dataarray2():
    """Generate data for testing"""
    data = np.linspace(0, 198, num=100).reshape(10, 10)
    return xr.DataArray(data, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})


@pytest.fixture
def sample_dataarray3():
    """Generate data for testing"""
    data = np.linspace(0, 9, num=10)
    return xr.DataArray(data, dims=("x"), coords={"x": np.arange(10)})


@pytest.fixture
def sample_dataarray4():
    """Generate data for testing"""
    data = [0, 1, np.nan, 3, 4]
    return xr.DataArray(data, dims=("x"), coords={"x": [1, 2, 3, 4, 5]})


@pytest.fixture
def sample_dataset1():
    """Generate data for testing"""
    data1 = np.linspace(0, 99, num=100).reshape(10, 10)
    da1 = xr.DataArray(data1, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})
    data2 = np.linspace(0, 198, num=100).reshape(10, 10)
    da2 = xr.DataArray(data2, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})
    ds = xr.Dataset({"var1": da1, "var2": da2})
    return ds


@pytest.fixture
def sample_dataset2():
    """Generate data for testing"""
    data1 = np.linspace(0, 198, num=100).reshape(10, 10)
    da1 = xr.DataArray(data1, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})
    data2 = np.linspace(0, 99, num=100).reshape(10, 10)
    da2 = xr.DataArray(data2, dims=("x", "y"), coords={"x": np.arange(10), "y": np.arange(10)})
    ds = xr.Dataset({"var1": da1, "var2": da2})
    return ds


@pytest.fixture
def expected_result1():
    """Expected result for testing"""
    data = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
            ],
            [
                [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
                [90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0],
                [180.0, 182.0, 184.0, 186.0, 188.0, 190.0, 192.0, 194.0, 196.0, 198.0],
            ],
        ]
    )
    da_xr = xr.DataArray(
        data,
        dims=("data_source", "quantile", "y"),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
            "y": np.arange(10),
        },
    )
    return da_xr


@pytest.fixture
def expected_result2():
    """Expected result for testing"""
    da_xr = xr.DataArray(
        np.array([[0, 49.5, 99], [0, 99, 198]]),
        dims=(
            "data_source",
            "quantile",
        ),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
        },
    )
    return da_xr


@pytest.fixture
def expected_result3():
    """Expected result for testing"""
    data = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
            ],
            [
                [0.0] * 10,
                [4.5] * 10,
                [9] * 10,
            ],
        ]
    )
    da_xr = xr.DataArray(
        data,
        dims=("data_source", "quantile", "y"),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
            "y": np.arange(10),
        },
    )
    return da_xr


@pytest.fixture
def expected_result4():
    """Expected result for testing"""
    da1 = xr.DataArray(
        np.array([[0, 99, 198], [0, 49.5, 99]]),
        dims=(
            "data_source",
            "quantile",
        ),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
        },
    )
    da2 = xr.DataArray(
        np.array([[0, 49.5, 99], [0, 99, 198]]),
        dims=(
            "data_source",
            "quantile",
        ),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
        },
    )
    ds = xr.Dataset({"var1": da2, "var2": da1})
    return ds


@pytest.fixture
def expected_result5():
    """Expected result for testing"""
    da_xr = xr.DataArray(
        np.array([[0, 2, 4], [0, 2, 4]]),
        dims=(
            "data_source",
            "quantile",
        ),
        coords={
            "data_source": ["fcst", "obs"],
            "quantile": [0.0, 0.5, 1.0],
        },
    )
    return da_xr


@pytest.mark.parametrize(
    "fcst_fixture, obs_fixture, quantiles, preserve_dims, reduce_dims, expected",
    [
        # Reduce one dimension
        ("sample_dataarray1", "sample_dataarray2", [0.0, 0.5, 1], None, "x", "expected_result1"),
        # Preserve one dimension
        ("sample_dataarray1", "sample_dataarray2", [0.0, 0.5, 1], "y", None, "expected_result1"),
        # Reduce all dimensions
        ("sample_dataarray1", "sample_dataarray2", [0.0, 0.5, 1], None, None, "expected_result2"),
        # Test broadcasting
        ("sample_dataarray1", "sample_dataarray3", [0.0, 0.5, 1], None, "x", "expected_result3"),
        # Test Dataset
        ("sample_dataset1", "sample_dataset2", [0.0, 0.5, 1], None, None, "expected_result4"),
        # Test with NaN values
        ("sample_dataarray4", "sample_dataarray4", [0.0, 0.5, 1], None, None, "expected_result5"),
    ],
)
def test_empirical_quantiles_with_varied_inputs(
    request, fcst_fixture, obs_fixture, quantiles, preserve_dims, reduce_dims, expected
):
    """Tests that qq produces the correct results"""
    fcst = request.getfixturevalue(fcst_fixture)
    obs = request.getfixturevalue(obs_fixture)
    result = qq(fcst, obs, quantiles=quantiles, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    expected = request.getfixturevalue(expected)
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize("method", NP_INTERP_METHODS)
def test_valid_interpolation_methods(method, sample_dataarray1):  # pylint: disable=redefined-outer-name
    """
    Check that all interpolation methods work. Doesn't check the correctness of the results.
    """
    result = qq(sample_dataarray1, sample_dataarray1.copy(), quantiles=[0.1, 0.5, 0.9], interpolation_method=method)
    assert isinstance(result, xr.DataArray)
    assert "data_source" in result.dims
    assert result.sizes["data_source"] == 2


def test_invalid_interpolation_method(sample_dataarray1):  # pylint: disable=redefined-outer-name
    """
    Tests that an error is raised if an invalid interpolation method is used
    """
    with pytest.raises(ValueError, match="Invalid interpolation method"):
        qq(sample_dataarray1, sample_dataarray1, quantiles=[0.1, 0.5], interpolation_method="invalid_method")


@pytest.mark.parametrize("quantiles", [[-0.01, 0.5], [0.5, 1.01]])
def test_invalid_quantiles(quantiles, sample_dataarray1):  # pylint: disable=redefined-outer-name
    """
    Tests that an error is raised if values outside of [0, 1] are passed into the quantile arg
    """
    with pytest.raises(ValueError, match="Quantiles must be in the range"):
        qq(sample_dataarray1, sample_dataarray1, quantiles=quantiles)


def test_disallowed_data_source_dim(sample_dataarray1):  # pylint: disable=redefined-outer-name
    """
    Tests that an error is raised if a dimension is named 'data_source'
    """
    da_xr = sample_dataarray1.expand_dims("data_source")
    with pytest.raises(ValueError, match="Dimensions named 'data_source'"):
        qq(da_xr, sample_dataarray1, quantiles=[0.1, 0.5])


def test_mismatched_dataset_variables():
    """
    Tests that an error is raised when Datasets have different data vars
    """
    ds1 = xr.Dataset({"var1": ("time", np.random.rand(10))})
    ds2 = xr.Dataset({"var2": ("time", np.random.rand(10))})
    with pytest.raises(ValueError, match="must contain the same variables"):
        qq(ds1, ds2, quantiles=[0.5])


def test_type_mismatch(sample_dataarray1, sample_dataset1):  # pylint: disable=redefined-outer-name
    """
    Tests that an error is raised when both xr.DataArrays and xr.Datasets are
    passed in at the same time
    """
    with pytest.raises(TypeError, match="must be either xarray DataArrays or xarray Datasets"):
        qq(sample_dataarray1, sample_dataset1, quantiles=[0.5])


def test_all_dims_preserved_error(sample_dataarray1, sample_dataarray2):  # pylint: disable=redefined-outer-name
    """
    Test that when 'all' is specified for preserve_dims, the result has the same dimensions as the forecast.
    """
    with pytest.raises(ValueError, match="You cannot preserve all dimensions with qq."):
        qq(sample_dataarray1, sample_dataarray2, quantiles=[0.1, 0.5, 0.9], preserve_dims="all")


def test_empirical_qq_dask(sample_dataarray4, expected_result5):  # pylint: disable=redefined-outer-name
    """
    Tests continuous.qq works with dask
    """
    if not HAS_DASK:  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover
    result = qq(sample_dataarray4.chunk(), sample_dataarray4.chunk(), quantiles=[0, 0.5, 1])
    assert isinstance(result.data, da.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, expected_result5)
