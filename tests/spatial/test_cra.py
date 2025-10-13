"""Unit tests for scores.spatial.cra_impl

These tests validate the CRA (Contiguous Rain Area) metric implementation, including
basic functionality, handling of NaNs, dataset input, and error handling
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scores.spatial.cra_impl import cra, cra_2d

THRESHOLD = 10


@pytest.fixture
def sample_data_2d():
    """2D synthetic forecast and analysis fields for cra_2d"""
    forecast = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])
    analysis = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])
    return forecast, analysis


@pytest.fixture
def sample_data_3d():
    """3D synthetic forecast and analysis fields for cra"""
    time = pd.date_range("2022-07-05", periods=3, freq="D")
    lat = np.linspace(-40, -10, 10)
    lon = np.linspace(140, 160, 10)

    forecast = xr.DataArray(
        np.random.rand(3, 10, 10) * 20,
        dims=["time", "latitude", "longitude"],
        coords={"time": ("time", time), "latitude": lat, "longitude": lon},
    )
    analysis = xr.DataArray(
        np.random.rand(3, 10, 10) * 20,
        dims=["time", "latitude", "longitude"],
        coords={"time": ("time", time), "latitude": lat, "longitude": lon},
    )
    return forecast, analysis


def test_cra_basic_output_type(sample_data_3d):
    forecast, analysis = sample_data_3d
    result = cra(forecast, analysis, THRESHOLD, y_name="latitude", x_name="longitude", reduce_dims=["time"])
    assert isinstance(result, dict)


def test_cra_2d_basic_output_type(sample_data_2d):
    """Test that CRA returns a dictionary for valid input."""
    forecast, analysis = sample_data_2d
    result = cra_2d(forecast, analysis, THRESHOLD, y_name="latitude", x_name="longitude")
    assert isinstance(result, dict), "CRA output should be a dictionary"


def test_cra_with_nans(sample_data_3d):
    """Test CRA handles NaNs in the forecast field ok"""
    forecast, analysis = sample_data_3d
    forecast[0, 0] = np.nan  # Introduce a NaN
    result = cra(forecast, analysis, THRESHOLD, y_name="latitude", x_name="longitude")

    assert isinstance(result, dict), "CRA output should be a dictionary even with NaNs"

    for key, value in result.items():
        assert value is not None, f"{key} is None"
        if isinstance(value, (float, int, np.number)):
            assert not np.isnan(value), f"{key} contains NaN"


def test_cra_dataset_input(sample_data_2d):
    """Test CRA works with xarray.Dataset input."""
    forecast, analysis = sample_data_2d
    ds = xr.Dataset({"forecast": forecast, "analysis": analysis})
    result = cra_2d(ds["forecast"], ds["analysis"], THRESHOLD, y_name="latitude", x_name="longitude")
    expected_keys = [
        "mse_total",
        "mse_displacement",
        "mse_volume",
        "mse_pattern",
        "fcst_blob",
        "obs_blob",
        "shifted_fcst",
        "optimal_shift",
        "num_gridpoints_above_threshold_fcst",
        "num_gridpoints_above_threshold_obs",
        "avg_fcst",
        "avg_obs",
        "max_fcst",
        "max_obs",
        "corr_coeff_original",
        "corr_coeff_shifted",
        "rmse_original",
        "rmse_shifted",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key in CRA output: {key}"


def test_cra_invalid_input():
    """Test CRA raises TypeError for non-xarray input."""
    with pytest.raises(TypeError, match="fcst must be an xarray DataArray"):
        cra("invalid", "input", THRESHOLD, y_name="latitude", x_name="longitude")

    valid_fcst = xr.DataArray(np.random.rand(1, 10, 10), dims=["time", "latitude", "longitude"], coords={"time": [0]})
    with pytest.raises(TypeError, match="obs must be an xarray DataArray"):
        cra(valid_fcst, "input", THRESHOLD, y_name="latitude", x_name="longitude")


def test_cra_mismatched_shapes():
    """Test CRA raises ValueError for mismatched input shapes."""
    fcst = xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"])
    obs = xr.DataArray(np.random.rand(80, 100), dims=["latitude", "longitude"])  # mismatched shape

    with pytest.raises(ValueError) as excinfo:
        cra_2d(fcst, obs, THRESHOLD, y_name="latitude", x_name="longitude")


@pytest.mark.parametrize(
    "fcst_shape, obs_shape",
    [
        ((100, 100), (80, 100)),
        ((100, 100), (100, 90)),
        ((100, 100), (99, 99)),
    ],
)
def test_cra_mismatched_shapes_parametrized(fcst_shape, obs_shape):
    """Test CRA raises ValueError for mismatched input shapes."""
    fcst = xr.DataArray(np.random.rand(*fcst_shape), dims=["latitude", "longitude"])
    obs = xr.DataArray(np.random.rand(*obs_shape), dims=["latitude", "longitude"])
    with pytest.raises(ValueError, match="fcst and obs must have the same shape"):
        cra(fcst, obs, THRESHOLD, y_name="latitude", x_name="longitude")


@pytest.mark.parametrize(
    "fcst, obs, expected_error, match_text",
    [
        ("invalid", xr.DataArray(np.random.rand(10, 10)), TypeError, "fcst must be an xarray DataArray"),
        (xr.DataArray(np.random.rand(10, 10)), "invalid", TypeError, "obs must be an xarray DataArray"),
        (
            xr.DataArray(np.random.rand(10, 10)),
            xr.DataArray(np.random.rand(8, 10)),
            ValueError,
            "fcst and obs must have the same shape",
        ),
    ],
)
def test_cra_invalid_inputs(fcst, obs, expected_error, match_text):
    """Test CRA raises appropriate errors for invalid inputs."""
    with pytest.raises(expected_error, match=match_text):
        cra(fcst, obs, THRESHOLD, y_name="latitude", x_name="longitude")


@pytest.mark.parametrize(
    "fcst, obs",
    [
        (
            xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
            xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"]),
        ),
        (
            xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"]),
            xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
        ),
        (
            xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
            xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
        ),
    ],
)
def test_cra_all_nan_inputs_warns(fcst, obs, caplog):
    """Test CRA logs a warning when forecast or observation is all NaNs."""
    with caplog.at_level("INFO"):
        result = cra_2d(fcst, obs, THRESHOLD, y_name="latitude", x_name="longitude")
    assert "Less than 10 points meet the condition." in caplog.text
    assert result is None
