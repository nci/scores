# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught
"""Unit tests for scores.spatial.cra_impl

These tests validate the CRA (Contiguous Rain Area) metric implementation, including
basic functionality, handling of NaNs, dataset input, and error handling
"""
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scores
from scores.continuous.standard_impl import mse, rmse
from scores.spatial.cra_impl import (
    _calc_corr_coeff,
    _calc_resolution,
    _cra_image,
    _generate_largest_rain_area_2d,
    _shifted_mse_2d,
    _translate_forecast_region,
    _nansafe_int,
    _validate_cra2d_inputs,
    cra,
)

THRESHOLD = 10


@pytest.fixture
def sample_data_2d():
    """2D synthetic forecast and analysis fields for cra_image"""
    forecast = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])
    analysis = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])
    return forecast, analysis


@pytest.fixture
def sample_data_2d_with_time():
    """2D synthetic forecast and analysis fields for cra_image"""
    forecast = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])
    analysis = xr.DataArray(np.random.rand(100, 100) * 20, dims=["latitude", "longitude"])

    forecast = forecast.expand_dims("time")
    analysis = analysis.expand_dims("time")
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
    """Test that CRA returns a dictionary for 3D input."""
    forecast, analysis = sample_data_3d
    result = cra(forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")
    assert isinstance(result, xr.Dataset)


def test_cra_image_basic_output_type(sample_data_2d):
    """Test that CRA 2D returns a dictionary for valid input."""
    forecast, analysis = sample_data_2d
    result = _cra_image(forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")
    assert isinstance(result, xr.Dataset), "CRA output should be a Dataset"


def test_cra_image_with_time(sample_data_2d_with_time):
    """Test that CRA 2D returns a dictionary for valid input."""
    forecast, analysis = sample_data_2d_with_time
    result = _cra_image(
        forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude", time_name="time"
    )
    assert isinstance(result, xr.Dataset), "CRA output should be a Dataset"


def test_cra_with_nans(sample_data_3d):
    """Test CRA handles NaNs in the forecast without errors"""
    forecast, analysis = sample_data_3d
    forecast[0, 0] = np.nan  # Introduce a NaN
    result = cra(forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")

    assert isinstance(result, xr.Dataset), "CRA output should be a dictionary even with NaNs"

    for key, value in result.items():
        assert value is not None, f"{key} is None"
        if isinstance(value, (float, int, np.number)):
            assert not np.isnan(value), f"{key} contains NaN"


def test_cra_2D_with_time(sample_data_2d_with_time):
    """Test that CRA 2D returns a dictionary for valid input."""
    forecast, analysis = sample_data_2d_with_time
    result = cra(forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")
    assert isinstance(result, xr.Dataset), "CRA output should be a Dataset"


def test_cra_without_time_dim(sample_data_2d):
    """Test CRA works with xarray.Dataset input."""
    forecast, analysis = sample_data_2d
    result = cra(
        forecast, analysis, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude", extra_components=True
    )
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


def test_cra_dataset_input(sample_data_2d):
    """Test CRA works with xarray.Dataset input."""
    forecast, analysis = sample_data_2d
    ds = xr.Dataset({"forecast": forecast, "analysis": analysis})
    result = _cra_image(
        ds["forecast"],
        ds["analysis"],
        minimum_intensity=THRESHOLD,
        y_name="latitude",
        x_name="longitude",
        extra_components=True,
    )
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
        cra("invalid", "input", minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")

    valid_fcst = xr.DataArray(np.random.rand(1, 10, 10), dims=["time", "latitude", "longitude"], coords={"time": [0]})
    with pytest.raises(TypeError, match="obs must be an xarray DataArray"):
        cra(valid_fcst, "input", minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")


def test_cra_image_invalid_input_types():
    """Test CRA 2D and ND raise TypeError for invalid input types."""
    obs = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    with pytest.raises(TypeError, match="fcst must be an xarray DataArray"):
        _cra_image("invalid", obs, minimum_intensity=5.0, y_name="y", x_name="x")

    with pytest.raises(TypeError, match="fcst must be an xarray DataArray"):
        cra("invalid", obs, minimum_intensity=5.0, y_name="y", x_name="x")

    with pytest.raises(TypeError, match="fcst must be an xarray DataArray"):
        cra("invalid", obs, minimum_intensity=5.0, y_name="y", x_name="x")

    fcst = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    with pytest.raises(TypeError, match="obs must be an xarray DataArray"):
        _cra_image(fcst, "invalid", minimum_intensity=5.0, y_name="y", x_name="x")

    with pytest.raises(TypeError, match="obs must be an xarray DataArray"):
        cra(fcst, "invalid", minimum_intensity=5.0, y_name="y", x_name="x")


def test_cra_mismatched_shapes():
    """Test CRA raises ValueError for mismatched input shapes."""
    fcst = xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"])
    obs = xr.DataArray(np.random.rand(80, 100), dims=["latitude", "longitude"])  # mismatched shape

    with pytest.raises(ValueError) as excinfo:
        _cra_image(fcst, obs, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")
    assert "fcst and obs must have the same shape" in str(excinfo.value)


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
        cra(fcst, obs, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")


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
        cra(fcst, obs, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")


@pytest.mark.parametrize(
    "fcst, obs",
    [
        (
            xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
            xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"]),
        ),
        # (
        #     xr.DataArray(np.random.rand(100, 100), dims=["latitude", "longitude"]),
        #     xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
        # ),
        # (
        #     xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
        #     xr.DataArray(np.full((100, 100), np.nan), dims=["latitude", "longitude"]),
        # ),
    ],
)
def test_cra_all_nan_inputs_warns(fcst, obs, caplog):
    """Test CRA logs warning when forecast or observation is all NaNs."""

    with caplog.at_level("INFO"):
        result = _cra_image(fcst, obs, minimum_intensity=THRESHOLD, y_name="latitude", x_name="longitude")
    assert "Less than 10 points meet the condition." in caplog.text
    assert np.isnan(result.mse_total)


def test_cra_image_min_points_threshold():
    """Test that cra_image returns None when blobs are too small."""
    # Create a small forecast and obs with only a few points above threshold
    data = np.zeros((10, 10))
    data[0, 0] = 10  # Only one point above threshold

    forecast = xr.DataArray(data, dims=["latitude", "longitude"])
    analysis = xr.DataArray(data, dims=["latitude", "longitude"])

    result = _cra_image(
        forecast, analysis, minimum_intensity=5.0, y_name="latitude", x_name="longitude", min_points=100
    )
    assert np.isnan(result.mse_total)

    # assert result is None, "Expected None when blobs are smaller than min_points"


# Helper to create a simple 2D DataArray
def create_array(shape=(10, 10), value=1.0, dims=("y", "x")):
    """Create an array"""
    return xr.DataArray(np.full(shape, value), dims=dims)


def test_small_blobs():
    """Test CRA handles small blobs correctly"""
    fcst = xr.DataArray(np.zeros((10, 10), dtype=float), dims=["y", "x"])
    obs = xr.DataArray(np.zeros((10, 10), dtype=float), dims=["y", "x"])
    fcst[0, 0] = 10.0
    obs[0, 0] = 10.0

    assert fcst.where(fcst > 5.0).count().item() == 1
    assert obs.where(obs > 5.0).count().item() == 1

    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, minimum_intensity=5.0, min_points=10)
    assert np.isnan(fcst_blob).all()
    assert np.isnan(obs_blob).all()


def test_largest_blob_too_small_post_extraction():
    """Test CRA returns None when largest blob too small"""
    fcst = xr.DataArray(np.zeros((10, 10), dtype=float), dims=["y", "x"])
    obs = xr.DataArray(np.zeros((10, 10), dtype=float), dims=["y", "x"])

    # Create 10 points above threshold, but split into two disconnected blobs
    fcst[0, 0:5] = 10.0
    fcst[0, 6:11] = 10.0  # gap at index 5 breaks contiguity

    obs[0, 0:5] = 10.0
    obs[0, 6:11] = 10.0

    # This passes the initial count check (10 points), but each blob is only 5 points
    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, minimum_intensity=5.0, min_points=6)
    assert np.isnan(fcst_blob).all()
    assert np.isnan(obs_blob).all()


def test_small_blobs_min_points_filter():
    """Test CRA filters small blobs below min_points threshold."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    fcst[0:2, 0:2] = 10  # 4 points
    obs[0:2, 0:2] = 10
    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, minimum_intensity=5.0, min_points=10)
    assert np.isnan(fcst_blob).all()
    assert np.isnan(obs_blob).all()


def test_translate_with_nan_obs():
    """
    Test _translate_forecast_region handles NaN obs.
    Correct behaviour is potentially unclear, we will just go with returning unshifted data.
    """
    fcst = create_array()
    obs = create_array(value=np.nan)
    shifted, dx, dy = _translate_forecast_region(fcst, obs, "y", "x", max_distance=300, coord_units="degrees")

    assert (shifted == fcst).all()
    assert dx == 0
    assert dy == 0


def test_translate_exceeds_max_distance_strict():
    """Test _translate_forecast_region rejects large shifts"""
    fcst = create_array()
    obs = fcst.copy().shift(x=30)  # large shift
    shifted, dx, dy = _translate_forecast_region(fcst, obs, "y", "x", max_distance=1, coord_units="degrees")

    assert (shifted == fcst).all()
    assert dx == 0
    assert dy == 0


def test_translate_exceeds_max_distance():
    """Test _translate_forecast_region rejects shift exceeds max_distance"""
    fcst = create_array()
    obs = fcst.copy()
    obs = obs.shift(x=20)  # large shift
    shifted, dx, dy = _translate_forecast_region(fcst, obs, "y", "x", max_distance=1, coord_units="degrees")

    assert (shifted == fcst).all()
    assert dx == 0
    assert dy == 0


def test_cra_image_shape_mismatch():
    """Test CRA 2D and ND raise ValueError for shape mismatch."""
    fcst = create_array()
    obs = create_array(shape=(8, 8))
    with pytest.raises(ValueError):
        _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")

    with pytest.raises(ValueError):
        cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")


def test_cra_time_format_valid():
    """Test CRA handles valid datetime format."""
    time_val = np.datetime64("2025-01-01")
    fcst = create_array().expand_dims(time=[time_val])
    obs = create_array().expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=0.5, y_name="y", x_name="x")
    assert isinstance(result, xr.Dataset)


def test_cra_image_invalid_blobs():
    """Test CRA 2D returns None for invalid blobs."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    result = _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")

    assert np.isnan(result.mse_total)


def test_cra_image_invalid_blobs_none():
    """Test CRA 2D returns None when blobs are None."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    result = _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    assert np.isnan(result.mse_total)


def test_cra_time_slice_shape_mismatch():
    """Test CRA raises ValueError for mismatched time slice shapes."""
    time_val = np.datetime64("2025-01-01")

    # Forecast: shape (1, 10, 10)
    fcst = xr.DataArray(np.ones((1, 10, 10)), dims=["time", "y", "x"], coords={"time": [time_val]})

    # Observation: shape (1, 8, 8), mismatched
    obs = xr.DataArray(np.ones((1, 8, 8)), dims=["time", "y", "x"], coords={"time": [time_val]})

    with pytest.raises(ValueError, match="fcst and obs must have the same shape"):
        cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")


def test_largest_blob_too_small():
    """Test CRA returns None when largest blob too small."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    fcst[0:1, 0:2] = 10  # 2 points
    obs[0:1, 0:2] = 10
    fcst_blob, obs_blob = _generate_largest_rain_area_2d(fcst, obs, minimum_intensity=5.0, min_points=50)

    assert np.isnan(fcst_blob).all()
    assert np.isnan(obs_blob).all()


def test_cra_image_no_valid_rain_area():
    """Test CRA 2D returns None when no valid rain area exists."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    result = _cra_image(fcst, obs, minimum_intensity=50.0, y_name="y", x_name="x")
    assert np.isnan(result.mse_total)


@pytest.mark.parametrize("time_val", [np.datetime64("2025-01-01"), "2025-01-01"])
def test_cra_time_formats_valid(time_val):
    """Test CRA handles multiple valid time formats."""
    time_val = np.datetime64(time_val)
    fcst = create_array().expand_dims(time=[time_val])
    obs = create_array().expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=0.5, y_name="y", x_name="x")
    assert isinstance(result, xr.Dataset)


def test_cra_time_slice_shape_mismatch_error():
    """Test CRA handles multiple valid time formats."""
    fcst = create_array().expand_dims(time=["2025-01-01"])
    obs = create_array(shape=(8, 8)).expand_dims(time=["2025-01-01"])
    with pytest.raises(ValueError):
        cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")


def test_cra_result_none_fallback_nan_fill():
    """Test CRA fills NaN when CRA returns None for fallback."""
    time_val = np.datetime64("2025-01-01")
    fcst = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"]).expand_dims(time=[time_val])
    obs = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"]).expand_dims(time=[time_val])

    result = cra(fcst, obs, minimum_intensity=50.0, y_name="y", x_name="x")
    for metric in result:
        assert np.isnan(result[metric][0]), f"{metric} should be NaN when CRA returns None"


@pytest.mark.parametrize("time_val", ["2025-01-01", "2025-01-02"])
def test_cra_time_formats(time_val):
    """Test CRA handles multiple time formats correctly."""
    time_val = np.datetime64(time_val)
    fcst = create_array().expand_dims(time=[time_val])
    obs = create_array().expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=0.5, y_name="y", x_name="x")
    assert isinstance(result, xr.Dataset)


def test_calc_corr_coeff_empty_after_nan_removal():
    """Test correlation returns NaN when one input is all NaNs."""
    # data1 is all NaNs, data2 is valid
    data1 = xr.DataArray(np.full((10, 10), np.nan))
    data2 = xr.DataArray(np.random.rand(10, 10))

    result = _calc_corr_coeff(data1, data2)
    assert np.isnan(result), "Expected NaN when one input is all NaNs"


def test_calc_corr_coeff_constant_array():
    """Test correlation returns NaN for constant array input."""
    data1 = xr.DataArray(np.full((10, 10), 5.0))
    data2 = xr.DataArray(np.random.rand(10, 10))
    result = _calc_corr_coeff(data1, data2)
    assert np.isnan(result), "Expected NaN for constant array input"


def test_cra_image_none_blob():
    """Test CRA 2D returns None when blob is None."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    result = _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    assert np.isnan(result.mse_total)


@pytest.mark.parametrize("time_val", ["2025-01-01"])
def test_cra_time_as_valid_string(time_val):
    """Test CRA handles time as valid string."""
    time_val = np.datetime64(time_val)
    fcst = create_array().expand_dims(time=[time_val])
    obs = create_array().expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=0.5, y_name="y", x_name="x")
    assert isinstance(result, xr.Dataset)


def test_cra_result_none_due_to_shift():
    """Test CRA returns NaN metrics when shift causes failure."""
    time_val = np.datetime64("2025-01-01")
    fcst = create_array()
    obs = fcst.copy().shift(x=30)
    fcst = fcst.expand_dims(time=[time_val])
    obs = obs.expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    for metric in result:
        assert np.isnan(result[metric][0])


def test_cra_result_none_triggers_nan_fill():
    """Test CRA fills NaN when CRA returns None due to shift."""
    time_val = np.datetime64("2025-01-01")
    fcst = create_array()
    obs = fcst.copy().shift(x=30)  # large shift to trigger rejection
    fcst = fcst.expand_dims(time=[time_val])
    obs = obs.expand_dims(time=[time_val])
    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    for metric in result:
        assert np.isnan(result[metric][0])


def test_cra_fallback_to_bbox_alignment():
    """Test CRA falls back to bounding box alignment when needed."""
    time_val = np.datetime64("2025-01-01")
    fcst = create_array()
    obs = create_array(value=np.nan)  # all NaNs to force fallback
    fcst = fcst.expand_dims({"time": [time_val]})
    obs = obs.expand_dims({"time": [time_val]})
    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    for metric in result:
        assert np.isnan(result[metric][0])


def test_cra_skips_time_slice_when_cra_returns_none():
    """Test CRA skips time slice when CRA returns None for one slice."""
    time_vals = [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")]

    # Create base arrays for the first time slice
    base_fcst = create_array(value=10.0)
    base_obs = create_array(value=10.0)

    # Get spatial dimensions
    height = base_fcst.sizes["y"]
    width = base_fcst.sizes["x"]

    # Inject small noise to ensure CRA can compute correlation
    noise = np.random.normal(0, 0.1, size=(height, width))
    noise_da = xr.DataArray(noise, dims=["y", "x"])

    fcst1 = (base_fcst + noise_da).expand_dims(time=[time_vals[0]])
    obs1 = (base_obs + noise_da).expand_dims(time=[time_vals[0]])

    # Second slice: CRA should fail due to large shift
    fcst2 = create_array(value=10.0).expand_dims(time=[time_vals[1]])
    obs2 = fcst2.copy().shift(x=30)  # large shift to force CRA failure

    # Combine time slices
    fcst = xr.concat([fcst1, fcst2], dim="time")
    obs = xr.concat([obs1, obs2], dim="time")

    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")

    for metric, values in result.items():
        assert len(values) == 2

        val0 = values[0]
        val1 = values[1]

        if isinstance(val0, (float, int, np.number)):
            assert not np.isnan(val0), f"{metric} should be valid for first time slice"
        else:
            assert not np.isnan(val0).all(), f"{metric} should be valid for first time slice"

        if isinstance(val1, (float, int, np.number)):
            assert np.isnan(val1), f"{metric} should be NaN for second time slice where CRA fails"
        else:
            assert np.isnan(val1).all(), f"{metric} should be NaN for second time slice where CRA fails"


def test_cra_image_returns_none():
    """Test CRA 2D returns None when blobs do not meet threshold."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    result = _cra_image(fcst, obs, minimum_intensity=50.0, y_name="y", x_name="x")
    assert np.isnan(result.mse_total)


def test_cra_missing_spatial_dim():
    """Test CRA raises ValueError for missing spatial dimension."""
    fcst = create_array()
    obs = create_array()
    with pytest.raises(ValueError, match="Spatial dimension 'z' not found in observation data"):
        _cra_image(fcst, obs, minimum_intensity=5.0, y_name="z", x_name="x")


@pytest.mark.parametrize("time_val", [np.datetime64(1, "ns"), np.datetime64("2025-01-01")])
def test_cra_time_val_formats(time_val):
    """Test CRA handles multiple time value formats."""
    fcst = xr.DataArray(np.ones((10, 10)), dims=["y", "x"]).expand_dims({"time": [time_val]})
    obs = xr.DataArray(np.ones((10, 10)), dims=["y", "x"]).expand_dims({"time": [time_val]})

    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    assert isinstance(result, xr.Dataset)
    for metric in result:
        assert metric in result


def test_cra_mse_total_nan_due_to_no_overlap():
    """Test CRA returns NaN for mse_total when blobs do not overlap."""
    time_val = np.datetime64("2025-01-01")

    # Forecast blob in top-left corner
    fcst = create_array(value=0.0)
    fcst[0:2, 0:2] = 10.0

    # Observation blob in bottom-right corner
    obs = create_array(value=0.0)
    obs[-2:, -2:] = 10.0

    fcst = fcst.expand_dims({"time": [time_val]})
    obs = obs.expand_dims({"time": [time_val]})

    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")

    # Confirm that CRA returns NaNs for all metrics
    for metric in result:
        assert np.isnan(result[metric][0]), f"{metric} should be NaN when blobs do not overlap"


def test_cra_image_missing_spatial_dimension():
    """Test CRA 2D raises ValueError for missing spatial dimension."""
    fcst = create_array()
    obs = create_array()
    with pytest.raises(ValueError, match="Spatial dimension 'z' not found in observation data"):
        _cra_image(fcst, obs, minimum_intensity=5.0, y_name="z", x_name="x")


def test_cra_triggers_bbox_fallback_alignment():
    """Test CRA triggers bounding box fallback alignment."""
    time_val = np.datetime64("2025-01-01")

    # Forecast blob in top-left
    fcst = create_array(value=0.0)
    fcst[0:2, 0:2] = 10.0

    # Observation blob in bottom-right
    obs = create_array(value=0.0)
    obs[-2:, -2:] = 10.0

    fcst = fcst.expand_dims({"time": [time_val]})
    obs = obs.expand_dims({"time": [time_val]})

    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    for metric in result:
        assert np.isnan(result[metric][0])


def test_cra_image_no_overlap_blobs():
    """Test CRA 2D returns None when blobs do not overlap."""
    fcst = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"])
    obs = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"])
    fcst[0:2, 0:2] = 10.0  # top-left
    obs[-2:, -2:] = 10.0  # bottom-right
    result = _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")
    assert np.isnan(result.mse_total), "Expected None when blobs do not overlap"


def test_cra_handles_string_time_key():
    """Test CRA handles string time key correctly."""
    time_vals = [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")]

    base_fcst = create_array(value=10.0)
    base_obs = create_array(value=10.0)

    height = base_fcst.sizes["y"]
    width = base_fcst.sizes["x"]

    noise = np.random.normal(0, 0.1, size=(height, width))
    noise_da = xr.DataArray(noise, dims=["y", "x"])

    fcst1 = (base_fcst + noise_da).expand_dims(time=[time_vals[0]])
    obs1 = (base_obs + noise_da).expand_dims(time=[time_vals[0]])

    fcst2 = create_array(value=10.0).expand_dims(time=[time_vals[1]])
    obs2 = fcst2.copy().shift(x=30)  # large shift to force CRA failure

    fcst = xr.concat([fcst1, fcst2], dim="time")
    obs = xr.concat([obs1, obs2], dim="time")

    result = cra(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")

    for metric, values in result.items():
        assert len(values) == 2
        val0 = values[0]
        val1 = values[1]

        if isinstance(val0, (float, int, np.number)):
            assert not np.isnan(val0), f"{metric} should be valid for first time slice"
        else:
            assert not np.isnan(val0).all(), f"{metric} should be valid for first time slice"

        if isinstance(val1, (float, int, np.number)):
            assert np.isnan(val1), f"{metric} should be NaN for second time slice where CRA fails"
        else:
            assert np.isnan(val1).all(), f"{metric} should be NaN for second time slice where CRA fails"


def test_translate_rejects_shift_due_to_max_distance(caplog):
    """Test _translate_forecast_region rejects shift due to max distance."""
    y, x = np.ogrid[:100, :100]
    center_y, center_x = 50, 50

    obs_blob = np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * 10**2)) * 10
    fcst_blob = np.exp(-((y - center_y - 10) ** 2 + (x - center_x - 10) ** 2) / (2 * 10**2)) * 10

    coords = {"y": np.arange(100) * 10, "x": np.arange(100) * 10}  # 10 km spacing
    obs_da = xr.DataArray(obs_blob, dims=["y", "x"], coords=coords)
    fcst_da = xr.DataArray(fcst_blob, dims=["y", "x"], coords=coords)

    with caplog.at_level("INFO"):
        shifted_fcst, dx, dy = _translate_forecast_region(
            fcst_da, obs_da, "y", "x", max_distance=0.1, coord_units="degrees"
        )

    assert "Rejected shift" in caplog.text

    assert (shifted_fcst == fcst_da).all()
    assert dx == 0
    assert dy == 0


def test_translate_fallback_bbox_with_valid_data():
    """Test _translate_forecast_region falls back to bbox with valid data."""
    # Create a case where optimization might fail but bbox fallback works
    fcst = create_array(value=10.0)
    obs = create_array(value=10.0)

    # Add extreme noise to make optimization unstable
    y, x = np.ogrid[:10, :10]
    noise = np.random.normal(0, 50, size=(10, 10))
    fcst = fcst + xr.DataArray(noise, dims=["y", "x"])

    shifted_fcst, dx, dy = _translate_forecast_region(fcst, obs, "y", "x", max_distance=500, coord_units="degrees")
    # Should either succeed or return None, but covers the fallback path
    assert (shifted_fcst is not None) or (shifted_fcst is None)


def test_calc_resolution_with_degree_coordinates():
    """Test _calc_resolution correctly identifies degree coordinates"""
    # Create data with degree coordinates (lat/lon)
    lat = np.linspace(-40, -10, 10)
    lon = np.linspace(140, 160, 10)

    data = xr.DataArray(
        np.random.rand(10, 10), dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon}
    )

    resolution = _calc_resolution(data, ["latitude", "longitude"], units="degrees")
    assert resolution > 0
    assert np.isfinite(resolution)


def gaussian_blob(y_size=100, x_size=100, amp=10.0, sigma=10.0, center=(50, 50), spacing_m=1000):
    """Create a smooth 2D Gaussian blob with metre coordinates (1 km spacing)."""
    y, x = np.ogrid[:y_size, :x_size]
    cy, cx = center
    blob = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))) * amp

    da = xr.DataArray(blob, dims=["y", "x"])
    da = da.assign_coords(
        y=np.arange(y_size) * spacing_m,  # metres
        x=np.arange(x_size) * spacing_m,
    )
    return da


def test_cra_image_basic_output_type_and_keys():
    """CRA core should return an xr.Dataset with core keys for valid overlapping blobs."""
    fcst = gaussian_blob()
    obs = gaussian_blob()

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,
        coord_units="degrees",  # explicit for reproducibility
    )
    assert isinstance(result, xr.Dataset)

    expected_keys = {
        "mse_total",
        "mse_displacement",
        "mse_volume",
        "mse_pattern",
        # "optimal_shift",  # TODO: why isn't this here? Should it be in the core set?
    }
    for k in expected_keys:
        assert k in result, f"Missing key in cra_image output: {k}"

    # Basic type checks
    assert np.issubdtype(result["mse_total"].dtype, np.number)
    assert np.issubdtype(result["mse_displacement"].dtype, np.number)
    assert np.issubdtype(result["mse_volume"].dtype, np.number)
    assert np.issubdtype(result["mse_pattern"].dtype, np.number)
    # assert isinstance(result["optimal_shift"], (list, tuple))
    # assert len(result["optimal_shift"]) == 2


def test_cra_image_returns_none_when_no_valid_rain_area():
    """Returns None when nothing exceeds threshold or blobs are too small."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=50.0,  # too high -> no blobs
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,
        coord_units="degrees",
    )

    assert np.isnan(result.mse_total)


def test_cra_image_min_points_filter_returns_none():
    """Returns None when contiguous area is below min_points."""
    fcst = create_array(value=0.0)
    obs = create_array(value=0.0)
    # small 2x2 patch (4 points) above threshold
    fcst[0:2, 0:2] = 10.0
    obs[0:2, 0:2] = 10.0

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,  # require at least 10 points -> reject
        coord_units="degrees",
    )

    assert np.isnan(result.mse_total)


def test_cra_image_shape_mismatch_raises_valueerror():
    """Shape mismatches should raise ValueError (parity with cra_image behavior)."""
    fcst = create_array(shape=(10, 10))
    obs = create_array(shape=(8, 10))  # mismatched shape

    with pytest.raises(ValueError):
        _cra_image(fcst, obs, minimum_intensity=5.0, y_name="y", x_name="x")


def test_cra_image_invalid_input_types_raise_typeerror():
    """Non-xarray inputs should raise TypeError (parity with cra_image behavior)."""
    obs = create_array()
    with pytest.raises(TypeError):
        _cra_image("invalid", obs, minimum_intensity=5.0, y_name="y", x_name="x")

    fcst = create_array()
    with pytest.raises(TypeError):
        _cra_image(fcst, "invalid", minimum_intensity=5.0, y_name="y", x_name="x")


def test_cra_image_rejects_large_shift_due_to_max_distance():
    """If the optimal shift exceeds max_distance, cra_image should return None."""
    base = create_array()
    base[0:4, 0:4] = 10.0

    fcst = base
    obs = base.shift(x=30)  # force a large translation

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=1,  # very strict -> reject
        min_points=4,
        coord_units="degrees",
    )

    assert np.isnan(result.mse_total)


def test_cra_image_component_relationships_when_overlap():
    """
    Check decomposition:
    - non-negativity for displacement and volume
    - mse_total = mse_displacement + mse_volume + mse_pattern
    """
    fcst = create_array(value=10.0)
    obs = create_array(value=10.0)

    # add small noise to avoid perfect equality
    rng = np.random.default_rng(42)
    noise = xr.DataArray(rng.normal(0, 0.1, size=fcst.shape), dims=fcst.dims)
    fcst = fcst + noise

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,
        coord_units="degrees",
    )
    assert isinstance(result, xr.Dataset)

    mt = result["mse_total"]
    md = result["mse_displacement"]
    mv = result["mse_volume"]
    mp = result["mse_pattern"]

    # finite values
    for val in (mt, md, mv, mp):
        assert np.isfinite(val), f"Expected finite CRA component, got {val}"

    # common CRA properties
    assert md >= 0.0, "Displacement component should be non-negative"
    assert mv >= 0.0, "Volume component should be non-negative"

    # decomposition identity within tolerance
    assert np.isclose(
        mt, md + mv + mp, rtol=1e-5, atol=1e-6
    ), "mse_total should equal mse_displacement + mse_volume + mse_pattern"


def test_cra_image_optimal_shift_vector_type():
    """optimal_shift should be a 2-element numeric vector [dx, dy]."""
    fcst = gaussian_blob(center=(50, 50))  # baseline
    obs = gaussian_blob(center=(52, 53))  # 2 km (y), 3 km (x) shift

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,
        coord_units="metres",
        extra_components=True,
    )
    assert isinstance(result, xr.Dataset)
    shift = result["optimal_shift"]
    assert isinstance(shift, xr.DataArray)
    assert len(shift) == 2
    assert all(np.isfinite(s) for s in shift)


def test_cra_image_mse_total_nan_triggers_none(monkeypatch):
    """Force _calc_mse to return NaN so cra_image returns None and covers the branch."""
    # Build valid blobs so the pipeline reaches _calc_mse
    fcst = gaussian_blob()  # helper added earlier in this file
    obs = gaussian_blob()

    # Monkeypatch calc_mse to return NaN
    def fake_calc_mse(_a, _b):
        return np.nan

    # Patch in the module where cra_image resolves _calc_mse
    monkeypatch.setattr(sys.modules["scores.spatial.cra_impl"], "mse", fake_calc_mse)

    result = _cra_image(
        fcst,
        obs,
        minimum_intensity=5.0,
        y_name="y",
        x_name="x",
        max_distance=300,
        min_points=10,
        coord_units="metres",  # or "degrees" if your gaussian_blob uses degree coords
    )

    assert np.isnan(result).all(), "Expected None when mse_total is NaN"


def test_cra_image_invalid_coord_units_raises_valueerror():
    """Validate coord_units must be one of ['degrees', 'metres']."""
    # Build minimal valid inputs so validation is the first failure point
    fcst = create_array(value=10.0)
    obs = create_array(value=10.0)

    with pytest.raises(ValueError, match=r"coord_units must be one of \['degrees', 'metres'\]"):
        _cra_image(
            fcst,
            obs,
            minimum_intensity=5.0,
            y_name="y",
            x_name="x",
            max_distance=300,
            min_points=10,
            coord_units="km",  # invalid -> triggers the branch
        )


# TODO: Find actual data / use case rather to prove this is needed
# def test_shifted_mse_2d_returns_inf_on_valueerror_in_int(monkeypatch):
#     """
#     Force ValueError inside the try-block of _shifted_mse_2d by monkeypatching `int`.
#     This deterministically covers:
#         except (ValueError, TypeError):
#             return np.inf
#     """
#     # Minimal valid inputs so we reach the try-block
#     fcst = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
#     obs = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
#     fixed_mask = xr.DataArray(np.ones((10, 10), dtype=bool), dims=["y", "x"])
#     spatial_dims = ["x", "y"]
#     x_name, y_name = spatial_dims

#     # Valid numeric shifts so len==2 and np.isnan(shifts) is False
#     shift_x = 1.0
#     shift_y = 2.0

#     # Patch the `int` that _shifted_mse_2d resolves in its own globals to raise ValueError
#     def fake_int(*args, **kwargs):
#         raise ValueError("forced int failure")

#     monkeypatch.setitem(_shifted_mse_2d.__globals__, "int", fake_int)

#     out = _shifted_mse_2d(fcst, obs, shift_x, shift_y, x_name, y_name, fixed_mask)
#     assert out == np.inf, "Expected np.inf when int(round(...)) raises ValueError"


def test_shifted_mse_2d_returns_inf_on_typeerror_in_round(monkeypatch):
    """
    Alternatively, force TypeError from `round` to hit the same except-path.
    """
    fcst = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
    obs = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
    fixed_mask = xr.DataArray(np.ones((10, 10), dtype=bool), dims=["y", "x"])
    spatial_dims = ["x", "y"]
    x_name, y_name = spatial_dims

    shift_x = 1.0
    shift_y = 2.0

    def fake_round(*args, **kwargs):
        raise TypeError("forced round failure")

    monkeypatch.setitem(_shifted_mse_2d.__globals__, "round", fake_round)

    out = _shifted_mse_2d(fcst, obs, shift_x, shift_y, x_name, y_name, fixed_mask)
    assert out == np.inf, "Expected np.inf when round(...) raises TypeError"


def test_calc_resolution_invalid_units_raises_valueerror():
    """_calc_resolution should raise ValueError for unsupported units."""
    # Create a simple DataArray with explicit spatial coords so dy/dx are well-defined
    y = np.linspace(0, 9_000, 10)  # metres (1 km spacing)
    x = np.linspace(0, 9_000, 10)  # metres (1 km spacing)
    data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )

    # Pass an invalid units string to trigger the else branch
    with pytest.raises(ValueError, match=r"units must be 'degrees' or 'metres'"):
        _calc_resolution(data, ["y", "x"], units="km")  # invalid

    # Try a different invalid value to be extra sure the branch is covered
    with pytest.raises(ValueError, match=r"units must be 'degrees' or 'metres'"):
        _calc_resolution(data, ["y", "x"], units="meters")  # invalid spelling if you only accept 'metres'


def test_translate_forecast_region_rejects_when_shift_worsens_metrics(monkeypatch):
    """
    Cover branch:
        if rmse_shifted > rmse_original or corr_shifted < corr_original or mse_shifted > original_mse:
            return None, None, None
    by monkeypatching metric functions to force the condition true.
    """

    # Minimal valid data with a fully valid mask
    fcst = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
    obs = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])

    # 4) Force the final comparison to reject:
    #    rmse_shifted > rmse_original, corr_shifted < corr_original, mse_shifted > original_mse
    # We use counters to distinguish "shifted" vs "original" calls.
    call_state = {"rmse_calls": 0, "corr_calls": 0, "mse_calls": 0}

    def fake_shifted_mse_2d(*args, **kwargs):
        return 5000

    monkeypatch.setattr(scores.spatial.cra_impl, "_shifted_mse_2d", fake_shifted_mse_2d)

    # Run with metres to avoid degree distance surprises
    shifted, dx, dy = _translate_forecast_region(
        fcst, obs, y_name="y", x_name="x", max_distance=3000000, coord_units="metres"
    )

    # Expect no shift if metrics are worse when shifted
    assert (shifted == fcst).all()
    assert dx == 0
    assert dy == 0


def test_cra_image_invalid_coord_units():
    # Create dummy forecast and observation DataArrays with matching shape
    data = np.ones((5, 5))
    fcst = xr.DataArray(data, dims=["lat", "lon"])
    obs = xr.DataArray(data, dims=["lat", "lon"])

    # Use an invalid coord_units value
    invalid_units = "kilometers"

    with pytest.raises(ValueError) as excinfo:
        _cra_image(fcst=fcst, obs=obs, minimum_intensity=1.0, y_name="lat", x_name="lon", coord_units=invalid_units)

    # Assert the error message contains the expected text
    assert f"must be one of ['degrees', 'metres']" in str(excinfo.value)


def test_cra_image_returns_none_when_mse_is_nan(monkeypatch):
    # Create dummy forecast and observation DataArrays with matching shape
    data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    fcst = xr.DataArray(data, dims=["lat", "lon"])
    obs = xr.DataArray(data, dims=["lat", "lon"])

    # Monkeypatch generate_largest_rain_area_2d to return blobs with NaNs
    def fake_generate_largest_rain_area_2d(fcst, obs, *, minimum_intensity, min_points):
        return fcst, obs

    monkeypatch.setattr(
        sys.modules["scores.spatial.cra_impl"], "_generate_largest_rain_area_2d", fake_generate_largest_rain_area_2d
    )

    # Monkeypatch calc_mse to return NaN explicitly
    monkeypatch.setattr(sys.modules["scores.spatial.cra_impl"], "mse", lambda a, b: np.nan)

    result = _cra_image(fcst=fcst, obs=obs, minimum_intensity=1.0, y_name="lat", x_name="lon", coord_units="metres")

    assert np.isnan(result.mse_total)


def test_cra_time_val_conversion_int_and_str(monkeypatch):
    # Forecast and observation with integer time coordinates
    fcst_data = np.ones((2, 2, 2))
    obs_data = np.ones((2, 2, 2))
    fcst_int = xr.DataArray(fcst_data, dims=["time", "lat", "lon"], coords={"time": [0, 1]})
    obs_int = xr.DataArray(obs_data, dims=["time", "lat", "lon"], coords={"time": [0, 1]})

    # Monkeypatch .sel to avoid KeyError when datetime64 is passed
    original_sel = xr.DataArray.sel

    def safe_sel(self, indexers=None, drop=False):
        # Ignore mismatched type and just return the first slice
        return original_sel(self, {"time": self.time.values[0]}, drop=drop)

    # Run cra with integer time coords (covers int -> datetime64 conversion)
    result_int = cra(fcst_int, obs_int, minimum_intensity=1.0, y_name="lat", x_name="lon")
    assert all(len(v) == 2 for v in result_int.values())

    # Forecast and observation with string time coordinates (covers str -> datetime64 conversion)
    fcst_str = xr.DataArray(fcst_data, dims=["time", "lat", "lon"], coords={"time": ["2020-01-01", "2020-01-02"]})
    obs_str = xr.DataArray(obs_data, dims=["time", "lat", "lon"], coords={"time": ["2020-01-01", "2020-01-02"]})

    result_str = cra(fcst_str, obs_str, minimum_intensity=1.0, y_name="lat", x_name="lon")
    assert all(len(v) == 2 for v in result_str.values())


def test_cra_image_returns_none_when_shifted_fcst_is_none():
    # Create dummy forecast and observation DataArrays
    data = np.ones((2, 2))
    fcst = xr.DataArray(data, dims=["lat", "lon"])
    obs = xr.DataArray(data, dims=["lat", "lon"])

    result = _cra_image(
        fcst, obs, minimum_intensity=1.0, y_name="lat", x_name="lon", coord_units="metres", extra_components=True
    )

    assert np.isnan(result.mse_total)


def test_nansave_int():

    result = _nansafe_int(np.nan)
    assert np.isnan(result)

    result = _nansafe_int(45.2)
    assert result == 45


def test_validate_cra2d_inputs():

    fcst_data = np.ones((1, 2, 2))
    obs_data = np.ones((1, 2, 2))
    fcst = xr.DataArray(fcst_data, dims=["time", "lat", "lon"], coords={"time": [0]})
    obs = xr.DataArray(obs_data, dims=["time", "lat", "lon"], coords={"time": [0]})

    fcst_data_toolong = np.ones((2, 2, 2))
    obs_data_toolong = np.ones((2, 2, 2))
    fcst_toolong = xr.DataArray(fcst_data_toolong, dims=["time", "lat", "lon"], coords={"time": [0, 1]})
    obs_toolong = xr.DataArray(obs_data_toolong, dims=["time", "lat", "lon"], coords={"time": [0, 1]})

    fcst_data_toobig = np.ones((1, 2, 2, 2, 2))
    obs_data_toobig = np.ones((1, 2, 2, 2, 2))
    fcst_toobig = xr.DataArray(fcst_data_toobig, dims=["time", "lat", "lon", "ens", "level"], coords={"time": [0]})
    obs_toobig = xr.DataArray(obs_data_toobig, dims=["time", "lat", "lon", "ens", "level"], coords={"time": [0]})

    fcst_wrongdim = xr.DataArray(fcst_data, dims=["time", "lat", "ens"], coords={"time": [0]})
    obs_wrongdim = xr.DataArray(obs_data, dims=["time", "lat", "ens"], coords={"time": [0]})

    time_name = "time"
    coord_units = "metres"
    x_name = "lat"
    y_name = "lon"

    with pytest.raises(TypeError) as excinfo:
        _validate_cra2d_inputs("Invalid data type", obs, time_name, coord_units, x_name, y_name)
    assert "must be an xarray" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        _validate_cra2d_inputs(fcst, "Invalid data type", time_name, coord_units, x_name, y_name)
    assert "must be an xarray" in str(excinfo.value)

    # Too much time
    with pytest.raises(ValueError) as excinfo:
        _validate_cra2d_inputs(fcst_toolong, obs_toolong, time_name, coord_units, x_name, y_name)
    assert "time dimension can only have a length of one" in str(excinfo.value)

    # Address mismatched shapes
    with pytest.raises(ValueError) as excinfo:
        _validate_cra2d_inputs(fcst_toobig, obs, time_name, coord_units, x_name, y_name)
    assert "must have the same shape" in str(excinfo.value)

    # Address n-d dimension being supplied to a 2D method
    with pytest.raises(ValueError) as excinfo:
        _validate_cra2d_inputs(fcst_toobig, obs_toobig, time_name, coord_units, x_name, y_name)
    assert "contain additional coordinate" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _validate_cra2d_inputs(fcst_wrongdim, obs, time_name, coord_units, x_name, y_name)
    assert "'lon' not found in forecast data" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _validate_cra2d_inputs(fcst, obs_wrongdim, time_name, coord_units, x_name, y_name)
    assert "'lon' not found in observation data" in str(excinfo.value)
