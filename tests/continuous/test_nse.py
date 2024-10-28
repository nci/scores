import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scores
from scores.continuous.standard_impl import nse

def test_nse_with_xr_dataset_and_weights():
    # Create forecast and observed datasets with the 'location' dimension only
    fcst_data = xr.Dataset(
        {
            "var": (("location"), [3, 4, 5])  # Only one dimension: 'location'
        },
        coords={"location": ["A", "B", "C"]}
    )
    obs_data = xr.Dataset(
        {
            "var": (("location"), [2, 3, 4])  # Matching 'location' dimension
        },
        coords={"location": ["A", "B", "C"]}
    )

    # Define weights for each location
    weights = xr.DataArray(
        [0.2, 0.5, 0.3],
        dims=["location"],
        coords={"location": ["A", "B", "C"]}
    )

    # Calculate NSE without reducing dimensions
    nse_value = nse(fcst_data, obs_data, weights=weights)

    expected_nse = -0.5 

    # Check if the computed NSE matches the expected value
    assert np.isclose(nse_value, expected_nse, atol=1e-5), f"Expected {expected_nse}, got {nse_value}"

def test_nse_with_multi_dimensional_data():
    # Test with multi-dimensional data and reduced dimensions
    fcst_data = xr.Dataset(
        {
            "var": (("lead_time", "location"), [[3, 4, 5], [6, 7, 8]])
        },
        coords={"lead_time": [0, 1], "location": ["A", "B", "C"]}
    )
    obs_data = xr.Dataset(
        {
            "var": (("lead_time", "location"), [[2, 3, 4], [5, 6, 7]])
        },
        coords={"lead_time": [0, 1], "location": ["A", "B", "C"]}
    )
    weights = xr.DataArray(
        [0.3, 0.3, 0.4],
        dims=["location"],
        coords={"location": ["A", "B", "C"]}
    )
    
    # Calculate NSE with reduction along the 'lead_time' dimension
    nse_value = nse(fcst_data, obs_data, reduce_dims=[], weights=weights)
    
   
    expected_nse = 0.65714286 
    
    assert np.isclose(nse_value, expected_nse, atol=1e-5), f"Expected {expected_nse}, got {nse_value}"

def test_nse_with_datasets():
    # Create dataset
    time = pd.date_range("2024-01-01", periods=5, freq="D")
    stations = ["Station1", "Station2", "Station3"]
    fcst_data = np.array([[3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]])
    obs_data = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    fcst_xr = xr.DataArray(fcst_data, coords={"time": time, "station": stations}, dims=["station", "time"])
    obs_xr = xr.DataArray(obs_data, coords={"time": time, "station": stations}, dims=["station", "time"])

    fcst_ds = xr.Dataset({"forecast": fcst_xr})
    obs_ds = xr.Dataset({"observed": obs_xr})

    # Convert datasets to data arrays if needed
    if isinstance(fcst_ds, xr.Dataset):
        data_variable_name = list(fcst_ds.data_vars.keys())[0]
        # Get the name of the first data variable
        forecast = fcst_ds.to_array(dim=data_variable_name)

    if isinstance(obs_ds, xr.Dataset):
        data_variable_name = list(obs_ds.data_vars.keys())[0]
        # Get the name of the first data variable
        observed = obs_ds.to_array(dim=data_variable_name)

    # Calculate NSE and assert the result
    assert nse(forecast, observed) == 0.5


def test_nse_with_datasets_noconversion():
    # Create dataset
    time = pd.date_range("2024-01-01", periods=5, freq="D")
    stations = ["Station1", "Station2", "Station3"]
    fcst_data = np.array([[3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]])
    obs_data = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    fcst_xr = xr.DataArray(fcst_data, coords={"time": time, "station": stations}, dims=["station", "time"])
    obs_xr = xr.DataArray(obs_data, coords={"time": time, "station": stations}, dims=["station", "time"])

    fcst_ds = xr.Dataset({"forecast": fcst_xr})
    obs_ds = xr.Dataset({"observed": obs_xr})

    # Calculate NSE and assert the result
    assert nse(fcst_ds, obs_ds) == 0.5


def test_weights():
    fcst = np.array([3, 4, 5, 6, 7])
    obs = np.array([2, 3, 4, 5, 6])
    weights = np.array([1, 2, 3, 2, 1])
    nse_value = nse(fcst, obs, weights=weights)
    assert nse_value == 0.5


def test_angular():
    fcst = np.array([0, 90, 180, 270, 360])
    obs = np.array([0, 90, 180, 270, 360])
    nse_value = nse(fcst, obs, angular=True)
    assert nse_value == 1.0


def test_nse_with_weights():
    # Test with different weights and NaNs
    fcst_data = xr.Dataset(
        {
            "var": (("location"), [3, 4, np.nan])  # One NaN value
        },
        coords={"location": ["A", "B", "C"]}
    )
    obs_data = xr.Dataset(
        {
            "var": (("location"), [2, np.nan, 4])  # One NaN value
        },
        coords={"location": ["A", "B", "C"]}
    )
    weights = xr.DataArray(
        [0.1, 0.3, 0.6],
        dims=["location"],
        coords={"location": ["A", "B", "C"]}
    )
    
    # Calculate NSE
    nse_value = nse(fcst_data, obs_data, weights=weights)
    
    
    expected_nse = 0.50  
    
    assert np.isclose(nse_value, expected_nse, atol=1e-5), f"Expected {expected_nse}, got {nse_value}"


def test_nse_with_missing_values():
    # Test with missing values in forecast and observed data
    fcst_data = xr.Dataset(
        {
            "var": (("location"), [3, np.nan, 5])  # One NaN value
        },
        coords={"location": ["A", "B", "C"]}
    )
    obs_data = xr.Dataset(
        {
            "var": (("location"), [2, 3, np.nan])  # One NaN value
        },
        coords={"location": ["A", "B", "C"]}
    )
    weights = xr.DataArray(
        [0.2, 0.5, 0.3],
        dims=["location"],
        coords={"location": ["A", "B", "C"]}
    )
    
    # Calculate NSE
    nse_value = nse(fcst_data, obs_data, weights=weights)

    expected_nse = -1.0  # Adjust this based on manual computation of the available values
    
    assert np.isclose(nse_value, expected_nse, atol=1e-5), f"Expected {expected_nse}, got {nse_value}"

