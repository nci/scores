import pytest
import numpy as np
import pandas as pd
import xarray as xr
import scores
from scores.continuous.standard_impl import nse


def test_xarray_dataarray():
    fcst_xr = xr.DataArray([3, 4, 5, 6, 7])
    obs_xr = xr.DataArray([2, 3, 4, 5, 6])
    assert nse(fcst_xr, obs_xr) == 0.5


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
