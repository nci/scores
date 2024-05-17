import pytest
import numpy as np
import pandas as pd
import xarray as xr
from scores.continuous import nse

def test_numpy_arrays():
    fcst_np = np.array([3, 4, 5, 6, 7])
    obs_np = np.array([2, 3, 4, 5, 6])
    assert nse(fcst_np, obs_np) == 0.5

def test_pandas_series():
    fcst_series = pd.Series([3, 4, 5, 6, 7])
    obs_series = pd.Series([2, 3, 4, 5, 6])
    assert nse(fcst_series, obs_series) == 0.5

def test_xarray_dataarray():
    fcst_xr = xr.DataArray([3, 4, 5, 6, 7])
    obs_xr = xr.DataArray([2, 3, 4, 5, 6])
    assert nse(fcst_xr, obs_xr) == 0.5

def test_lists():
    fcst_list = [3, 4, 5, 6, 7]
    obs_list = [2, 3, 4, 5, 6]
    assert nse(fcst_list, obs_list) == 0.5

def test_mixed_types():
    fcst_mix = np.array([3, 4, 5, 6, 7])
    obs_mix = pd.Series([2, 3, 4, 5, 6])
    assert nse(fcst_mix, obs_mix) == 0.5

def test_angular():
    fcst = np.array([0, 90, 180, 270, 360])
    obs = np.array([0, 90, 180, 270, 360])
    assert nse(fcst, obs, angular=True) == 1.0

def test_weights():
    fcst = np.array([3, 4, 5, 6, 7])
    obs = np.array([2, 3, 4, 5, 6])
    weights = np.array([1, 2, 3, 2, 1])
    assert nse(fcst, obs, weights=weights) == 0.5