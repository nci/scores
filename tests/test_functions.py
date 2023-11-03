"""
Tests scores.functions
"""
import numpy as np
import pytest
import xarray as xr

from scores.functions import angular_difference

DA_DIR_A = xr.DataArray(
    data=[[[0, 50], [340, 100]], [[300, np.nan], [90, 0]]],  # include nan
    dims=["i", "j", "k"],
    coords={"i": [1, 2], "j": [10000, 10001], "k": [1, 2]},
)
DA_DIR_B = xr.DataArray(
    data=[[160, 359, 10], [100, np.nan, 20]],
    dims=["k", "j"],  # reversed dim order, missing 'i' dim
    coords={"k": [1, 2], "j": [10001, 10000, 10002]},  # extra station, changed order
)
DS_DIR_A = xr.Dataset({"a": DA_DIR_A, "b": DA_DIR_B})
DS_DIR_B = xr.Dataset({"a": DA_DIR_B, "b": DA_DIR_A})
DA_DIR_NAN = xr.DataArray(  # array of only nans
    data=[[[np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]]],
    dims=["i", "j", "k"],
    coords={"i": [1, 2], "j": [10000, 10001], "k": [1, 2]},
)
EXP_AD = xr.DataArray(
    data=[[[1, np.nan], [180, 0]], [[59, np.nan], [70, 100]]],
    dims=["i", "j", "k"],
    coords={"i": [1, 2], "j": [10000, 10001], "k": [1, 2]},
)
EXP_AD_NAN = xr.DataArray(
    data=[[[np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]]],
    dims=["i", "j", "k"],
    coords={"i": [1, 2], "j": [10000, 10001], "k": [1, 2]},
)
EXP_DS = xr.Dataset({"a": EXP_AD, "b": EXP_AD})


@pytest.mark.parametrize(
    ("source_a", "source_b", "expected"),
    [
        (DA_DIR_A, DA_DIR_B, EXP_AD),
        (DA_DIR_A, DA_DIR_NAN, EXP_AD_NAN),
        (DS_DIR_A, DS_DIR_B, EXP_DS),
    ],
)
def test_angular_difference(source_a, source_b, expected):
    """Tests that angular_difference returns correct values"""
    output = angular_difference(source_a, source_b)
    output.broadcast_equals(expected)
