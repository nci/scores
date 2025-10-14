"""
Generation of the test data used to test scores.continuous.quantile_score
"""

import numpy as np
import pandas as pd
import xarray as xr

FCST_ARRAY = [
    [[0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    [[0.5, 0.5, 0.5], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0]],
]
FCST_COORDS = {
    "lead_time": [15, 39, 63, 87],
    "valid_start": pd.date_range("2000-01-01 15:00:00", periods=2),
    "station_index": [10000, 10001, 10002],
}
FCST1 = xr.DataArray(data=FCST_ARRAY, coords=FCST_COORDS)
FCST2 = xr.DataArray([0.0, 1, 2, np.nan], coords=[[15, 39, 63, 87]], dims=["lead_time"])
OBS_COORDS = {"valid_start": pd.date_range("2000-01-02 15:00:00", periods=2), "station_index": [10001, 10002, 10003]}
OBS_ARRAY = np.tile(np.array([1.0]), (2, 3))
OBS1 = xr.DataArray(data=OBS_ARRAY, coords=OBS_COORDS)
OBS_DS = xr.Dataset(
    data_vars={
        "temperature_1": (["valid_start", "station_index"], OBS_ARRAY),
        "temperature_2": (["valid_start", "station_index"], OBS_ARRAY),
    },
    coords=OBS_COORDS,
)
FCST_DS = xr.Dataset(
    data_vars={
        "temperature_1": (["lead_time", "valid_start", "station_index"], FCST_ARRAY),
        "temperature_2": (["lead_time", "valid_start", "station_index"], FCST_ARRAY),
    },
    coords=FCST_COORDS,
)
DA1_2X2 = xr.DataArray([[5, 3], [7, 2]], coords=[[0, 1], [0, 1]], dims=["i", "j"])
DA1_2X2X2 = xr.DataArray(
    [[[5, 3], [7, 2]], [[7, 4], [8, 2]]],
    coords=[[0, 1], [0, 1], [0, 1]],
    dims=["k", "i", "j"],
)
DA2_2X2 = xr.DataArray([[2, 3], [8, 4]], coords=[[0, 1], [0, 1]], dims=["i", "j"])
EXPECTED1 = xr.DataArray(
    [[[0.0, 0.0]], [[0.1, 0.1]], [[0.2, 0.2]], [[0.2, 0.2]]],
    coords=[[15, 39, 63, 87], [pd.to_datetime("2000-01-02 15:00:00")], [10001, 10002]],
    dims=["lead_time", "valid_start", "station_index"],
)
EXPECTED2 = xr.DataArray([0.0, 0.05, 0.1, 0.1], coords=[[15, 39, 63, 87]], dims=["lead_time"])
EXPECTED3 = xr.DataArray([0.0, 0, 0, np.nan], coords=[[15, 39, 63, 87]], dims=["lead_time"])
EXPECTED4 = xr.DataArray(0.1)
EXPECTED_DS1 = xr.Dataset(
    data_vars={
        "temperature_1": (["lead_time"], [0.0, 0.05, 0.1, 0.1]),
        "temperature_2": (["lead_time"], [0.0, 0.05, 0.1, 0.1]),
    },
    coords={"lead_time": [15, 39, 63, 87]},
)
EXPECTED_DS2 = xr.Dataset(
    data_vars={
        "temperature_1": (0.1),
        "temperature_2": (0),
    },
)
EXPECTED_DS3 = xr.Dataset(
    data_vars={
        "temperature_1": xr.DataArray(0.0625),
        "temperature_2": xr.DataArray(0.0625),
    }
)
WEIGHTS_ARRAY1 = [
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1]],
]
WEIGHTS_ARRAY2 = [
    [[0, 0, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]],
]
WEIGHTS = xr.DataArray(data=WEIGHTS_ARRAY1, coords=FCST_COORDS)
WEIGHTS_DS = xr.Dataset(
    data_vars={
        "temperature_1": (["lead_time", "valid_start", "station_index"], WEIGHTS_ARRAY1),
        "temperature_2": (["lead_time", "valid_start", "station_index"], WEIGHTS_ARRAY2),
    },
    coords=FCST_COORDS,
)
