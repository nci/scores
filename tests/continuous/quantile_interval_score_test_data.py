"""
Generation of the test data used to test scores.continuous.quantile_interval_score and
scores.continuous.interval_score
"""

import numpy as np
import xarray as xr

ERROR_MESSAGE_QTILE = (
    "Expected 0 < lower_qtile_level < upper_qtile_level < 1. But got lower_qtile_level ="
    " {} and upper_qtile_level = {}"
)
ERROR_MESSAGE_FCST_COND = "Input does not satisfy fcst_lower_qtile < fcst_upper_qtile condition."
ERROR_MESSAGE_INTERVAL_RANGE = "`interval_range` must be strictly between 0 and 1"
FCST_LOWER = xr.DataArray([10, 15, 21], dims="time", coords={"time": np.arange(3)})
FCST_UPPER = xr.DataArray([20, 25, 30], dims="time", coords={"time": np.arange(3)})
FCST_UPPER_STATION = FCST_UPPER.rename({"time": "station"})
OBS = xr.DataArray([12, 14, 32], dims="time", coords={"time": np.arange(3)})
OBS_STATION = OBS.rename({"time": "station"})

EXPECTED_WITH_TIME = xr.Dataset(
    data_vars={
        "interval_width_penalty": ("time", [10, 10, 9]),
        "overprediction_penalty": ("time", [0, 10, 0]),
        "underprediction_penalty": ("time", [0, 0, 20]),
        "total": ("time", [10, 20, 29]),
    },
    coords={"time": np.arange(3)},
)
FCST_LOWER_2D = xr.DataArray(
    [[10, 15, 20], [5, 10, 15]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
    name="temperature",
)
FCST_UPPER_2D = xr.DataArray(
    [[20, 25, 30], [6, 12, 19]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
    name="temperature",
)
FCST_LOWER_2D_WITH_NAN = xr.DataArray(
    [[10, 15, np.nan], [5, np.nan, 15]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
    name="temperature",
)
FCST_UPPER_2D_WITH_NAN = xr.DataArray(
    [[20, 25, np.nan], [6, np.nan, 19]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
    name="temperature",
)
OBS_2D = xr.DataArray(
    [[12, 14, 32], [4, 10.5, 21]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
    name="temperature",
)
WEIGHTS = xr.DataArray(
    [[0, 1, 1], [1, 1, 0]],
    dims=["time", "station"],
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[10, 10, 10], [1, 2, 4]]),
        "overprediction_penalty": (["time", "station"], [[0, 10, 0], [10, 0, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, 20], [0, 0, 20]]),
        "total": (["time", "station"], [[10, 20, 30], [11, 2, 24]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D_WITHOUT_TIME = xr.Dataset(
    data_vars={
        "interval_width_penalty": ("station", [5.5, 6, 7]),
        "overprediction_penalty": ("station", [5, 5, 0]),
        "underprediction_penalty": ("station", [0, 0, 20]),
        "total": ("station", [10.5, 11, 27]),
    },
    coords={"station": np.arange(3)},
)
EXPECTED_2D_NON_BALANCE = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[10, 10, 10], [1, 2, 4]]),
        "overprediction_penalty": (["time", "station"], [[0, 5, 0], [5, 0, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, 20], [0, 0, 20]]),
        "total": (["time", "station"], [[10, 15, 30], [6, 2, 24]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D_WITH_NAN = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[10, 10, np.nan], [1, np.nan, 4]]),
        "overprediction_penalty": (["time", "station"], [[0, 10, np.nan], [10, np.nan, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, np.nan], [0, np.nan, 20]]),
        "total": (["time", "station"], [[10, 20, np.nan], [11, np.nan, 24]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D_WITH_WEIGHTS = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[0, 10, 10], [1, 2, 0]]),
        "overprediction_penalty": (["time", "station"], [[0, 10, 0], [10, 0, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, 20], [0, 0, 0]]),
        "total": (["time", "station"], [[0, 20, 30], [11, 2, 0]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D_INTERVAL = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[10, 10, 10], [1, 2, 4]]),
        "overprediction_penalty": (["time", "station"], [[0, 4, 0], [4, 0, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, 8], [0, 0, 8]]),
        "total": (["time", "station"], [[10, 14, 18], [5, 2, 12]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
EXPECTED_2D_WITHOUT_TIME_INTERVAL = xr.Dataset(
    data_vars={
        "interval_width_penalty": ("station", [5.5, 6, 7]),
        "overprediction_penalty": ("station", [1.66666667, 1.66666667, 0]),
        "underprediction_penalty": ("station", [0, 0, 6.66666667]),
        "total": ("station", [7.16666667, 7.66666667, 13.66666667]),
    },
    coords={"station": np.arange(3)},
)
EXPECTED_2D_WITH_WEIGHTS_INTERVAL = xr.Dataset(
    data_vars={
        "interval_width_penalty": (["time", "station"], [[0, 10, 10], [1, 2, 0]]),
        "overprediction_penalty": (["time", "station"], [[0, 3.33333333, 0], [3.33333333, 0, 0]]),
        "underprediction_penalty": (["time", "station"], [[0, 0, 6.66666667], [0, 0, 0]]),
        "total": (["time", "station"], [[0, 13.33333333, 16.66666667], [4.33333333, 2, 0]]),
    },
    coords={"station": np.arange(3), "time": np.arange(2)},
)
