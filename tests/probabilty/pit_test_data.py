"""
Test data for `scores.probability.pit`.
"""

import numpy as np
import xarray as xr
from numpy import nan

DA_PIT_VALUES = xr.DataArray(
    data=[0, 0.4, 0.5, nan],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004]},
)

ARRAY_PIT_THRESH = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

EXP_PIT_PTMASSS = xr.DataArray(
    data=[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fcst cdf had no point mass at .2
        [0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1, 1, 1],  # fcst cdf with point mass
        [0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1, 1, 1],  # fcst cdf with point mass
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    ],
    dims=["station", "pit_threshold"],
    coords={"station": [1001, 1002, 1003, 1004], "pit_threshold": ARRAY_PIT_THRESH},
)

EXP_PIT_NOPTMASS = xr.DataArray(
    data=[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    ],
    dims=["station", "pit_threshold"],
    coords={"station": [1001, 1002, 1003, 1004], "pit_threshold": ARRAY_PIT_THRESH},
)


DA_FCST_CDF = xr.DataArray(
    data=[
        [
            [0, 0, 0.1, 0.1, 0.4, 0.5, 0.9, 1],
            [0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1],
            [0, 0, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9],
            [0, 0, 0.4, 0.5, nan, 0.7, 0.8, 0.9],
            [0, 0, 0, 0.5, 0.7, 0.8, 1, 1],
            [nan, nan, nan, nan, nan, nan, nan, nan],
            [0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1],
        ],
        [
            [0, 0, 0.1, 0.1, 0.4, 0.5, 0.9, 1],
            [0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1],
            [0, 0, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0, 0, 0.4, 0.5, nan, 0.7, 0.8, 0.9],
            [0, 0, 0, 0.5, 0.7, 0.8, 1, 1],
            [nan, nan, nan, nan, nan, nan, nan, nan],
            [0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1],
        ],
    ],
    dims=["lead_day", "station", "x"],
    coords={
        "station": [1000, 1001, 1002, 1003, 1004, 1005, 1006],
        "lead_day": [1, 2],
        "x": [0, 1, 2, 3, 4, 5, 6, 7],
    },
)

DA_OBS = xr.DataArray(
    data=[2, 7.5, 4.3, 2, 3.2, nan, 3.0],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006, 2000]},
)

EXP_PIT = xr.DataArray(
    data=[
        [
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # obs at point mass
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # obs at 95th p'tile
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # obs nearest 65th p'tile
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # obs at poss pt mass, cdf cts
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        ],
        [
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # obs at point mass
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # obs at 50th p'tile
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # obs nearest 65th p'tile
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # obs at poss pt mass, cdf cts
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        ],
    ],
    dims=["lead_day", "station", "pit_threshold"],
    coords={
        "lead_day": [1, 2],
        "station": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 2000],
        "pit_threshold": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 1],
    },
)

DA_OBS2 = xr.DataArray(
    data=[2, 7.5, 4.3, 2, 3.2, nan],
    dims=["x"],
    coords={"x": [1001, 1002, 1003, 1004, 1005, 1006]},
)

DA_OBS3 = xr.DataArray(
    data=[2, 7.5, 4.3, 2, 3.2, nan],
    dims=["y"],
    coords={"y": [1001, 1002, 1003, 1004, 1005, 1006]},
)

DA_FCST_CDF2 = xr.DataArray(
    data=[[0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1.2]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 1, 2, 3, 4, 5, 6, 7]},
)

DA_FCST_CDF3 = xr.DataArray(
    data=[[0, 0, 0.4, 0.5, 0.7, 0.8, 1, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 1, 20, 3, 4, 5, 6, 7]},
)

DA_PIT_CDF1 = xr.DataArray(  # pit_cdf(0) = 0
    data=[[[0, 0.5, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [nan, nan, nan, nan, nan, nan]]],
    dims=["date", "station", "pit_thresh"],
    coords={
        "station": [1001, 1002, 1003],
        "date": ["2020-01-01"],
        "pit_thresh": [0, 0.2, 0.4, 0.6, 0.8, 1],
    },
)

EXP_PIT_HIST1A = xr.DataArray(  # 4 bins
    data=[0.625 / 2, 1.5 / 2 - 0.625 / 2, 1 - 1.5 / 2, 1 - 1],
    dims=["right_endpoint"],
    coords={
        "right_endpoint": [0.25, 0.5, 0.75, 1],
        "left_endpoint": ("right_endpoint", [0, 0.25, 0.5, 0.75]),
        "bin_centre": ("right_endpoint", [0.125, 0.375, 0.625, 0.875]),
    },
)

EXP_PIT_HIST1B = xr.DataArray(  # 2 bins
    data=[0.625 / 2 + 1.5 / 2 - 0.625 / 2, 1 - 1.5 / 2 + 1 - 1],
    dims=["right_endpoint"],
    coords={
        "right_endpoint": [0.5, 1],
        "left_endpoint": ("right_endpoint", [0, 0.5]),
        "bin_centre": ("right_endpoint", [0.25, 0.75]),
    },
)

DA_PIT_CDF2 = xr.DataArray(  # pit_cdf(0) > 0
    data=[[[0.2, 0.5, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [nan, nan, nan, nan, nan, nan]]],
    dims=["date", "station", "pit_thresh"],
    coords={
        "station": [1001, 1002, 1003],
        "date": ["2020-01-01"],
        "pit_thresh": [0, 0.2, 0.4, 0.6, 0.8, 1],
    },
)

DA_PIT_CDF3 = xr.DataArray(  # support outside [0,1]
    data=[[[0.2, 0.2, 0.5, 1]]],
    dims=["date", "station", "pit_thresh"],
    coords={
        "station": [1001],
        "date": ["2020-01-01"],
        "pit_thresh": [-1, 0, 0.5, 1],
    },
)

DA_PIT_CDF4 = xr.DataArray(  # support outside [0,1]
    data=[[[0, 0.2, 0.5, 0.9]]],
    dims=["date", "station", "pit_thresh"],
    coords={
        "station": [1001],
        "date": ["2020-01-01"],
        "pit_thresh": [-1, 0, 1, 2],
    },
)

EXP_PIT_HIST2A = xr.DataArray(  # preserve some dims, 5 bins
    data=[[0.5, 0.5, 0, 0, 0], [0, 0, 1, 0, 0], [nan, nan, nan, nan, nan]],
    dims=["station", "right_endpoint"],
    coords={
        "station": [1001, 1002, 1003],
        "right_endpoint": [0.2, 0.4, 0.6, 0.8, 1.0],
        "left_endpoint": ("right_endpoint", [0, 0.2, 0.4, 0.6, 0.8]),
        "bin_centre": ("right_endpoint", [0.1, 0.3, 5.0, 0.7, 0.9]),
    },
)

EXP_PIT_HIST2B = xr.DataArray(  # collapse all dims
    data=[0.625 / 2, 1.5 / 2 - 0.625 / 2, 1 - 1.5 / 2, 1 - 1],
    dims=["right_endpoint"],
    coords={
        "right_endpoint": [0.25, 0.5, 0.75, 1],
        "left_endpoint": ("right_endpoint", [0, 0.25, 0.5, 0.75]),
        "bin_centre": ("right_endpoint", [0.125, 0.375, 0.625, 0.875]),
    },
)

DA_PIT_SCORE1 = xr.DataArray(  # support outside [0,1]
    data=[
        [[0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], [0, 0, 0.25, 0.5, 0.75, 1, 1]],
        [[0.0, 0.0, 0.25, 0.5, 0.75, 1, 1], [0, 0, 0.25, nan, 0.75, 1, 1]],
        [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0], [nan, nan, nan, nan, nan, nan, nan]],
    ],
    dims=["station", "date", "pit_thresh"],
    coords={
        "station": [1001, 1002, 1003],
        "date": ["2020-01-01", "2020-01-02"],
        "pit_thresh": [-1, 0, 0.25, 0.5, 0.75, 1, 2],
    },
)

EXP_PIT_SCORE1A = xr.Dataset(data_vars={"score": 0.0, "expectation": 1 / 2, "variance": 1 / 12})

EXP_PIT_SCORE1B = xr.Dataset(
    data_vars={
        "score": (
            ["station", "date"],
            [[1 / 24 + 1 / 24, 0], [0, 0], [1 / 24 + 1 / 24, nan]],
        ),
        "expectation": (["station", "date"], [[0.25, 0.5], [0.5, 0.5], [0.75, nan]]),
        "variance": (
            ["station", "date"],
            [[1 / (12 * 4), 1 / 12], [1 / 12, 1 / 12], [1 / (12 * 4), nan]],
        ),
    },
    coords={"station": [1001, 1002, 1003], "date": ["2020-01-01", "2020-01-02"]},
)

DA_PIT_SCORE2 = xr.DataArray(  # support outside [0,1]
    data=[
        [[0.0, 0.0, 0.25, 0.5, 0.75, 1, 1], [0, 0, 0.25, nan, 0.75, 1, 1]],
        [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0], [nan, nan, nan, nan, nan, nan, nan]],
    ],
    dims=["station", "date", "pit_thresh"],
    coords={
        "station": [1001, 1002],
        "date": ["2020-01-01", "2020-01-02"],
        "pit_thresh": [-1, 0, 0.25, 0.5, 0.75, 1, 2],
    },
)


EXP_PIT_SCORE2 = xr.Dataset(
    data_vars={
        "score": (["station"], [0.0, 1 / 24 + 1 / 24]),
        "expectation": (["station"], [1 / 2, 0.75]),
        "variance": (["station"], [1 / 12, 1 / (12 * 4)]),
    },
    coords={"station": [1001, 1002]},
)
