"""
Test data for testing scores.emerging functions
"""

import numpy as np
import xarray as xr

# test data for risk_matrix_score and related functions
DA_RMS_FCST = xr.DataArray(
    data=[
        [[0.51, 0.38, 0.02], [0.51, 0.38, 0.02]],
        [[0.4, 0.35, 0.12], [np.nan, 0.6, np.nan]],
    ],
    dims=["stn", "day", "sev"],
    coords={
        "stn": [101, 102],
        "day": [6, 7],
        "sev": [1, 2, 3],
    },
)
DA_RMS_OBS = xr.DataArray(
    data=[
        [[1.0, 1.0, 0.0], [1, np.nan, np.nan]],
        [[1.0, 0, 0], [1.0, 0, 0]],
    ],
    dims=["stn", "day", "sev"],
    coords={
        "stn": [101, 102],
        "day": [6, 7],
        "sev": [1, 2, 3],
    },
)
DA_RMS_WT0 = xr.DataArray(
    data=[
        [1, 1, 1],
        [1, 1, 1],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.1, 0.4],
        "sev": [1, 2, 3],
    },
)
EXP_RMS_CASE0 = xr.DataArray(
    data=[
        [0 + 0.6 + 0, np.nan],  # written as sum of column scores
        [0 + 0.1 + 0.1, np.nan],
    ],
    dims=["stn", "day"],
    coords={
        "stn": [101, 102],
        "day": [6, 7],
    },
)
EXP_RMS_CASE1 = xr.DataArray(
    data=[
        [0.6, np.nan],  # written as sum of column scores
        [0.6 + 0.1 + 0.1, np.nan],
    ],
    dims=["stn", "day"],
    coords={
        "stn": [101, 102],
        "day": [6, 7],
    },
)
DA_RMS_FCST1 = xr.DataArray(  # sydney example from paper
    data=[[0.66, 0.38, 0.18], [0.05, 0.05, 0.05]],
    dims=["forecaster", "sev"],
    coords={
        "forecaster": ["first", "second"],
        "sev": [1, 2, 3],
    },
)
DA_RMS_OBS1 = xr.DataArray(
    data=[[1.0, 0, 0], [1.0, 1, 1], [0, 0, 0]],
    dims=["obs_case", "sev"],
    coords={"sev": [1, 2, 3], "obs_case": [0, 1, 2]},
)
DA_RMS_WT1 = xr.DataArray(
    data=[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.1, 0.4, 0.7],
        "sev": [1, 2, 3],
    },
)
DA_RMS_WT2 = xr.DataArray(  # escalation scaling, written as matrix form
    data=[
        [2, 3, 0],
        [0, 2, 3],
        [1, 0, 2],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.7, 0.4, 0.1],  # note coordinate written in descending order
        "sev": [1, 2, 3],
    },
)
DA_RMS_WT2A = xr.DataArray(  # escalation scaling
    data=[
        [2, 0, 3],
        [0, 3, 2],
        [1, 2, 0],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.7, 0.4, 0.1],
        "sev": [1, 3, 2],  # note coordinates transposed
    },
)
EXP_RMS_CASE2 = xr.DataArray(  # Sydney example, wts = 1
    data=[  # written as sum of column scores
        [0.3 + 0.1 + 0.1, 0.3 + 0.9 + 0.9, 0.5 + 0.1 + 0.1],
        [1.8 + 0 + 0, 1.8 + 1.8 + 1.8, 0],
    ],
    dims=["forecaster", "obs_case"],
    coords={"forecaster": ["first", "second"], "obs_case": [0, 1, 2]},
)
EXP_RMS_CASE3 = xr.DataArray(  # Sydney example, wts escalate
    data=[  # written as sum of column scores
        [0.6 + 0 + 0.2, 0.6 + 2.1 + 1.8, 0.1 + 0 + 0.2],
        [1.5, 1.5 + 2.1 + 3.6, 0.0],
    ],
    dims=["forecaster", "obs_case"],
    coords={"forecaster": ["first", "second"], "obs_case": [0, 1, 2]},
)
DA_RMS_WEIGHTS_SYD = xr.DataArray(  # For Sydney example
    data=[1, 2, 3],
    dims=["obs_case"],
    coords={"obs_case": [0, 1, 2]},
)
EXP_RMS_CASE3A = xr.DataArray(  # Sydney example, wts escalate, no mean score
    data=[  # written as sum of column scores
        [0.8, 4.5, 0.3],
        [1.5, 7.2, 0.0],
    ],
    dims=["forecaster", "obs_case"],
    coords={"forecaster": ["first", "second"], "obs_case": [0, 1, 2]},
)
EXP_RMS_CASE3B = xr.DataArray(  # Sydney example, wts escalate, unweighted mean
    data=[  # written as sum of column scores
        (0.8 + 4.5 + 0.3) / 3,
        (1.5 + 7.2 + 0.0) / 3,
    ],
    dims=["forecaster"],
    coords={"forecaster": ["first", "second"]},
)
EXP_RMS_CASE3C = xr.DataArray(  # Sydney example, wts escalate, unweighted mean
    data=[  # written as sum of column scores
        (0.8 + 2 * 4.5 + 3 * 0.3) / 3,
        (1.5 + 2 * 7.2 + 0.0) / 3,
    ],
    dims=["forecaster"],
    coords={"forecaster": ["first", "second"]},
)
DA_RMS_WEIGHTS = xr.DataArray(
    data=[0.1, 0.2, 0.3],
    dims=["obs_case"],
    coords={"obs_case": [0, 1, 2]},
)
DA_RMS_WT3 = xr.DataArray(  # 3-dimensional decision weights, which is not allowed
    data=[
        [[2, 2], [3, 3], [0, 0]],
        [[0, 0], [2, 2], [3, 3]],
        [[1, 1], [0, 0], [2, 2]],
    ],
    dims=["prob", "sev", "i"],
    coords={
        "prob": [0.7, 0.4, 0.1],
        "sev": [1, 2, 3],
        "i": [0, 1],
    },
)
DA_RMS_WT4 = xr.DataArray(  # probability thresholds outside allowable range
    data=[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.1, 0.4, 1],
        "sev": [1, 2, 3],
    },
)
DA_RMS_WT5 = xr.DataArray(  # probability thresholds outside allowable range
    data=[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0, 0.4, 0.9],
        "sev": [1, 2, 3],
    },
)
DA_RMS_WT6 = xr.DataArray(
    data=[
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dims=["prob", "sev"],
    coords={
        "prob": [0.1, 0.4, 0.7],
        "sev": [1, 2, 4],
    },
)
DA_RMS_OBS2 = xr.DataArray(
    data=[
        [[1.0, 1.0, 0.0], [1, np.nan, np.nan]],
        [[1.0, 0, 0], [1.0, 0, 0]],
    ],
    dims=["stn", "day", "sev"],
    coords={
        "stn": [101, 102],
        "day": [6, 7],
        "sev": [1, 2, 4],
    },
)
EXP_DECISION_WEIGHT = xr.DataArray(
    data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
    dims=["prob", "sev"],
    coords={"prob": [0.7, 0.4, 0.2, 0.1], "sev": [0, 1, 2]},
)
