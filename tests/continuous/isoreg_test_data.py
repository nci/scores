"""Test data for `scores.isoreg`."""

from functools import partial

import numpy as np
import xarray as xr
from numpy import nan
from scipy import interpolate

# _xr_to_np test data

FCST_XRTONP = xr.DataArray(
    data=[
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    ],
    dims=["stn", "date", "leadday"],
    coords={"stn": [1001, 1002], "date": ["01", "02", "03"], "leadday": [1, 2, 3, 4]},
)

OBS_XRTONP = xr.DataArray(
    data=[
        [[1, 1, 1, 1], [2, 2, 2, 2]],
        [[4, 4, 4, 4], [0, 0, 0, 0]],
        [[7, 7, 7, 7], [3, 3, 3, 3]],
    ],
    dims=["date", "stn", "leadday"],
    coords={"stn": [1001, 1003], "date": ["01", "02", "03"], "leadday": [1, 2, 3, 4]},
)

EXP_FCST_XRTONP = np.array(  # dims=["stn", "date", "leadday"]
    [
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        [[nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]],
    ]
)

EXP_OBS_XRTONP = np.array(  # dims=["stn", "date", "leadday"]
    [
        [[1, 1, 1, 1], [4, 4, 4, 4], [7, 7, 7, 7]],
        [[nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]],
        [[2, 2, 2, 2], [0, 0, 0, 0], [3, 3, 3, 3]],
    ]
)

WT_XRTONP = xr.DataArray(
    data=[
        [[10], [15], [20]],
        [[25], [30], [35]],
    ],
    dims=["stn", "date", "leadday"],
    coords={"stn": [1001, 1002], "date": ["01", "02", "03"], "leadday": [2]},
)

EXP_WT_XRTONP = np.array(  # dims=["stn", "date", "leadday"]
    [
        [[nan, 10, nan, nan], [nan, 15, nan, nan], [nan, 20, nan, nan]],
        [[nan, 25, nan, nan], [nan, 30, nan, nan], [nan, 35, nan, nan]],
        [[nan, nan, nan, nan], [nan, nan, nan, nan], [nan, nan, nan, nan]],
    ]
)

EXP_XRTONP1 = (EXP_FCST_XRTONP, EXP_OBS_XRTONP, None)

EXP_XRTONP2 = (EXP_FCST_XRTONP, EXP_OBS_XRTONP, EXP_WT_XRTONP)


OBS_XRTONP2 = xr.DataArray(
    data=[[[10], [20], [30]]],
    dims=["date", "leadhour", "stn"],
    coords={"date": [0], "stn": [1], "leadhour": [1, 2, 3]},
)


# _tidy_ir_inputs data

FCST_TIDY1 = np.array([[1.0, 2.0, nan, 4.0], [2.0, 5.0, 2.0, 1.0]])
OBS_TIDY1 = np.array([[3.0, 1.0, 6.0, 7.0], [5.0, nan, -1.0, 0.0]])
WEIGHT_TIDY1 = np.array([[1, 2, 3, 4], [5, 6, 7, nan]])
EXP_TIDY1 = (
    np.array([1.0, 1.0, 2.0, 2.0, 2.0, 4.0]),
    np.array([3.0, 0.0, 5.0, 1.0, -1.0, 7.0]),
    None,
)
EXP_TIDY2 = (
    np.array([1.0, 2.0, 2.0, 2.0, 4.0]),
    np.array([3.0, 5.0, 1.0, -1.0, 7.0]),
    np.array([1, 5, 2, 7, 4]),
)

# data for _contiguous_ir tests

Y1 = np.array([1.0, 2, 3, 0, 5, 4, 1, 7])
W1 = np.array([1, 1, 1, 1, 2, 1, 1, 1])
EXP_IR_MEDIAN = np.array([1, 2, 2, 2, 4, 4, 4, 7])
EXP_IR_MEAN = np.array([1, 5 / 3, 5 / 3, 5 / 3, 10 / 3, 10 / 3, 10 / 3, 7])
EXP_IR_WMEAN = np.array([1, 5 / 3, 5 / 3, 5 / 3, 15 / 4, 15 / 4, 15 / 4, 7])


# data and working for bootstrap tests

BS_FCST = np.array([0.0, 1, 1, 3, 4, 5])
BS_OBS = np.array([2.0, 3, 1, 0, 2, 6])
BS_WT = np.array([1.0, 1, 1, 2, 1, 1])

# manual working with seed=1
# 1st boot sample: selection = [5, 3, 4, 0, 1, 3]
# samples then tidied:
#   fcst_sample = [5, 3, 4, 0, 1, 3] -> [0, 1, 3, 3, 4, 5]
#   obs_sample  = [6, 0, 2, 2, 3, 0] -> [2, 3, 0, 0, 2, 6]
#   wt_sample   = [1, 2, 1, 1, 1, 2] -> [1, 1, 2, 2, 1, 1]
# regression output y_reg:              [5/6, 5/6, 5/6, 5/6, 2, 6]
# interpolated to original fcst values: [5/6, 5/6, 5/6, 5/6, 2, 6]
# 2nd boot sample: selection = [5, 0, 0, 1, 4, 5]
# samples then tidied:
#   fcst_sample = [5, 0, 0, 1, 4, 5] -> [0, 0, 1, 4, 5, 5]
#   obs_sample  = [6, 2, 2, 3, 2, 6] -> [2, 2, 3, 2, 6, 6]
#   wt_sample   = [1, 1, 1, 1, 1, 1] -> [1, 1, 1, 1, 1, 1]
# regression output y_reg:              [2, 2, 5/2, 5/2, 6, 6]
# interpolated to original fcst values: [2, 5/2, 5/2, 5/2, 5/2, 6]
# 3rd boot sample: selection = [4, 1, 2, 4, 5, 2]
#   fcst_sample = [4, 1, 1, 4, 5, 1] -> [1, 1, 1, 4, 4, 5]
#   obs_sample  = [2, 3, 1, 2, 6, 1] -> [3, 1, 1, 2, 2, 6]
#   wt_sample   = [1, 1, 1, 1, 1, 1] -> [1, 1, 1, 1, 1, 1]
# regression output y_reg:              [5/3, 5/3, 5/3, 2, 2, 6]
# interpolated to original fcst values: [nan, 5/3, 5/3, 17/9, 2, 6]
BS_EXP1 = np.array(
    [
        [5 / 6, 5 / 6, 5 / 6, 5 / 6, 2, 6],
        [2, 5 / 2, 5 / 2, 5 / 2, 5 / 2, 6],
        [nan, 5 / 3, 5 / 3, 17 / 9, 2, 6],  # fcst=0 not in selection so gets NaN
    ]
)

# 1st selection yreg = [5/4, 5/4, 5/4, 5/4, 2, 6]
# 2nd and 3rd selections same results as BS_EXP1
BS_EXP2 = np.array(
    [
        [5 / 4, 5 / 4, 5 / 4, 5 / 4, 2, 6],
        [2, 5 / 2, 5 / 2, 5 / 2, 5 / 2, 6],
        [nan, 5 / 3, 5 / 3, 17 / 9, 2, 6],  # fcst=0 not in selection so gets NaN
    ]
)
BS_EXP3 = np.array([[1.0, 1, 1, 1, 2, 6]])


# data for _confidence_band test

CB_BOOT_INPUT = np.array(
    [
        [nan, 1.0, 1, 1, 6, 8, 9],
        [0, 3, 3, 3, 4, 7, 8],
        [nan, nan, -1, 3, 4, 6, 8],
        [0, 0, 3, 5, 6, 8, 9],
        [2, 5, 5, 5, 6, 6, 6],
    ]
)

EXP_CB = (
    np.array([nan, np.quantile([1.0, 3, 0, 5], 0.25), 1, 3, 4, 6, 8]),
    np.array([nan, np.quantile([1.0, 3, 0, 5], 0.75), 3, 5, 6, 8, 9]),
)


EXP_CB2 = {
    0.75: np.array([1, 3.5, 3, 5, 6, 8, 9]),
    0.25: np.array([0, 0.75, 1, 3, 4, 6, 8]),
}

# data for isotonic_fit tests
# we'll base this around the same data for bootstrap tests
# FCST_ARRAY and OBS_ARRAY are rearrangements of BS_FCST and BS_OBS with additional NaNs
FCST_ARRAY = np.array([[0.0, 3, 1, nan], [4, 1, 12, 5]])
OBS_ARRAY = np.array([[2.0, 0, 1, -30], [2, 3, nan, 6]])

DIMS = ["a", "b"]
COORDS = {"a": [1, 2], "b": [1, 2, 3, 4]}
FCST_XR = xr.DataArray(data=FCST_ARRAY, dims=DIMS, coords=COORDS)
OBS_XR = xr.DataArray(data=OBS_ARRAY, dims=DIMS, coords=COORDS)

# mean functional regression of BS_OBS (i.e., sorted OBS_ARRAY)
Y_REG = np.array([1.5, 1.5, 1.5, 1.5, 2, 6])
FCST_COUNTS = np.array([1, 2, 1, 1, 1])

TEST_POINTS = np.linspace(0.0, 5.0)

Q1 = np.quantile([5 / 4, 5 / 2, 5 / 3], (0.25, 0.75))
Q2 = np.quantile([5 / 4, 5 / 2, 17 / 9], (0.25, 0.75))
Q3 = np.quantile([2, 5 / 2, 2], (0.25, 0.75))
CB_LOWER = np.array([nan, Q1[0], Q1[0], Q2[0], Q3[0], 6])
CB_UPPER = np.array([nan, Q1[1], Q1[1], Q2[1], Q3[1], 6])

EXP_IF1 = {
    "fcst_sorted": np.unique(BS_FCST),
    "fcst_counts": FCST_COUNTS,
    "weight_sorted": None,
    "regression_values": np.delete(Y_REG, 1),
    "regression_func": interpolate.interp1d(BS_FCST, Y_REG, bounds_error=False),
    "confidence_band_lower_values": None,
    "confidence_band_upper_values": None,
    "confidence_band_lower_func": partial(np.full_like, fill_value=np.nan),
    "confidence_band_upper_func": partial(np.full_like, fill_value=np.nan),
    "confidence_band_levels": (None, None),
}

EXP_IF2 = {
    "fcst_sorted": np.unique(BS_FCST),
    "fcst_counts": FCST_COUNTS,
    "weight_sorted": None,
    "regression_values": np.delete(Y_REG, 1),
    "regression_func": interpolate.interp1d(BS_FCST, Y_REG, bounds_error=False),
    "confidence_band_lower_values": np.delete(CB_LOWER, 1),
    "confidence_band_upper_values": np.delete(CB_UPPER, 1),
    "confidence_band_lower_func": interpolate.interp1d(BS_FCST, CB_LOWER, bounds_error=False),
    "confidence_band_upper_func": interpolate.interp1d(BS_FCST, CB_UPPER, bounds_error=False),
    "confidence_band_levels": (0.25, 0.75),
}

EXP_IF3 = {**EXP_IF2, "bootstrap_results": BS_EXP2}
