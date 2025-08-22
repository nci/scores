"""
Contains test data for test_pit.py
"""

import numpy as np
import xarray as xr
from numpy import nan

# test data for _get_pit_x_values
DA_GPV = xr.DataArray(
    data=[[0.1, nan, 0.5], [0.8, nan, 0.5]],
    dims=["uniform_endpoint", "stn"],
    coords={"uniform_endpoint": ["lower", "upper"], "stn": [101, 102, 103]},
)
EXP_GPV = xr.DataArray(data=[0, 0.1, 0.5, 0.8, 1], dims=["pit_x_value"], coords={"pit_x_value": [0, 0.1, 0.5, 0.8, 1]})

# test data for _pit_cdfvalues_for_unif
EXP_PCVFU = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)

# test data for _pit_cdfvalues_for_jumps
EXP_PCVFJ_LEFT = xr.DataArray(
    data=[[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [0.0, 0, 0, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ_RIGHT = xr.DataArray(
    data=[[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ = {"left": EXP_PCVFJ_LEFT, "right": EXP_PCVFJ_RIGHT}


# test data for _pit_cdf_values
EXP_PCV_LEFT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 0, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCV_RIGHT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCV = {"left": EXP_PCV_LEFT, "right": EXP_PCV_RIGHT}


# test data for many functions, including the Pit class; mostly fcst, obs and calculating
# the PIT CDF for fcst and obs.
DA_FCST = xr.DataArray(
    data=[
        [[0.0, 0, 4, 2, 1], [0, 0, 0, 0, 1]],
        [[5, 3, 7, 2, 1], [nan, nan, nan, nan, nan]],
        [[2, 2, 5, 1, 2], [3, 2, 1, 4, 0]],
    ],
    dims=["stn", "lead_day", "ens_member"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "ens_member": [0, 1, 2, 3, 4]},
)
DA_OBS = xr.DataArray(data=[0, 4, nan], dims=["stn"], coords={"stn": [101, 102, 103]})
# keep all dims
EXP_PITCDF_LEFT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],  # Unif[0, 0.4], Unif[0, 0.8]
        [[0, 0, 0, 1, 1], [nan, nan, nan, nan, nan]],  # Unif[0.6, 0.6], nan
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],  # Unif[0, 0.4], Unif[0, 0.8]
        [[0, 0, 1, 1, 1], [nan, nan, nan, nan, nan]],  # Unif[0.6, 0.6], nan
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF1 = {"left": EXP_PITCDF_LEFT1, "right": EXP_PITCDF_RIGHT1}
# preserve lead day, weights=None
EXP_PITCDF_LEFT2 = xr.DataArray(
    data=[[0, 0.5, 0.5, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT2 = xr.DataArray(
    data=[[0, 0.5, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF2 = {"left": EXP_PITCDF_LEFT2, "right": EXP_PITCDF_RIGHT2}
# preserve lead day, station weights = [1, 2, 3]
WTS_STN = xr.DataArray(data=[1, 2, 3], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_PITCDF_LEFT3 = xr.DataArray(
    data=[[0, 1 / 3, 1 / 3, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT3 = xr.DataArray(
    data=[[0, 1 / 3, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF3 = {"left": EXP_PITCDF_LEFT3, "right": EXP_PITCDF_RIGHT3}
# reduce all dims, no weights
EXP_PITCDF_LEFT4 = xr.DataArray(
    data=[0, 0.5, 1.75 / 3, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT4 = xr.DataArray(
    data=[0, 0.5, 2.75 / 3, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF4 = {"left": EXP_PITCDF_LEFT4, "right": EXP_PITCDF_RIGHT4}


EXP_PLOTTING_POINTS2 = xr.DataArray(
    data=[0, 0, 0.5, 0.5, 1.75 / 3, 2.75 / 3, 1, 1, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0, 0, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1]},
)

# test data for plotting point functions
# case with several dimensions; left and right equal when pit_x_value is 0 or 3, NaNs involved
DA_GPP_LEFT1 = xr.DataArray(
    data=[[[1, 2, 2, 5], [0, 0, 3, 6], [1, 1, 2, 3]], [[1, 2, 2, 5], [0, 0, 3, 6], [nan, nan, nan, nan]]],
    dims=["lead_day", "stn", "pit_x_value"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "pit_x_value": [0, 1, 2, 3]},
)
DA_GPP_RIGHT1 = xr.DataArray(
    data=[[[1, 2, 3, 5], [0, 1, 3, 6], [1, 1, 2, 3]], [[1, 2, 3, 5], [0, 1, 3, 6], [nan, nan, nan, nan]]],
    dims=["lead_day", "stn", "pit_x_value"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "pit_x_value": [0, 1, 2, 3]},
)
EXP_GPP_X1 = xr.DataArray(
    data=[0, 1, 1, 2, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP_Y1 = xr.DataArray(
    data=[
        [[1, 2, 2, 2, 3, 5], [0, 0, 1, 3, 3, 6], [1, 1, 1, 2, 2, 3]],
        [[1, 2, 2, 2, 3, 5], [0, 0, 1, 3, 3, 6], [nan, nan, nan, nan, nan, nan]],
    ],
    dims=["lead_day", "stn", "plotting_point"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "plotting_point": [0, 1, 2, 3, 4, 5]},
)
EXP_GPP1 = {"x_plotting_position": EXP_GPP_X1, "y_plotting_position": EXP_GPP_Y1}
# case with only one dimension; left and right equal when pit_x_value is 2 or 3
DA_GPP_LEFT2 = xr.DataArray(data=[1, 2, 2, 5], dims=["pit_x_value"], coords={"pit_x_value": [0, 1, 2, 3]})
DA_GPP_RIGHT2 = xr.DataArray(data=[2, 2, 3, 5], dims=["pit_x_value"], coords={"pit_x_value": [0, 1, 2, 3]})
EXP_GPP_X2 = xr.DataArray(
    data=[0, 0, 1, 2, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP_Y2 = xr.DataArray(
    data=[1, 2, 2, 2, 3, 5], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP2 = {"x_plotting_position": EXP_GPP_X2, "y_plotting_position": EXP_GPP_Y2}
# when left and right always equal
EXP_GPP_X3 = xr.DataArray(data=[0, 1, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3]})
EXP_GPP_Y3 = xr.DataArray(data=[1, 2, 2, 5], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3]})
EXP_GPP3 = {"x_plotting_position": EXP_GPP_X3, "y_plotting_position": EXP_GPP_Y3}

# test data for histogram functions
DATA_CHV = [np.array([[0, 0, nan]]), np.array([[0.4, 0.7, nan]]), np.array([[1, 1, nan]])]
LIST_CHV1 = [
    xr.DataArray(data=dat, dims=["pit_x_value", "stn"], coords={"stn": [10, 11, 12], "pit_x_value": [x]})
    for dat, x in zip(DATA_CHV, [0, 0.5, 1])
]
EXP_CHV1 = xr.DataArray(
    data=[[0.4, 0.7, nan], [0.6, 0.3, nan]],
    dims=["bin_centre", "stn"],
    coords={
        "stn": [10, 11, 12],
        "bin_centre": [0.25, 0.75],
        "bin_left_endpoint": (["bin_centre"], [0, 0.5]),
        "bin_right_endpoint": (["bin_centre"], [0.5, 1]),
    },
)
DA_PH_LEFT = xr.DataArray(
    data=[[0, 0.2, 0.7, 1], [nan, nan, nan, nan]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [10, 11], "pit_x_value": [0, 0.5, 0.8, 1]},
)
DA_PH_RIGHT = xr.DataArray(
    data=[[0, 0.4, 0.7, 1], [nan, nan, nan, nan]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [10, 11], "pit_x_value": [0, 0.5, 0.8, 1]},
)
EXP_PHL1 = xr.DataArray(  # left endpoints of bins included
    data=[[0.2, 0.8], [nan, nan]],
    dims=["stn", "bin_centre"],
    coords={
        "stn": [10, 11],
        "bin_centre": [0.25, 0.75],
        "bin_left_endpoint": (["bin_centre"], [0, 0.5]),
        "bin_right_endpoint": (["bin_centre"], [0.5, 1]),
    },
)
EXP_PHR1 = xr.DataArray(  # right endpoints of bins included
    data=[[0.4, 0.6], [nan, nan]],
    dims=["stn", "bin_centre"],
    coords={
        "stn": [10, 11],
        "bin_centre": [0.25, 0.75],
        "bin_left_endpoint": (["bin_centre"], [0, 0.5]),
        "bin_right_endpoint": (["bin_centre"], [0.5, 1]),
    },
)
# right endpoints of bins included; use EXP_PITCDF_LEFT4, EXP_PITCDF_RIGHT4
EXP_HV1 = xr.DataArray(
    data=[0.25, 0.25, 2.75 / 3 - 0.5, 1 - 2.75 / 3, 0],
    dims=["bin_centre"],
    coords={
        "bin_centre": [0.1, 0.3, 0.5, 0.7, 0.9],
        "bin_left_endpoint": (["bin_centre"], [0.0, 0.2, 0.4, 0.6, 0.8]),
        "bin_right_endpoint": (["bin_centre"], [0.2, 0.4, 0.6, 0.8, 1]),
    },
)
# left endpoints of bins included; use EXP_PITCDF_LEFT4, EXP_PITCDF_RIGHT4
EXP_HV2 = xr.DataArray(
    data=[0.25, 0.25, 1.75 / 3 - 0.5, 1 - 1.75 / 3, 0],
    dims=["bin_centre"],
    coords={
        "bin_centre": [0.1, 0.3, 0.5, 0.7, 0.9],
        "bin_left_endpoint": (["bin_centre"], [0.0, 0.2, 0.4, 0.6, 0.8]),
        "bin_right_endpoint": (["bin_centre"], [0.2, 0.4, 0.6, 0.8, 1]),
    },
)

# test data for alpha score
DA_AS = xr.DataArray(
    data=[[0, 0, 0.3, 0.7, 1, 1], [0, 0, 0.6, 0.6, 1, 1], [nan, nan, nan, nan, nan, nan]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [3, 4, 5], "pit_x_value": [0, 0, 0.4, 0.4, 1, 1]},
)
EXP_AS = xr.DataArray(data=[(0.1 * 0.4 + 0.3 * 0.6) / 2, 0.2 / 2, nan], dims=["stn"], coords={"stn": [3, 4, 5]})

# test data for expected value
EXP_EV = xr.DataArray(
    data=[1 - (0.3 * 0.4 + 0.6 * (0.7 + 1)) / 2, 1 - (0.6 * 0.4 + 0.6 * (0.6 + 1)) / 2, nan],
    dims=["stn"],
    coords={"stn": [3, 4, 5]},
)

# test data for _variance_integral_term
DA_VIT = xr.DataArray(
    data=[
        [0, 0, 0.4, 0.4, 1, 1],  # integral(x(1-x))
        [0, 1, 1, 1, 1, 1],  # integral(x(1-1))
        [0, 0, 0, 0, 0, 1],  # integral(x(1-0))
        [0, 0, 0, 1, 1, 1],  # integral(x(1-0) on [0, 0.4]) + integral(x(1-1) on [0.4, 1])
        [0, 0, 1, 1, 1, 1],  # integral(x(1-5x/2) on [0, 0.4]) + integral(x(1-1) on [0.4, 1])
        [0, 0, 0.8, 0.8, 1, 1],  # integral(x(1-2x) on [0, 0.4]) + integral(x(1-((x-0.4)/3 + 0.8)) on [0.4, 1])
        [nan, nan, nan, nan, nan, nan],
    ],
    dims=["stn", "pit_x_value"],
    coords={"stn": [3, 4, 5, 6, 7, 8, 9], "pit_x_value": [0, 0, 0.4, 0.4, 1, 1]},
)
EXP_VIT = xr.DataArray(
    data=[
        1 / 6,
        0,
        0.5,
        (0.4**2) / 2,
        (0.4**2) / 2 - 5 * (0.4**3) / 6,
        (0.4**2) / 2 - 2 * (0.4**3) / 3 + (1 + 0.4 / 3 - 0.8) * (1 - 0.4**2) / 2 - (1 - 0.4**3) / 9,
        nan,
    ],
    dims=["stn"],
    coords={"stn": [3, 4, 5, 6, 7, 8, 9]},
)

# test data for _variance; uniform distribution, nans, point mass at 0, point mass at 1
DA_VAR = xr.DataArray(
    data=[[0, 0, 1, 1], [nan, nan, nan, nan], [0, 1, 1, 1], [0, 0, 0, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [3, 4, 5, 6], "pit_x_value": [0, 0, 1, 1]},
)
EXP__VAR = xr.DataArray([1 / 12, nan, 0, 0], dims=["stn"], coords={"stn": [3, 4, 5, 6]})
# simple forecast/obs example for .variance
DA_FCST_VAR = xr.DataArray(
    data=[
        [0, 0, 0, 0, 0],
        [0, 2, 5, 3, 7],
        [nan, nan, nan, nan, nan],
    ],
    dims=["stn", "member"],
    coords={"stn": [101, 102, 103], "member": [1, 2, 3, 4, 5]},
)
DA_OBS_VAR = xr.DataArray(data=[0, 4, 10], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_VAR = xr.DataArray(data=[1 / 12, 0, nan], dims=["stn"], coords={"stn": [101, 102, 103]})


# test data for _pit_values_for_cdf
DA_FCST_CDF_LEFT = xr.DataArray(
    data=[[0, 0.2, 0.5, 0.8, 1], [0, 0.1, 0.1, 0.9, 1], [0, 0, nan, 0.5, 0.9]],
    dims=["stn", "thld"],
    coords={"stn": [101, 102, 103], "thld": [0.0, 1, 2, 3, 4]},
)
DA_FCST_CDF_RIGHT = xr.DataArray(
    data=[[0, 0.2, 0.7, 0.8, 1], [0.1, 0.1, 0.1, 0.9, 1], [0, 0, nan, 0.5, 0.9]],
    dims=["stn", "thld"],
    coords={"stn": [101, 102, 103], "thld": [0.0, 1, 2, 3, 4]},
)
DA_OBS_PVCDF = xr.DataArray(
    data=[
        [1, 2, 2.5, 1.5],  # two obs between thresholds, two at thresholds
        [0, nan, -1, 5],  # nan obs, obs < thresholds, obs > thresholds
        [0, 0, 0, 0],
    ],
    dims=["stn", "instrument"],
    coords={"stn": [101, 102, 103], "instrument": [0, 1, 2, 4]},
)
EXP_PVCDF = xr.DataArray(
    data=[
        [[0.2, 0.5, 0.75, 0.35], [0, nan, 0, 1], [nan, nan, nan, nan]],  # lower
        [[0.2, 0.7, 0.75, 0.35], [0.1, nan, 0, 1], [nan, nan, nan, nan]],  # upper
    ],
    dims=["uniform_endpoint", "stn", "instrument"],
    coords={"stn": [101, 102, 103], "instrument": [0, 1, 2, 4], "uniform_endpoint": ["lower", "upper"]},
)
