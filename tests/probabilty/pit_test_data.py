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
# The next uses ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4
# PIT CDF crosses diagonal with the chord joining points (0.4, 0.5), (0.6, 1.75 / 3)
grad_chord = (1.75 / 3 - 0.5) / 0.2
intersection_pt = (0.5 - grad_chord * 0.4) / (1 - grad_chord)
EXP_AS1 = xr.DataArray(
    (
        0.1 * 0.4
        + 0.1 * (intersection_pt - 0.4)
        + (0.6 - 1.75 / 3) * (0.6 - intersection_pt)
        + (2.75 / 3 - 0.6 + 0.2) * 0.2
        + 0.2 * 0.2
    )
    / 2
)

# test data for expected value
EXP_EV = xr.DataArray(
    data=[1 - (0.3 * 0.4 + 0.6 * (0.7 + 1)) / 2, 1 - (0.6 * 0.4 + 0.6 * (0.6 + 1)) / 2, nan],
    dims=["stn"],
    coords={"stn": [3, 4, 5]},
)
# uses ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4
EXP_EXPVAL = xr.DataArray(
    data=[1 - (0.4 * 0.5 + 0.2 * (0.5 + 1.75 / 3) + 0.2 * (2.75 / 3 + 1) + 0.2 * (1 + 1)) / 2],
    dims=["blah"],
    coords={"blah": [1]},
).mean("blah")

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


# test data for _value_at_pit_cdf
EXP_VAPC1 = xr.DataArray(
    data=[(0.5 + 1.75 / 3) / 2],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.5]},
)
EXP_VAPC2 = xr.DataArray(
    data=[[(4 / 7 + 1) / 2, nan, 1]],
    dims=["pit_x_value", "stn"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0.65]},
)

# test data for _pit_values_for_cdf, _pit_values_for_cdf_array
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
EXP__PVCDF = xr.DataArray(  # uniform distribution format
    data=[
        [[0.2, 0.5, 0.75, 0.35], [0, nan, 0, 1], [nan, nan, nan, nan]],  # lower
        [[0.2, 0.7, 0.75, 0.35], [0.1, nan, 0, 1], [nan, nan, nan, nan]],  # upper
    ],
    dims=["uniform_endpoint", "stn", "instrument"],
    coords={"stn": [101, 102, 103], "instrument": [0, 1, 2, 4], "uniform_endpoint": ["lower", "upper"]},
)
# test data for _pit_values_for_cdf_array warning tests
DA_FCST_WARN1 = xr.DataArray(
    data=[[0, 0.2, 0.5, 0.8, 1], [0, 0.1, 0.1, 0.9, 1], [0, 0, 1, 0.5, 0.9]],
    dims=["stn", "thld"],
    coords={"stn": [101, 102, 103], "thld": [0.0, 1, 2, 3, 4]},
)
DA_FCST_WARN2 = xr.DataArray(
    data=[[0, 0.2, 0.5, 0.8, 1], [0, nan, 0.1, 0.9, 1], [0, 0, 1, 0.5, 0.9]],
    dims=["stn", "thld"],
    coords={"stn": [101, 102, 103], "thld": [-1, 1, 2, 3, 4.5]},
)
DA_OBS_WARN1 = xr.DataArray(data=[1, 2, 4.1], dims=["stn"], coords={"stn": [101, 102, 103]})
DA_OBS_WARN2 = xr.DataArray(data=[-0.1, 2, 4], dims=["stn"], coords={"stn": [101, 102, 103]})

# data data for pit_distribution_for_cdf
DA_FCST_CDF_LEFT1 = xr.DataArray(
    data=[
        [[0, 0.2, 0.5, 1], [0, 0, 0.6, 1]],
        [[0, 0, 0, 1], [nan, nan, 1, nan]],
        [[0, 0.8, 1, 1], [0, 0.5, 0.9, 1]],
    ],
    dims=["stn", "lead_day", "thld"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "thld": [0.0, 1, 2, 3]},
)
DA_FCST_CDF_RIGHT1 = xr.DataArray(
    data=[
        [[0, 0.2, 0.5, 1], [0, 0, 0.8, 1]],  # obs=2: uniform [0.5, 0.5], [0.6, 0.8]
        [[0, 0, 1, 1], [0, 0, 1, 1]],  # obs=1: uniform [0, 0], nan
        [[0, 0.8, 1, 1], [0, 0.5, 0.9, 1]],  # obs is nan
    ],
    dims=["stn", "lead_day", "thld"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "thld": [0.0, 1, 2, 3]},
)
DA_OBS_PDCDF = xr.DataArray(data=[2.0, 1, nan], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_PDCDF_LEFT1 = xr.DataArray(
    data=[
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
        [[0, 1, 1, 1, 1], [nan, nan, nan, nan, nan]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "pit_x_value": [0, 0.5, 0.6, 0.8, 1]},
)
EXP_PDCDF_RIGHT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0, 0, 1, 1]],
        [[1, 1, 1, 1, 1], [nan, nan, nan, nan, nan]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "pit_x_value": [0, 0.5, 0.6, 0.8, 1]},
)
EXP_PDCDF1 = {"left": EXP_PDCDF_LEFT1, "right": EXP_PDCDF_RIGHT1}
EXP_PDCDF_LEFT2 = xr.DataArray(
    data=[
        [[0, 0, 1, 1], [0, 0, 0, 1]],
        [[0, 1, 1, 1], [nan, nan, nan, nan]],
        [[nan, nan, nan, nan], [nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "pit_x_value": [0, 0.5, 0.6, 1]},
)
EXP_PDCDF_RIGHT2 = xr.DataArray(
    data=[
        [[0, 1, 1, 1], [0, 0, 1, 1]],
        [[1, 1, 1, 1], [nan, nan, nan, nan]],
        [[nan, nan, nan, nan], [nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [1, 2], "pit_x_value": [0, 0.5, 0.6, 1]},
)
EXP_PDCDF2 = {"left": EXP_PDCDF_LEFT2, "right": EXP_PDCDF_RIGHT2}
EXP_PDCDF_LEFT3 = xr.DataArray(
    data=[0, 1 / 3, 2 / 3, 1], dims=["pit_x_value"], coords={"pit_x_value": [0, 0.5, 0.6, 1]}
)
EXP_PDCDF_RIGHT3 = xr.DataArray(
    data=[1 / 3, 2 / 3, 1, 1], dims=["pit_x_value"], coords={"pit_x_value": [0, 0.5, 0.6, 1]}
)
EXP_PDCDF3 = {"left": EXP_PDCDF_LEFT3, "right": EXP_PDCDF_RIGHT3}
DA_FCST_CDF_LEFT_RAISES = xr.DataArray(
    data=[
        [[0, 0.2, 0.5, 1], [0, 0, 0.6, 1]],
        [[0, 0, 0, 1], [nan, nan, 1, nan]],
    ],
    dims=["stn", "lead_day", "thld"],
    coords={"stn": [101, 102], "lead_day": [1, 2], "thld": [0.0, 1, 2, 3]},
)

# test data for _right_left_checks
DA_RLC1 = xr.DataArray(data=[0, 0.4, 0.2, nan, 0.4], dims=["thld"], coords={"thld": [1, 3, 2, 5, 3]})
DA_RLC2 = xr.DataArray(data=[0, 0.4, nan, 0.6, 0.4], dims=["thld"], coords={"thld": [1, 2, 3, 4, 5]})
DA_RLC3 = xr.DataArray(data=[0, 0.2, 0.8, nan, 0.41], dims=["thld"], coords={"thld": [1, 2, 3, 4, 5]})


# test data for _pit_values_for_fcst_at_obs
DA_FAO = xr.DataArray(
    data=[[0.2, 0.4, nan], [0.4, 0.6, 1]], dims=["stn", "lead_day"], coords={"stn": [101, 102], "lead_day": [0, 1, 2]}
)
DA_FAO_LEFT = xr.DataArray(
    data=[[0, 0.4, nan], [0.4, 0.2, 1]], dims=["stn", "lead_day"], coords={"stn": [101, 102], "lead_day": [0, 1, 2]}
)
# keep all dims with fcst_at_obs_left = DA_FOA_LEFT
EXP_PVFAO_LEFT0 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [nan, nan, nan, nan, nan]],
        [[0, 0, 0, 1, 1], [0, 0, 0.5, 1, 1], [0, 0, 0, 0, 0]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102], "lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO_RIGHT0 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [nan, nan, nan, nan, nan]],
        [[0, 0, 1, 1, 1], [0, 0, 0.5, 1, 1], [0, 0, 0, 0, 1]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102], "lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO0 = {"left": EXP_PVFAO_LEFT0, "right": EXP_PVFAO_RIGHT0}
# keep all dims, fcst_at_obs_left is None
EXP_PVFAO_LEFT1 = xr.DataArray(
    data=[
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [nan, nan, nan, nan, nan]],
        [[0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102], "lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO_RIGHT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [nan, nan, nan, nan, nan]],
        [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102], "lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO1 = {"left": EXP_PVFAO_LEFT1, "right": EXP_PVFAO_RIGHT1}
# preserve lead day with weight, fcst_at_obs_left is None
DA_WT = xr.DataArray(data=[1, 3], dims=["stn"], coords={"stn": [101, 102]})
EXP_PVFAO_LEFT2 = xr.DataArray(
    data=[[0, 0, 0.25, 1, 1], [0, 0, 0, 0.25, 1], [0, 0, 0, 0, 0]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO_RIGHT2 = xr.DataArray(
    data=[[0, 0.25, 1, 1, 1], [0, 0, 0.25, 1, 1], [0, 0, 0, 0, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1, 2], "pit_x_value": [0, 0.2, 0.4, 0.6, 1]},
)
EXP_PVFAO2 = {"left": EXP_PVFAO_LEFT2, "right": EXP_PVFAO_RIGHT2}


# test data for _pit_values_final_processing
DA_PVFP = xr.DataArray(
    data=[
        [[1, 1], [nan, nan]],
        [[0.2, 0.2], [0, 0]],
        [[nan, nan], [0, 0.2]],
    ],
    dims=["stn", "lead_day", "uniform_endpoint"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "uniform_endpoint": ["lower", "upper"]},
)
DA_PVPF_WTS = xr.DataArray(data=[1, 4, 2], dims=["stn"], coords={"stn": [101, 102, 103]})
# results with weights
EXP_PVFP_LEFT = xr.DataArray(
    data=[[0, 0, 0.8], [0, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0.2, 1]},
)
EXP_PVFP_RIGHT = xr.DataArray(
    data=[[0, 0.8, 1], [2 / 3, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0.2, 1]},
)
# results with weights None
EXP_PVFP_LEFT1 = xr.DataArray(
    data=[[0, 0, 0.5], [0, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0.2, 1]},
)
EXP_PVFP_RIGHT1 = xr.DataArray(
    data=[[0, 0.5, 1], [0.5, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0.2, 1]},
)

# test data for simple tests of Pit_fcst_at_obs methods
DA_FCST_AT_OBS = xr.DataArray(data=[0.4, 0.8], dims=["time"], coords={"time": [10, 11]})
EXP_FAO_PP = xr.DataArray(
    data=[0, 0, 0, 0.5, 0.5, 1, 1, 1], dims=["pit_x_value"], coords={"pit_x_value": [0, 0, 0.4, 0.4, 0.8, 0.8, 1, 1]}
)
EXP_FAO_PPP = {
    "x_plotting_position": xr.DataArray(
        [0, 0.4, 0.4, 0.8, 0.8, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
    ),
    "y_plotting_position": xr.DataArray(
        [0, 0, 0.5, 0.5, 1, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
    ),
}
EXP_FAO_HV = xr.DataArray(
    data=[0.5, 0.5],
    dims=["bin_centre"],
    coords={
        "bin_centre": [0.25, 0.75],
        "bin_left_endpoint": (["bin_centre"], [0, 0.5]),
        "bin_right_endpoint": (["bin_centre"], [0.5, 1]),
    },
)

# test data for _get_plotting_points, .plotting_points, etc
EXP_PP1 = xr.DataArray(  # uses EXP_PITCDF_LEFT2, EXP_PITCDF_RIGHT2
    data=[[0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1], [0, 0, 0.5, 0.5, 0.75, 0.75, 1, 1, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1]},
)
EXP_PPP = {
    "x_plotting_position": xr.DataArray(
        data=[0, 0.4, 0.6, 0.6, 0.8, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
    ),
    "y_plotting_position": xr.DataArray(
        data=[0, 0.5, 1.75 / 3, 2.75 / 3, 1, 1],
        dims=["plotting_point"],
        coords={"plotting_point": [0, 1, 2, 3, 4, 5]},
    ),
}

# test data for _diagonal_intersection_points
# example testing 5 scenarios
DA_PPP_Y = xr.DataArray(
    [
        [0, 0, 0, 0, 1, 1, 1, 1],  # crosses diagonal from below at 10/35; all continuous
        [0, 0, 0, 0.6, 0.6, 0.6, 1, 1],  # step function; 1 horizontal crossing at 0.6; 2 vertical
        [0, 0.3, 0.3, 0.3, 0.4, 0.6, 0.6, 1],  # crosses diagonal from above at 21/60; all continuous
        [nan, nan, nan, nan, nan, nan, nan, nan],
    ],
    dims=["stn", "plotting_point"],
    coords={"stn": [101, 102, 103, 104], "plotting_point": [0, 1, 2, 3, 4, 5, 6, 7]},
)
DA_PPP_X = xr.DataArray(
    [0, 0, 0.2, 0.2, 0.5, 0.8, 0.8, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5, 6, 7]}
)
DICT_DIP = {"x_plotting_position": DA_PPP_X, "y_plotting_position": DA_PPP_Y}
# example test empty output
DA_PPP_Y2 = xr.DataArray([0, 0, 0.2, 0.2, 1, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]})
DA_PPP_X2 = xr.DataArray(
    [0, 0, 0.2, 0.2, 1, 1],
    dims=["plotting_point"],
    coords={
        "plotting_point": [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    },
)
DICT_DIP2 = {"x_plotting_position": DA_PPP_X2, "y_plotting_position": DA_PPP_Y2}


# test data alpha_score_array
# this data has parametric plotting points DICT_DIP
DA_ASA_LEFT = xr.DataArray(
    data=[
        [0, 0, 1, 1, 1],
        [0, 0, 0.6, 0.6, 1],
        [0, 0.3, 0.4, 0.6, 1],
        [nan, nan, nan, nan, nan],
    ],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103, 104], "pit_x_value": [0, 0.2, 0.5, 0.8, 1]},
)
DA_ASA_RIGHT = xr.DataArray(
    data=[
        [0, 0, 1, 1, 1],
        [0, 0.6, 0.6, 1, 1],
        [0.3, 0.3, 0.4, 0.6, 1],
        [nan, nan, nan, nan, nan],
    ],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103, 104], "pit_x_value": [0, 0.2, 0.5, 0.8, 1]},
)
EXP_ASA = xr.DataArray(
    data=[
        (0.2 * (10 / 35) + 0.5 * (1 - 10 / 35)) / 2,
        (0.2 * 0.2 + 0.4 * 0.4 + 0.2 * 0.2 + 0.2 * 0.2) / 2,
        ((0.3 + 0.1) * 0.2 + 0.1 * (21 / 60 - 0.2) + (0.5 - 21 / 60) * 0.1 + (0.1 + 0.2) * 0.3 + 0.2 * 0.2) / 2,
        nan,
    ],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104]},
)
