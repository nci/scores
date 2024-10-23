"""
Generation of the test data used to test scores.probability.crps
"""

import numpy as np
import xarray as xr
from numpy import nan

# pylint disable=no-name-in-module
from scores.processing.cdf import add_thresholds

DA_ISPL1 = xr.DataArray(  # simple case, easy to calculate
    data=[[0, 1], [1, 1], [nan, 1]],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003], "x": [0, 1]},
)

EXP_ISPL1 = xr.DataArray(data=[1 / 3, 1, nan], dims=["station"], coords={"station": [1001, 1002, 1003]})

DA_ISPL2 = xr.DataArray(  # three thresholds, evenly spaced
    data=[[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2]},
)

EXP_ISPL2 = xr.DataArray(
    data=[2 / 3, 4 / 3, 8 / 3, 1 / 3],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004]},
)

DA_ISPL3 = xr.DataArray(  # three thresholds, not evenly spaced
    data=[[0, 0.5, 2], [0, 0.5, 0.5]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0, 0.5, 2]},
)

EXP_ISPL3 = xr.DataArray(
    data=[8 / 3, 1 / 24 + 1.5 * 0.5**2],
    dims=["station"],
    coords={"station": [1001, 1002]},
)

DA_STEP_WEIGHT = xr.DataArray(data=[2.31, 5.76, 0.89], dims=["station"], coords={"station": [1001, 1002, 1003]})

EXP_STEP_WEIGHT_UPPER = xr.DataArray(
    data=[
        [0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0, 0, 0, 0, 0, 0, 0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ],
    dims=["station", "x"],
    coords={"x": [0.8, 1, 2, 2.4, 3, 4, 5, 5.8, 6], "station": [1001, 1002, 1003]},
)

EXP_STEP_WEIGHT_LOWER = xr.DataArray(
    data=[
        [1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dims=["station", "x"],
    coords={"x": [0.8, 1, 2, 2.4, 3, 4, 5, 5.8, 6], "station": [1001, 1002, 1003]},
)

DA_FCST_CRPS_EXACT = xr.DataArray(
    data=[
        [0, 1, nan, 1],  # fcst has nan
        [0, 1, 1, 1],  # obs has nan
        [0, 1, 1, 1],  # weight has nan
        [0.2, 0.5, 0.8, 1],  # obs is .5, wts are 1
        [0.2, 0.5, 0.8, 1],  # obs is .5, wts are 1 when threshold >= 1
        [0.2, 0.5, 0.8, 1],  # obs is .5, wts are 1 when threshold < 1
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006], "x": [0, 0.5, 1, 2]},
)

DA_WT_CRPS_EXACT = xr.DataArray(
    data=[
        [1, 1, 1, 1],  # fcst has nan
        [1, 1, 1, 1],  # obs has nan
        [nan, 1, 1, 1],  # weight has nan
        [1, 1, 1, 1],  # obs is .5, wts are 1
        [0, 0, 1, 1],  # obs is .5, wts are 1 when threshold >= 1
        [1, 1, 0, 0],  # obs is .5, wts are 1 when threshold < 1
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006], "x": [0, 0.5, 1, 2]},
)

DA_OBS_CRPS_EXACT = xr.DataArray(
    data=[
        [0, 1, 1, 1],  # fcst has nan
        [0, 1, nan, 1],  # obs has a nan
        [0, 1, 1, 1],  # weight has nan
        [0, 1, 1, 1],  # obs is .5, wts are 1
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006], "x": [0, 0.5, 1, 2]},
)


# manually calculated expected outputs using calculus:
C1 = (0.5**3 - 0.2**3) / 1.8  # integral from 0 to .5
C2 = (0.5**3 - 0.2**3) / 1.8  # integral from .5 to 1
C3 = 0.2**2 / 3  # integral from 1 to 2

EXP_OVER_CRPS_EXACT = xr.DataArray(
    data=[nan, nan, nan, C2 + C3, C3, C2],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="overforecast_penalty",
)

EXP_UNDER_CRPS_EXACT = xr.DataArray(
    data=[nan, nan, nan, C1, 0, C1],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="underforecast_penalty",
)

EXP_TOTAL_CRPS_EXACT = EXP_OVER_CRPS_EXACT + EXP_UNDER_CRPS_EXACT

EXP_CRPS_EXACT = xr.merge([EXP_TOTAL_CRPS_EXACT.rename("total"), EXP_UNDER_CRPS_EXACT, EXP_OVER_CRPS_EXACT])


NEW_THRESHOLDS = np.linspace(0, 2, 200001)

DA_FCST_CRPS_DENSE = add_thresholds(DA_FCST_CRPS_EXACT, "x", NEW_THRESHOLDS, "linear").where(
    DA_FCST_CRPS_EXACT["station"] != 1001
)

DA_OBS_CRPS_DENSE = add_thresholds(DA_OBS_CRPS_EXACT, "x", NEW_THRESHOLDS, "step").where(
    DA_OBS_CRPS_EXACT["station"] != 1002
)

DA_WT_CRPS_DENSE = add_thresholds(DA_WT_CRPS_EXACT, "x", NEW_THRESHOLDS, "step").where(
    DA_WT_CRPS_EXACT["station"] != 1003
)


DA_FCST_REFORMAT1 = xr.DataArray(
    data=[
        [[0, 0.2, 0.6, 0.9], [0.5, 0.7, 1, 1]],
        [[nan, nan, nan, nan], [0.5, 0.7, 1, 1]],
        [[0, 0.2, 0.6, 0.9], [0.5, 0.7, nan, 1]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11], "x": [0, 1, 2, 3]},
)

DA_OBS_REFORMAT1 = xr.DataArray(
    data=[
        [1.0, 2.5],
        [3.0, nan],
        [0.0, 2.0],
    ],
    dims=["station", "date"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11]},
)

DA_WT_REFORMAT1 = xr.DataArray(
    data=[
        [0, 0, 0, 1.0],
        [0, 1.0, 1.0, 1.0],
        [0, 0, 1.0, 1.0],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003], "x": [0, 0.5, 1, 2]},
)

EXP_FCST_REFORMAT1 = xr.DataArray(  # "linear" fill
    data=[
        [[0, 0.1, 0.2, 0.6, 0.75, 0.9], [0.5, 0.6, 0.7, 1, 1, 1]],
        [[nan, nan, nan, nan, nan, nan], [0.5, 0.6, 0.7, 1, 1, 1]],
        [[0, 0.1, 0.2, 0.6, 0.75, 0.9], [0.5, 0.6, 0.7, 0.85, 0.925, 1]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 2, 2.5, 3],
    },
)

EXP_OBS_REFORMAT1 = xr.DataArray(  # "step" fill
    data=[
        [[0, 0, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 1.0, 1.0]],
        [[0, 0, 0, 0, 0, 1.0], [nan, nan, nan, nan, nan, nan]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 2, 2.5, 3],
    },
)

EXP_WT_REFORMAT1 = xr.DataArray(  # "forward" fill
    data=[
        [[0, 0, 0, 1.0, 1.0, 1.0], [0, 0, 0, 1.0, 1.0, 1.0]],
        [[0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[0, 0, 1.0, 1.0, 1.0, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 2, 2.5, 3],
    },
)

EXP_REFORMAT1 = EXP_FCST_REFORMAT1, EXP_OBS_REFORMAT1, EXP_WT_REFORMAT1

# additional thresholds: [0, 1.5]
EXP_FCST_REFORMAT2 = xr.DataArray(  # "linear" fill
    data=[
        [[0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.9], [0.5, 0.6, 0.7, 0.85, 1, 1, 1]],
        [[nan, nan, nan, nan, nan, nan, nan], [0.5, 0.6, 0.7, 0.85, 1, 1, 1]],
        [[0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.9], [0.5, 0.6, 0.7, 0.775, 0.85, 0.925, 1]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    },
)

EXP_OBS_REFORMAT2 = xr.DataArray(  # "step" fill
    data=[
        [[0, 0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 0, 1.0, 1.0]],
        [[0, 0, 0, 0, 0, 0, 1.0], [nan, nan, nan, nan, nan, nan, nan]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    },
)

EXP_WT_REFORMAT2 = xr.DataArray(  # "forward" fill
    data=[
        [[0, 0, 0, 0, 1.0, 1.0, 1.0], [0, 0, 0, 0, 1.0, 1.0, 1.0]],
        [[0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[0, 0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "date": [10, 11],
        "x": [0, 0.5, 1, 1.5, 2, 2.5, 3],
    },
)

EXP_REFORMAT2 = EXP_FCST_REFORMAT2, EXP_OBS_REFORMAT2, EXP_WT_REFORMAT2

# weight = None
EXP_FCST_REFORMAT3 = xr.DataArray(  # "linear" fill
    data=[
        [[0, 0.2, 0.6, 0.75, 0.9], [0.5, 0.7, 1, 1, 1]],
        [[nan, nan, nan, nan, nan], [0.5, 0.7, 1, 1, 1]],
        [[0, 0.2, 0.6, 0.75, 0.9], [0.5, 0.7, 0.85, 0.925, 1]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11], "x": [0, 1, 2, 2.5, 3]},
)

EXP_OBS_REFORMAT3 = xr.DataArray(  # "step" fill
    data=[
        [[0, 1.0, 1.0, 1.0, 1.0], [0, 0, 0, 1.0, 1.0]],
        [[0, 0, 0, 0, 1.0], [nan, nan, nan, nan, nan]],
        [[1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11], "x": [0, 1, 2, 2.5, 3]},
)

EXP_WT_REFORMAT3 = xr.DataArray(  # "forward" fill
    data=[
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11], "x": [0, 1, 2, 2.5, 3]},
)

EXP_REFORMAT3 = EXP_FCST_REFORMAT3, EXP_OBS_REFORMAT3, EXP_WT_REFORMAT3

DA_WT_CHECK_CRPS1 = xr.DataArray(
    data=[
        [0, 0, 0, 1.0],
        [0, 1.0, 1.0, 1.0],
        [0, 0, 1.0, 1.0],
    ],
    dims=["station", "y"],
    coords={"station": [1001, 1002, 1003], "y": [0, 0.5, 1, 2]},
)

DA_WT_CHECK_CRPS2 = xr.DataArray(
    data=[
        [0, 0, 0, 1.0],
        [0, 1.0, 1.0, 1.0],
        [0, 0, 1.0, 1.0],
    ],
    dims=["unicorns", "x"],
    coords={"unicorns": [1001, 1002, 1003], "x": [0, 0.5, 1, 2]},
)

DA_WT_CHECK_CRPS3 = xr.DataArray(  # x not increasing
    data=[[0.2, -2.0, 0.8, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 0.5, 1, 2]},
)

DA_OBS_CHECK_CRPS = xr.DataArray(
    data=[
        [0, 0, 0, 1.0],
        [0, 1.0, 1.0, 1.0],
        [0, 0, 1.0, 1.0],
    ],
    dims=["vehicle", "station"],
    coords={"vehicle": [1001, 1002, 1003], "station": [0, 0.5, 1, 2]},
)

DA_FCST_CHECK_CRPS = xr.DataArray(
    data=[
        [[0], [0.5]],
        [[nan], [0.5]],
        [[0], [0.5]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [1001, 1002, 1003], "date": [10, 11], "x": [0]},
)

DA_FCST_CHECK_CRPS2 = xr.DataArray(  # x not increasing
    data=[[0.2, 0.5, 0.8, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 0.5, 10, 2]},
)

DA_FCST_CHECK_CRPS2A = xr.DataArray(  # x not increasing
    data=[[0.2, 0.5, 0.8, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 0.5, 1, 2]},
)

DA_OBS_CHECK_CRPS2 = xr.DataArray(
    data=[0.2],
    dims=["station"],
    coords={"station": [1001]},
)

DA_FCST_CRPS = DA_FCST_CRPS_EXACT.copy()

DA_WT_CRPS = DA_WT_CRPS_EXACT.copy()

DA_OBS_CRPS = xr.DataArray(
    data=[0.5, nan, 0.5, 0.5, 0.5, 0.5],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
)

EXP_CRPS1 = EXP_CRPS_EXACT.copy()

EXP_OVER_CRPS_MEAN = xr.DataArray(
    data=(C2 + C3 + C3 + C2) / 3,
    name="overforecast_penalty",
)

EXP_UNDER_CRPS_MEAN = xr.DataArray(
    data=(C1 + C1) / 3,
    name="underforecast_penalty",
)

EXP_TOTAL_CRPS_MEAN = EXP_OVER_CRPS_MEAN + EXP_UNDER_CRPS_MEAN

EXP_CRPS2 = xr.merge([EXP_TOTAL_CRPS_MEAN.rename("total"), EXP_UNDER_CRPS_MEAN, EXP_OVER_CRPS_MEAN])

# weight = 1, propagate nans
EXP_OVER_CRPS3 = xr.DataArray(
    data=[nan, nan, 0, C2 + C3, C2 + C3, C2 + C3],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="overforecast_penalty",
)

EXP_UNDER_CRPS3 = xr.DataArray(
    data=[nan, nan, 1 / 6, C1, C1, C1],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="underforecast_penalty",
)

EXP_TOTAL_CRPS3 = EXP_OVER_CRPS3 + EXP_UNDER_CRPS3

EXP_CRPS3 = xr.merge([EXP_TOTAL_CRPS3.rename("total"), EXP_UNDER_CRPS3, EXP_OVER_CRPS3])

# weight = 1, don't propagate nans
EXP_OVER_CRPS4 = xr.DataArray(
    data=[0, nan, 0, C2 + C3, C2 + C3, C2 + C3],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="overforecast_penalty",
)

EXP_UNDER_CRPS4 = xr.DataArray(
    data=[1 / 6, nan, 1 / 6, C1, C1, C1],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="underforecast_penalty",
)

EXP_TOTAL_CRPS4 = EXP_OVER_CRPS4 + EXP_UNDER_CRPS4

EXP_CRPS4 = xr.merge([EXP_TOTAL_CRPS4.rename("total"), EXP_UNDER_CRPS4, EXP_OVER_CRPS4])

# don't propagate nans
EXP_OVER_CRPS5 = xr.DataArray(
    data=[0, nan, 0, C2 + C3, C3, C2],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="overforecast_penalty",
)

EXP_UNDER_CRPS5 = xr.DataArray(
    data=[1 / 6, nan, 1 / 6, C1, 0, C1],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
    name="underforecast_penalty",
)

EXP_TOTAL_CRPS5 = EXP_OVER_CRPS5 + EXP_UNDER_CRPS5

EXP_CRPS5 = xr.merge([EXP_TOTAL_CRPS5.rename("total"), EXP_UNDER_CRPS5, EXP_OVER_CRPS5])

DA_FCST_ADJUST1 = xr.DataArray(
    data=[
        [0, 0.5, 1],
        [0, nan, 1],
        [1, 0.5, 0],
        [1, 0.5, 0],
        [1, 0.5, 0],
        [nan, nan, nan],
        [1, 0.5, 0],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006, 1007], "x": [0, 0.5, 1]},
)

DA_OBS_ADJUST1 = xr.DataArray(
    data=[0.5, 0.5, 0, 0.5, 1, 0.2, nan],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006, 1007]},
)

EXP_FCST_ADJUST1 = xr.DataArray(  # decreasing_tolerance=0
    data=[
        [0, 0.5, 1],
        [nan, nan, nan],
        [0, 0, 0],
        [1, 0.5, 0],
        [1, 1, 1],
        [nan, nan, nan],
        [1, 0.5, 0],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006, 1007], "x": [0, 0.5, 1]},
)

DA_FCST_ADJUST2 = xr.DataArray(
    data=[
        [0, 0.5, 1],
        [1, 0.5, 0],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0, 0.5, 1]},
)

DA_OBS_ADJUST2 = xr.DataArray(
    data=[0.5, 0.5],
    dims=["station"],
    coords={"station": [1001, 1002]},
)

EXP_FCST_ADJUST2 = xr.DataArray(  # decreasing_tolerance=10
    data=[
        [0, 0.5, 1],
        [1, 0.5, 0],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0, 0.5, 1]},
)

DA_FCST_CRPS_BD = xr.DataArray(
    data=[[0, 0.5, 1], [0, 0.2, 0.6], [0.5, 1, 1], [1, 1, 1], [nan, 0.7, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005], "x": [0, 0.5, 1]},
)

DA_OBS_CRPS_BD = xr.DataArray(
    data=[0, 0.5, 1, nan, 0],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005]},
)

EXP_UNDER_CRPS_BD1 = xr.DataArray(
    data=[(0 + 0 + 0.25) / 3, (0 + 0 + 1) / 3, (0 + 0 + 0) / 3],
    dims=["x"],
    coords={"x": [0, 0.5, 1]},
    name="underforecast_penalty",
)

EXP_OVER_CRPS_BD1 = xr.DataArray(
    data=[(1 + 0 + 0) / 3, (0.25 + 0.64 + 0) / 3, (0 + 0.16 + 0) / 3],
    dims=["x"],
    coords={"x": [0, 0.5, 1]},
    name="overforecast_penalty",
)

EXP_TOTAL_CRPS_BD1 = xr.DataArray(
    data=[(1 + 0 + 0.25) / 3, (0.25 + 0.64 + 1) / 3, (0 + 0.16 + 0) / 3],
    dims=["x"],
    coords={"x": [0, 0.5, 1]},
    name="total_penalty",
)

EXP_CRPS_BD1 = xr.merge([EXP_TOTAL_CRPS_BD1, EXP_UNDER_CRPS_BD1, EXP_OVER_CRPS_BD1])

EXP_UNDER_CRPS_BD2 = xr.DataArray(  # don't collapse station
    data=[[0, 0, 0], [0, 0, 0], [0.25, 1, 0], [nan, nan, nan], [nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005], "x": [0, 0.5, 1]},
    name="underforecast_penalty",
)

EXP_OVER_CRPS_BD2 = xr.DataArray(  # don't collapse station
    data=[[1, 0.25, 0], [0, 0.64, 0.16], [0, 0, 0], [nan, nan, nan], [nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005], "x": [0, 0.5, 1]},
    name="overforecast_penalty",
)

EXP_TOTAL_CRPS_BD2 = xr.DataArray(  # don't collapse station
    data=[
        [1, 0.25, 0],
        [0, 0.64, 0.16],
        [0.25, 1, 0],
        [nan, nan, nan],
        [nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005], "x": [0, 0.5, 1]},
    name="total_penalty",
)

EXP_CRPS_BD2 = xr.merge([EXP_TOTAL_CRPS_BD2, EXP_UNDER_CRPS_BD2, EXP_OVER_CRPS_BD2])

# test data for CRPS for ensembles

DA_FCST_CRPSENS = xr.DataArray(
    data=[[0.0, 4, 3, 7], [1, -1, 2, 4], [0, 1, 4, np.nan], [2, 3, 4, 1], [2, np.nan, np.nan, np.nan]],
    dims=["stn", "ens_member"],
    coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4]},
)
DA_FCST_CRPSENS_LT = xr.DataArray(
    data=[
        [[0.0, 4, 3, 7], [1, -1, 2, 4], [0, 1, 4, np.nan], [2, 3, 4, 1], [2, np.nan, np.nan, np.nan]],
        [
            [2, 2, 2, 2],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
    ],
    dims=["lead_time", "stn", "ens_member"],
    coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4], "lead_time": [1, 2]},
)
DA_OBS_CRPSENS = xr.DataArray(
    data=[2.0, 3, 1, np.nan, 4, 5], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105, 106]}
)
DA_WT_CRPSENS = xr.DataArray(data=[1, 2, 1, 0, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
DA_T_TWCRPSENS = xr.DataArray(data=[np.nan, 1, 10, 1, -2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
DA_LI_TWCRPSENS = xr.DataArray(data=[np.nan, 2, 2, 100, -200], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
DA_UI_TWCRPSENS = xr.DataArray(data=[np.nan, 5, 5, 200, -100], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
DA_LI_CONS_TWCRPSENS = xr.DataArray(data=[2, 2, 2, 2, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
DA_UI_CONS_TWCRPSENS = xr.DataArray(data=[5, 5, 5, 5, 5], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})

DS_FCST_CRPSENS = xr.Dataset({"a": DA_FCST_CRPSENS, "b": DA_FCST_CRPSENS})
DS_OBS_CRPSENS = xr.Dataset({"a": DA_OBS_CRPSENS, "b": DA_OBS_CRPSENS})
DS_WT_CRPSENS = xr.Dataset({"a": DA_WT_CRPSENS, "b": DA_WT_CRPSENS})
DS_T_TWCRPSENS = xr.Dataset({"a": DA_T_TWCRPSENS, "b": DA_T_TWCRPSENS})
# first and second (spread) terms from crps for ensembles, "ecdf" and "fair" methods
FIRST_TERM = xr.DataArray(
    data=[10 / 4, 8 / 4, 4 / 3, np.nan, 2],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
SPREAD_ECDF = xr.DataArray(
    data=[(14 + 8 + 8 + 14) / 32, (6 + 10 + 6 + 10) / 32, (5 + 4 + 7) / 18, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
SPREAD_FAIR = xr.DataArray(
    data=[(14 + 8 + 8 + 14) / 24, (6 + 10 + 6 + 10) / 24, (5 + 4 + 7) / 12, np.nan, np.nan],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
UNDER_TERM = xr.DataArray(
    data=[2 / 4, 7 / 4, 1 / 3, np.nan, 2],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
OVER_TERM = xr.DataArray(
    data=[8 / 4, 1 / 4, 3 / 3, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)

# expected results
EXP_CRPSENS_ECDF = FIRST_TERM - SPREAD_ECDF
EXP_CRPSENS_ECDF_DECOMPOSITION = xr.concat([EXP_CRPSENS_ECDF, UNDER_TERM, OVER_TERM, SPREAD_ECDF], dim="component")
EXP_CRPSENS_ECDF_DECOMPOSITION = EXP_CRPSENS_ECDF_DECOMPOSITION.assign_coords(
    component=["total", "underforecast_penalty", "overforecast_penalty", "spread"]
)

EXP_CRPSENS_FAIR = FIRST_TERM - SPREAD_FAIR
EXP_CRPSENS_WT = (EXP_CRPSENS_ECDF * DA_WT_CRPSENS).mean("stn")

EXP_CRPSENS_ECDF_DS = xr.Dataset({"a": EXP_CRPSENS_ECDF, "b": EXP_CRPSENS_ECDF})
EXP_CRPSENS_ECDF_DECOMPOSITION_DS = xr.Dataset(
    {"a": EXP_CRPSENS_ECDF_DECOMPOSITION, "b": EXP_CRPSENS_ECDF_DECOMPOSITION}
)
EXP_CRPSENS_FAIR_DS = xr.Dataset({"a": EXP_CRPSENS_FAIR, "b": EXP_CRPSENS_FAIR})
EXP_CRPSENS_WT_DS = xr.Dataset({"a": EXP_CRPSENS_WT, "b": EXP_CRPSENS_WT})

# Test data for broadcasting
FIRST_TERM_BC = xr.DataArray(
    data=[[10 / 4, 8 / 4, 4 / 3, np.nan, 2], [0, np.nan, np.nan, np.nan, np.nan]],
    dims=["lead_time", "stn"],
    coords={"lead_time": [1, 2], "stn": [101, 102, 103, 104, 105]},
)
SPREAD_ECDF_BC = xr.DataArray(
    data=[
        [(14 + 8 + 8 + 14) / 32, (6 + 10 + 6 + 10) / 32, (5 + 4 + 7) / 18, np.nan, 0],
        [0, np.nan, np.nan, np.nan, np.nan],
    ],
    dims=["lead_time", "stn"],
    coords={"lead_time": [1, 2], "stn": [101, 102, 103, 104, 105]},
)
EXP_CRPSENS_ECDF_BC = FIRST_TERM_BC - SPREAD_ECDF_BC

# exp test data for twCRPS with upper tail for thresholds >=1
UPPER_TAIL_FIRST_TERM_DA = xr.DataArray(
    data=[9 / 4, 6 / 4, 3 / 3, np.nan, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
UPPER_TAIL_SPREAD_ECDF_DA = xr.DataArray(
    data=[(11 + 7 + 7 + 13) / 32, (4 + 4 + 4 + 8) / 32, (3 + 3 + 6) / 18, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
UPPER_TAIL_SPREAD_FAIR_DA = xr.DataArray(
    data=[(11 + 7 + 7 + 13) / 24, (4 + 4 + 4 + 8) / 24, (3 + 3 + 6) / 12, np.nan, np.nan],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
UPPER_TAIL_UNDER_DA = xr.DataArray(
    data=[1 / 4, 5 / 4, 0, np.nan, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
OVER_TAIL_UNDER_DA = xr.DataArray(
    data=[8 / 4, 1 / 4, 1, np.nan, 0], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
EXP_UPPER_TAIL_CRPSENS_ECDF_DA = UPPER_TAIL_FIRST_TERM_DA - UPPER_TAIL_SPREAD_ECDF_DA
EXP_UPPER_TAIL_CRPSENS_FAIR_DA = UPPER_TAIL_FIRST_TERM_DA - UPPER_TAIL_SPREAD_FAIR_DA
EXP_UPPER_TAIL_CRPSENS_ECDF_DS = xr.Dataset({"a": EXP_UPPER_TAIL_CRPSENS_ECDF_DA, "b": EXP_UPPER_TAIL_CRPSENS_ECDF_DA})
EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA = xr.concat(
    [EXP_UPPER_TAIL_CRPSENS_ECDF_DA, UPPER_TAIL_UNDER_DA, OVER_TAIL_UNDER_DA, UPPER_TAIL_SPREAD_ECDF_DA],
    dim="component",
)
EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA = EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA.assign_coords(
    component=["total", "underforecast_penalty", "overforecast_penalty", "spread"]
)
EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DS = xr.Dataset(
    {"a": EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA, "b": EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA}
)
# exp test data for twCRPS with upper tail for thresholds >=1 with broadcasting
UPPER_TAIL_FIRST_TERM_BC = xr.DataArray(
    data=[[9 / 4, 6 / 4, 3 / 3, np.nan, 2], [0, np.nan, np.nan, np.nan, np.nan]],
    dims=["lead_time", "stn"],
    coords={"lead_time": [1, 2], "stn": [101, 102, 103, 104, 105]},
)
UPPER_TAIL_SPREAD_ECDF_BC = xr.DataArray(
    data=[
        [(11 + 7 + 7 + 13) / 32, (4 + 4 + 4 + 8) / 32, (3 + 3 + 6) / 18, np.nan, 0],
        [0, np.nan, np.nan, np.nan, np.nan],
    ],
    dims=["lead_time", "stn"],
    coords={"lead_time": [1, 2], "stn": [101, 102, 103, 104, 105]},
)
EXP_UPPER_TAIL_CRPSENS_ECDF_BC = UPPER_TAIL_FIRST_TERM_BC - UPPER_TAIL_SPREAD_ECDF_BC

# exp test data for twCRPS with lower tail for thresholds <=1
LOWER_TAIL_FIRST_TERM_DA = xr.DataArray(
    data=[1 / 4, 2 / 4, 1 / 3, np.nan, 0], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
LOWER_TAIL_SPREAD_ECDF_DA = xr.DataArray(
    data=[(3 + 1 + 1 + 1) / 32, (2 + 6 + 2 + 2) / 32, (2 + 1 + 1) / 18, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
LOWER_TAIL_SPREAD_FAIR_DA = xr.DataArray(
    data=[(3 + 1 + 1 + 1) / 24, (2 + 6 + 2 + 2) / 24, (2 + 1 + 1) / 12, np.nan, np.nan],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
EXP_LOWER_TAIL_CRPSENS_ECDF_DA = LOWER_TAIL_FIRST_TERM_DA - LOWER_TAIL_SPREAD_ECDF_DA
EXP_LOWER_TAIL_CRPSENS_FAIR_DA = LOWER_TAIL_FIRST_TERM_DA - LOWER_TAIL_SPREAD_FAIR_DA

# exp test data for twCRP with the upper tail varying across a dimension
VAR_THRES_FIRST_TERM_DA = xr.DataArray(
    data=[np.nan, 6 / 4, 0, np.nan, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
VAR_THRES_SPREAD_ECDF_DA = xr.DataArray(
    data=[np.nan, (4 + 4 + 4 + 8) / 32, 0, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
EXP_VAR_THRES_CRPSENS_DA = VAR_THRES_FIRST_TERM_DA - VAR_THRES_SPREAD_ECDF_DA
EXP_VAR_THRES_CRPSENS_DS = xr.Dataset({"a": EXP_VAR_THRES_CRPSENS_DA, "b": EXP_VAR_THRES_CRPSENS_DA})

# exp test data for interval twCRPS
INTERVAL_FIRST_TERM_DA = xr.DataArray(
    data=[6 / 4, 4 / 4, 2 / 3, np.nan, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
INTERVAL_SPREAD_ECDF_DA = xr.DataArray(
    data=[(6 + 4 + 4 + 6) / 32, (2 + 2 + 2 + 6) / 32, (2 + 2 + 4) / 18, np.nan, 0],
    dims=["stn"],
    coords={"stn": [101, 102, 103, 104, 105]},
)
EXP_INTERVAL_CRPSENS_ECDF_DA = INTERVAL_FIRST_TERM_DA - INTERVAL_SPREAD_ECDF_DA

INTERVAL_UNDER_DA = xr.DataArray(data=[0, 3 / 4, 0, np.nan, 2], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]})
INTERVAL_OVER_DA = xr.DataArray(
    data=[6 / 4, 1 / 4, 2 / 3, np.nan, 0], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA = xr.concat(
    [EXP_INTERVAL_CRPSENS_ECDF_DA, INTERVAL_UNDER_DA, INTERVAL_OVER_DA, INTERVAL_SPREAD_ECDF_DA], dim="component"
)
EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA = EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA.assign_coords(
    component=["total", "underforecast_penalty", "overforecast_penalty", "spread"]
)
EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DS = xr.Dataset(
    {"a": EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA, "b": EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA}
)
EXP_VAR_INTERVAL_CRPSENS_ECDF_DA = EXP_INTERVAL_CRPSENS_ECDF_DA * xr.DataArray(
    data=[np.nan, 1, 1, np.nan, 0], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
