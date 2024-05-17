"""
Data for probability tests. 
"""

import numpy as np
import xarray as xr

DA_ROUND = xr.DataArray(
    data=[[0.2, 1.323, 10.412], [-3.914, 0.001, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

EXP_ROUND1 = DA_ROUND.copy()

EXP_ROUND2 = xr.DataArray(  # round to nearest .2
    data=[[0.2, 1.4, 10.4], [-4.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

EXP_ROUND3 = xr.DataArray(  # round to nearest 5
    data=[[0.0, 0.0, 10.0], [-5.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

DA_ADD_THRESHOLDS = xr.DataArray(
    data=[[[0.2, 0.4, 1, 1], [0, 0, 0.6, 1]]],
    dims=["date", "station", "x"],
    coords=dict(station=[1001, 1002], date=["2020-01-01"], x=[0, 0.2, 0.5, 1]),  # pylint: disable=use-dict-literal
)

EXP_ADD_THRESHOLDS1 = xr.DataArray(
    data=[[[0.2, 0.4, 1, 1, 1], [0, 0, 0.6, 0.8, 1]]],
    dims=["date", "station", "x"],
    coords=dict(
        station=[1001, 1002], date=["2020-01-01"], x=[0, 0.2, 0.5, 0.75, 1]
    ),  # pylint: disable=use-dict-literal
)

EXP_ADD_THRESHOLDS2 = xr.DataArray(
    data=[[[0.2, 0.4, 1, np.nan, 1], [0, 0, 0.6, np.nan, 1]]],
    dims=["date", "station", "x"],
    coords=dict(
        station=[1001, 1002], date=["2020-01-01"], x=[0, 0.2, 0.5, 0.75, 1]
    ),  # pylint: disable=use-dict-literal
)

DA_DECREASING_CDFS1 = xr.DataArray(
    data=[
        [0.2, 0.4, 0.5, 0.8, 1],
        [0.2, 0.1, 0.7, 0.4, 0.9],
        [0, 0.6, 0.59, 0.7, 0.69],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ],
    dims=["station", "x"],
    coords=dict(station=[1001, 1002, 1003, 1004], x=[0, 1, 2, 3, 4]),  # pylint: disable=use-dict-literal
)

EXP_DECREASING_CDFS1A = xr.DataArray(
    data=[False, True, True, False],
    dims=["station"],
    coords=dict(station=[1001, 1002, 1003, 1004]),  # pylint: disable=use-dict-literal
)

EXP_DECREASING_CDFS1B = xr.DataArray(
    data=[False, True, False, False],
    dims=["station"],
    coords=dict(station=[1001, 1002, 1003, 1004]),  # pylint: disable=use-dict-literal
)
