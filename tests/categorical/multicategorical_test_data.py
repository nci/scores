"""
Test data for testing scores.categorical.multicategorical functions
"""

import numpy as np
import pandas as pd
import xarray as xr

DA_FCST_SC = xr.DataArray(
    data=[[[np.nan, 7, 4], [-100, 0, 1]], [[0, 1, 5], [10, 16, 1]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11, 12],
    },
)
DA_FCST_SC2 = xr.DataArray(
    data=[3, 3, 1, 2, 2, 1],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)

DA_OBS_SC = xr.DataArray(
    data=[[10, np.nan], [0, 1]],
    dims=["j", "k"],
    coords={
        "j": [100001, 10000],  # coords in different order to forecast
        "k": [10, 11],
    },
)

DA_OBS_SC2 = xr.DataArray(
    data=[1, 2, 3, 4, 1, 2],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)
DA_THRESHOLD_SC = xr.DataArray(
    data=[[5, 5], [-200, np.nan]],
    dims=["j", "k"],
    coords={
        "j": [100001, 10000],  # coords in different order to forecast
        "k": [10, 11],
    },
)

EXP_SC_TOTAL_CASE0 = xr.DataArray(
    data=[[[np.nan, 0.3], [0.7, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_UNDER_CASE0 = xr.DataArray(
    data=[[[np.nan, 0], [0.7, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_OVER_CASE0 = xr.DataArray(
    data=[[[np.nan, 0.3], [0, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_CASE0 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE0,
        "underforecast_penalty": EXP_SC_UNDER_CASE0,
        "overforecast_penalty": EXP_SC_OVER_CASE0,
    }
)

EXP_SC_TOTAL_CASE1 = xr.DataArray(
    data=[[[np.nan, 0], [0, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)

EXP_SC_CASE1 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE1,
        "underforecast_penalty": EXP_SC_TOTAL_CASE1,
        "overforecast_penalty": EXP_SC_TOTAL_CASE1,
    }
)

EXP_SC_TOTAL_CASE2 = xr.DataArray(
    data=[[[np.nan, 1.2], [3.5, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_UNDER_CASE2 = xr.DataArray(
    data=[[[np.nan, 0], [3.5, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_OVER_CASE2 = xr.DataArray(
    data=[[[np.nan, 1.2], [0, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_CASE2 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE2,
        "underforecast_penalty": EXP_SC_UNDER_CASE2,
        "overforecast_penalty": EXP_SC_OVER_CASE2,
    }
)

EXP_SC_TOTAL_CASE3 = xr.DataArray(
    data=[[[np.nan, 0.15], [0.35, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_UNDER_CASE3 = xr.DataArray(
    data=[[[np.nan, 0], [0.35, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_OVER_CASE3 = xr.DataArray(
    data=[[[np.nan, 0.15], [0, np.nan]], [[0.0, 0], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_CASE3 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE3,
        "underforecast_penalty": EXP_SC_UNDER_CASE3,
        "overforecast_penalty": EXP_SC_OVER_CASE3,
    }
)


EXP_SC_TOTAL_CASE4 = xr.DataArray(
    data=[0.3, 0.3, 0.7, 0.7, 0, 0],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)
EXP_SC_UNDER_CASE4 = xr.DataArray(
    data=[0, 0, 0.7, 0.7, 0, 0],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)
EXP_SC_OVER_CASE4 = xr.DataArray(
    data=[0.3, 0.3, 0, 0, 0, 0],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)

EXP_SC_CASE4 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE4,
        "underforecast_penalty": EXP_SC_UNDER_CASE4,
        "overforecast_penalty": EXP_SC_OVER_CASE4,
    }
)

EXP_SC_TOTAL_CASE5 = xr.DataArray(
    data=[0.3, 0, 0.7, 0.0, 0.3, 0.7],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)
EXP_SC_UNDER_CASE5 = xr.DataArray(
    data=[0, 0, 0.7, 0, 0, 0.7],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)
EXP_SC_OVER_CASE5 = xr.DataArray(
    data=[0.3, 0.0, 0, 0, 0.3, 0],
    dims=["i"],
    coords={
        "i": [1, 2, 3, 4, 5, 6],
    },
)

EXP_SC_CASE5 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE5,
        "underforecast_penalty": EXP_SC_UNDER_CASE5,
        "overforecast_penalty": EXP_SC_OVER_CASE5,
    }
)

EXP_SC_TOTAL_CASE6 = xr.DataArray(
    data=[[[np.nan, np.nan], [0.7, np.nan]], [[0.0, np.nan], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_UNDER_CASE6 = xr.DataArray(
    data=[[[np.nan, np.nan], [0.7, np.nan]], [[0.0, np.nan], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_OVER_CASE6 = xr.DataArray(
    data=[[[np.nan, np.nan], [0, np.nan]], [[0.0, np.nan], [0, np.nan]]],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001],
        "k": [10, 11],
    },
)
EXP_SC_CASE6 = xr.Dataset(
    {
        "firm_score": EXP_SC_TOTAL_CASE6,
        "underforecast_penalty": EXP_SC_UNDER_CASE6,
        "overforecast_penalty": EXP_SC_OVER_CASE6,
    }
)

DA_FCST_FIRM = xr.DataArray(
    data=[
        [[np.nan, 7, 4], [-100, 0, 1], [0, -100, 1]],
        [[0, 1, 5], [10, 16, 1], [-10, -16, 1]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11, 12],
    },
)
DA_OBS_FIRM = xr.DataArray(
    data=[[0, 1], [10, np.nan], [0, 10]],
    dims=["j", "k"],
    coords={
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)

DA_THRESHOLD_FIRM = [
    xr.DataArray(
        data=[0, 0, 0],
        dims=["j"],
        coords={"j": [10000, 100001, 900000]},
    ),
    xr.DataArray(
        data=[5, 5, 5],
        dims=["j"],
        coords={"j": [10000, 100001, 900000]},
    ),
]

DA_THRESHOLD_FIRM2 = [
    xr.DataArray(
        data=[0, 5, 0],
        dims=["j"],
        coords={"j": [10000, 100001, 900000]},
    ),
    xr.DataArray(
        data=[5, 0, 5],
        dims=["j"],
        coords={"j": [10000, 100001, 900000]},
    ),
]

LIST_WEIGHTS_FIRM0 = [
    xr.DataArray(
        data=[2, 2, 2],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
    xr.DataArray(
        data=[1, 1, 1],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
]
LIST_WEIGHTS_FIRM1 = [
    xr.DataArray(
        data=[200, 20, 0.2],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
    xr.DataArray(
        data=[1000, 10, 0.1],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
]
LIST_WEIGHTS_FIRM2 = [
    xr.DataArray(
        data=[200, 20, 0.2],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
    xr.DataArray(
        data=[1000, 10, np.nan],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
]

LIST_WEIGHTS_FIRM3 = [
    xr.DataArray(
        data=[20, np.nan, -200],  # negative weight
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
    xr.DataArray(
        data=[10, 1, 100],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
]
LIST_WEIGHTS_FIRM4 = [
    xr.DataArray(
        data=[20, np.nan, 0.0],  # zero weight
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
    xr.DataArray(
        data=[10, 1, 100],
        dims=["j"],
        coords={
            "j": [10000, 100001, 900000],
        },
    ),
]
LIST_WEIGHTS_FIRM5 = [
    xr.DataArray(
        data=[[0, 1], [10, np.nan], [0, 10]],
        dims=["j", "z"],
        coords={
            "j": [10000, 100001, 900000],
            "z": [10, 11],
        },
    ),
    xr.DataArray(
        data=[[0, 1], [10, np.nan], [0, 10]],
        dims=["j", "k"],
        coords={
            "j": [10000, 100001, 900000],
            "k": [10, 11],
        },
    ),
]
EXP_FIRM_TOTAL_CASE0 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [0.7, np.nan], [0.0, 0.7]],
        [[0.0, 0], [0, np.nan], [0.0, 0.7]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_UNDER_CASE0 = xr.DataArray(
    data=[
        [[np.nan, 0.0], [0.7, np.nan], [0.0, 0.7]],
        [[0.0, 0], [0, np.nan], [0.0, 0.7]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_OVER_CASE0 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [0.0, np.nan], [0.0, 0.0]],
        [[0.0, 0], [0, np.nan], [0.0, 0.0]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_CASE0 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE0,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE0,
        "overforecast_penalty": EXP_FIRM_OVER_CASE0,
    }
)

EXP_FIRM_TOTAL_CASE1 = xr.DataArray(
    data=[0.425, 0.14],
    dims=["i"],
    coords={"i": [1, 2]},
)
EXP_FIRM_UNDER_CASE1 = xr.DataArray(
    data=[0.35, 0.14],
    dims=["i"],
    coords={"i": [1, 2]},
)
EXP_FIRM_OVER_CASE1 = xr.DataArray(
    data=[0.075, 0.0],
    dims=["i"],
    coords={"i": [1, 2]},
)
EXP_FIRM_CASE1 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE1,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE1,
        "overforecast_penalty": EXP_FIRM_OVER_CASE1,
    }
)

EXP_FIRM_TOTAL_CASE2 = xr.DataArray(data=0.2666666666666)
EXP_FIRM_OVER_CASE2 = xr.DataArray(data=0.3 / 9)
EXP_FIRM_UNDER_CASE2 = xr.DataArray(data=2.1 / 9)
EXP_FIRM_CASE2 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE2,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE2,
        "overforecast_penalty": EXP_FIRM_OVER_CASE2,
    }
)

EXP_FIRM_TOTAL_CASE3 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [1.4, np.nan], [0.0, 1.4]],
        [[0.0, 0], [0, np.nan], [0.0, 1.4]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_UNDER_CASE3 = xr.DataArray(
    data=[
        [[np.nan, 0.0], [1.4, np.nan], [0.0, 1.4]],
        [[0.0, 0], [0, np.nan], [0.0, 1.4]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_OVER_CASE3 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [0.0, np.nan], [0.0, 0.0]],
        [[0.0, 0], [0, np.nan], [0.0, 0.0]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_CASE3 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE3,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE3,
        "overforecast_penalty": EXP_FIRM_OVER_CASE3,
    }
)

EXP_FIRM_TOTAL_CASE4 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [2.1, np.nan], [0.0, 2.1]],
        [[0.0, 0], [0, np.nan], [0.0, 2.1]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_UNDER_CASE4 = xr.DataArray(
    data=[
        [[np.nan, 0.0], [2.1, np.nan], [0.0, 2.1]],
        [[0.0, 0], [0, np.nan], [0.0, 2.1]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_OVER_CASE4 = xr.DataArray(
    data=[
        [[np.nan, 0.3], [0.0, np.nan], [0.0, 0.0]],
        [[0.0, 0], [0, np.nan], [0.0, 0.0]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_CASE4 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE4,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE4,
        "overforecast_penalty": EXP_FIRM_OVER_CASE4,
    }
)

EXP_FIRM_TOTAL_CASE5 = xr.DataArray(
    data=[
        [[np.nan, 300], [21, np.nan], [0.0, 0.21]],
        [[0.0, 0], [0, np.nan], [0.0, 0.21]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_UNDER_CASE5 = xr.DataArray(
    data=[
        [[np.nan, 0.0], [21, np.nan], [0.0, 0.21]],
        [[0.0, 0], [0, np.nan], [0.0, 0.21]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_OVER_CASE5 = xr.DataArray(
    data=[
        [[np.nan, 300], [0.0, np.nan], [0.0, 0.0]],
        [[0.0, 0], [0, np.nan], [0.0, 0.0]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_CASE5 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE5,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE5,
        "overforecast_penalty": EXP_FIRM_OVER_CASE5,
    }
)

EXP_FIRM_TOTAL_CASE6 = xr.DataArray(
    data=[
        [[np.nan, 300], [21, np.nan], [np.nan, np.nan]],
        [[0.0, 0], [0, np.nan], [np.nan, np.nan]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_UNDER_CASE6 = xr.DataArray(
    data=[
        [[np.nan, 0.0], [21, np.nan], [np.nan, np.nan]],
        [[0.0, 0], [0, np.nan], [np.nan, np.nan]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_OVER_CASE6 = xr.DataArray(
    data=[
        [[np.nan, 300], [0.0, np.nan], [np.nan, np.nan]],
        [[0.0, 0], [0, np.nan], [np.nan, np.nan]],
    ],
    dims=["i", "j", "k"],
    coords={
        "i": [1, 2],
        "j": [10000, 100001, 900000],
        "k": [10, 11],
    },
)
EXP_FIRM_CASE6 = xr.Dataset(
    {
        "firm_score": EXP_FIRM_TOTAL_CASE6,
        "underforecast_penalty": EXP_FIRM_UNDER_CASE6,
        "overforecast_penalty": EXP_FIRM_OVER_CASE6,
    }
)
# Data for SEEPS testing
DA_OBS_SEEPS = xr.DataArray(
    data=[[0, 0.21, 15, -0.1, 5], [20, 0.2, 10, 200, 3], [20, 0.2, 10, 200, 3]],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=3),
    },
)
DS_OBS_SEEPS = DA_OBS_SEEPS.to_dataset(name="i")

DA_FCST_SEEPS = xr.DataArray(
    data=[[0, 0.1, 0.2, 0.21, 5], [10, 15, 20, 200, np.nan]],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
DA_FCST2_SEEPS = xr.DataArray(
    data=[
        [[0, np.nan, np.nan, np.nan, 5], [np.nan, np.nan, np.nan, 200, np.nan]],
        [[0, 0.1, 0.2, 0.21, 5], [10, 15, 20, 200, np.nan]],
    ],
    dims=["i", "t", "j"],
    coords={
        "i": [1, 2],
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)

DS_FCST_SEEPS = DA_FCST2_SEEPS.to_dataset(dim="i")  # .to_array(dim="i")

DA_SEEPS_WEIGHTS = xr.DataArray(
    data=[1, 2],
    dims=["i"],
    coords={"i": [1, 2]},
)

DA_P1_SEEPS = xr.DataArray(0.5)
DA_P3_SEEPS = (1 - DA_P1_SEEPS) / 3
DA_P1_VARY1_SEEPS = xr.DataArray(
    data=[0.5, 0.09, 0.1, 0.5, 0.5],
    dims=["j"],
    coords={"j": [1, 2, 3, 4, 5]},
)
DA_P3_VARY1_SEEPS = (1 - DA_P1_VARY1_SEEPS) / 3

DA_P1_VARY2_SEEPS = xr.DataArray(
    data=[0.5, 0.85, 0.86, 0.5, 0.5],
    dims=["j"],
    coords={"j": [1, 2, 3, 4, 5]},
)
DA_P3_VARY2_SEEPS = (1 - DA_P1_VARY2_SEEPS) / 3


DA_LIGHT_HEAVY_THRESHOLD_SEEPS = xr.DataArray(10)
DA_LIGHT_HEAVY_THRESHOLD_VARY_SEEPS = xr.DataArray(
    data=[10, 20], dims=["t"], coords={"t": pd.date_range("2020-01-01", periods=2)}
)

P1 = 0.5
P3 = (1 - P1) / 3

ALL_INDEX_ARRAY_RESULT = [
    [0, 1 / (1 - P1), 1 / P3 + 1 / (1 - P1), 1 / P1, 0],
    [1 / P3, 1 / P1 + 1 / (1 - P3), 1 / (1 - P3), 0, np.nan],
]

EXP_SEEPS_CASE0 = 0.5 * xr.DataArray(
    data=ALL_INDEX_ARRAY_RESULT,
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
EXP_SEEPS_CASE1 = EXP_SEEPS_CASE0.mean()
EXP_SEEPS_CASE2 = EXP_SEEPS_CASE0.mean(dim="j")
EXP_SEEPS_CASE3 = EXP_SEEPS_CASE0.mean(dim="t")
EXP_SEEPS_CASE4 = 0.5 * xr.DataArray(
    data=[
        [
            [0, np.nan, np.nan, np.nan, 0],
            [np.nan, np.nan, np.nan, 0, np.nan],
        ],
        ALL_INDEX_ARRAY_RESULT,
    ],
    dims=["i", "t", "j"],
    coords={
        "i": [1, 2],
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
EXP_SEEPS_CASE5 = 0.5 * xr.DataArray(
    data=[
        np.array([[0, np.nan, np.nan, np.nan, 0], [np.nan, np.nan, np.nan, 0, np.nan]]),
        2 * np.array(ALL_INDEX_ARRAY_RESULT),
    ],
    dims=["i", "t", "j"],
    coords={
        "i": [1, 2],
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)

EXP_SEEPS_CASE6 = EXP_SEEPS_CASE0 * 0

# P1 varies by j with 0.5, 0.09, 0.1, 0.5, 0.5
EXP_SEEPS_CASE7 = 0.5 * xr.DataArray(
    data=[
        [0, 1 / (1 - 0.09), 1 / ((1 - 0.1) / 3) + 1 / (1 - 0.1), 1 / 0.5, 0],
        [1 / (1 / 6), 1 / 0.09 + 1 / (1 - ((1 - 0.09) / 3)), 1 / (1 - (0.9 / 3)), 0, np.nan],
    ],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
# P1 varies by j with 0.5, 0.85, 0.86, 0.5, 0.5
EXP_SEEPS_CASE8 = 0.5 * xr.DataArray(
    data=[
        [0, 1 / (1 - 0.85), 1 / ((1 - 0.86) / 3) + 1 / (1 - 0.86), 1 / 0.5, 0],
        [1 / (1 / 6), 1 / 0.85 + 1 / (1 - 0.15 / 3), 1 / (1 - ((1 - 0.86) / 3)), 0, np.nan],
    ],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
# j = 2 should be masked
EXP_SEEPS_CASE9 = 0.5 * xr.DataArray(
    data=[
        [0, np.nan, 1 / ((1 - 0.1) / 3) + 1 / (1 - 0.1), 1 / 0.5, 0],
        [1 / (1 / 6), np.nan, 1 / (1 - (0.9 / 3)), 0, np.nan],
    ],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)
# j =3 should be masked
EXP_SEEPS_CASE10 = 0.5 * xr.DataArray(
    data=[
        [0, 1 / (1 - 0.85), np.nan, 1 / 0.5, 0],
        [1 / (1 / 6), 1 / 0.85 + 1 / (1 - 0.15 / 3), np.nan, 0, np.nan],
    ],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)

EXP_SEEPS_CASE11 = 0.5 * xr.DataArray(
    data=[
        [0, 1 / (1 - P1), 1 / P3 + 1 / (1 - P1), 1 / P1, 0],
        [0, 1 / P1, 0, 0, np.nan],
    ],
    dims=["t", "j"],
    coords={
        "j": [1, 2, 3, 4, 5],
        "t": pd.date_range("2020-01-01", periods=2),
    },
)

EXP_SEEPS_CASE12 = EXP_SEEPS_CASE5.to_dataset(dim="i")
