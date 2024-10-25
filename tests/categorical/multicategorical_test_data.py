"""
Test data for testing scores.categorical.multicategorical functions
"""

import numpy as np
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

# test data for _risk_matrix_score
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
