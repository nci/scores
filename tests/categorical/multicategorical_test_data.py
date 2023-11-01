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

DA_OBS_SC = xr.DataArray(
    data=[[10, np.nan], [0, 1]],
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
