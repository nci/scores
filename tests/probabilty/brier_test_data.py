import numpy as np
import xarray as xr

FCST1 = xr.DataArray(
    [[[0.5, 0], [0, 0.5], [1, 0]], [[0.5, 0], [0.5, 0], [0, np.nan]]],
    dims=["a", "b", "c"],
    coords={"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]},
)
OBS1 = xr.DataArray(
    [[[1, 0], [0, 0], [np.nan, 0]], [[0, 0], [1, 0], [0, 1]]],
    dims=["a", "b", "c"],
    coords={"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]},
)

DA_FCST_ENS = xr.DataArray(
    data=[[0.0, 4, 3, 7], [0, -1, 2, 4], [0, 1, 4, np.nan], [2, 3, 4, 1], [0, np.nan, np.nan, np.nan]],
    dims=["stn", "ens_member"],
    coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4]},
)
DA_FCST_ENS_LT = xr.DataArray(
    data=[
        [[0.0, 4, 3, 7], [0, -1, 2, 4], [0, 1, 4, np.nan], [2, 3, 4, 1], [0, np.nan, np.nan, np.nan]],
        [[0.0, 0, 0, 0], [0, -1, 2, 4], [np.nan, np.nan, 4, np.nan], [2, 3, 4, 1], [0, np.nan, np.nan, np.nan]],
    ],
    dims=[
        "lead_time",
        "stn",
        "ens_member",
    ],
    coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4], "lead_time": [1, 2]},
)
DA_OBS_ENS = xr.DataArray(data=[0, 3, 1, np.nan, 4, 5], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105, 106]})


EXP_BRIER_ENS_ALL = xr.DataArray(
    data=[[(3 / 4) ** 2, (2 / 4 - 1) ** 2, (2 / 3 - 1) ** 2, np.nan, 1]],
    dims=["threshold", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
).T

EXP_BRIER_ENS_ALL_GREATER = xr.DataArray(
    data=[[(3 / 4) ** 2, (2 / 4 - 1) ** 2, (1 / 3 - 1) ** 2, np.nan, 1]],
    dims=["threshold", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
).T

EXP_BRIER_ENS_ALL_LT = xr.DataArray(
    data=[
        [[(3 / 4) ** 2, (2 / 4 - 1) ** 2, (2 / 3 - 1) ** 2, np.nan, 1]],
        [[0, (2 / 4 - 1) ** 2, 0, np.nan, 1]],
    ],
    dims=["lead_time", "threshold", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [1], "lead_time": [1, 2]},
)
I1 = 3
M1 = 4
I2 = 2
M2 = 4
I3 = 2
M3 = 3
FAIR_CORR_ALL = xr.DataArray(
    data=[
        [
            I1 * (M1 - I1) / (M1**2 * (M1 - 1)),
            I2 * (M2 - I2) / (M2**2 * (M2 - 1)),
            I3 * (M3 - I3) / (M3**2 * (M3 - 1)),
            np.nan,
            0,
        ]
    ],
    dims=["threshold", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
)

EXP_BRIER_ENS_FAIR_ALL = EXP_BRIER_ENS_ALL - FAIR_CORR_ALL
EXP_BRIER_ENS_FAIR_ALL_MEAN = EXP_BRIER_ENS_FAIR_ALL.mean("stn")

EXP_BRIER_ENS_ALL_MULTI = xr.DataArray(
    data=[
        [0, 0, 0, np.nan, 0],
        [(3 / 4) ** 2, (2 / 4 - 1) ** 2, (2 / 3 - 1) ** 2, np.nan, 1],
        [0, 0, 0, np.nan, 0],
    ],
    dims=["threshold", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [-100, 1, 100]},
).T

I12 = 0
FAIR_CORR_ALL_LT = xr.DataArray(
    data=[
        [
            [
                I1 * (M1 - I1) / (M1**2 * (M1 - 1)),
                I2 * (M2 - I2) / (M2**2 * (M2 - 1)),
                I3 * (M3 - I3) / (M3**2 * (M3 - 1)),
                np.nan,
                0,
            ],
            [
                I12 * (M1 - I12) / (M1**2 * (M1 - 1)),
                I2 * (M2 - I2) / (M2**2 * (M2 - 1)),
                0,
                np.nan,
                0,
            ],
        ]
    ],
    dims=["threshold", "lead_time", "stn"],
    coords={"stn": [101, 102, 103, 104, 105], "threshold": [1], "lead_time": [1, 2]},
)
EXP_BRIER_ENS_FAIR_ALL_LT = EXP_BRIER_ENS_ALL_LT - FAIR_CORR_ALL_LT
EXP_BRIER_ENS_FAIR_ALL_LT = EXP_BRIER_ENS_FAIR_ALL_LT.transpose("lead_time", "stn", "threshold")
ENS_BRIER_WEIGHTS = xr.DataArray(
    [2, 1, np.nan, np.nan, np.nan], dims=["stn"], coords={"stn": [101, 102, 103, 104, 105]}
)
EXP_BRIER_ENS_WITH_WEIGHTS = xr.DataArray(
    data=[(2 * (3 / 4) ** 2 + (2 / 4 - 1) ** 2) / 2],
    dims=["threshold"],
    coords={"threshold": [1]},
)

FCST_ENS_DS = xr.Dataset(
    {
        "a": xr.DataArray(
            [[0.0, 4, 3, 7], [0, -1, 2, 4], [0, 1, 4, np.nan], [2, 3, 4, 1], [0, np.nan, np.nan, np.nan]],
            dims=["stn", "ens_member"],
            coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4]},
        ),
        "b": xr.DataArray(
            [[0.0, 0, 0, 0], [0, -1, 2, 4], [np.nan, np.nan, 4, np.nan], [2, 3, 4, 1], [0, np.nan, np.nan, np.nan]],
            dims=["stn", "ens_member"],
            coords={"stn": [101, 102, 103, 104, 105], "ens_member": [1, 2, 3, 4]},
        ),
    },
)
OBS_ENS_DS = xr.Dataset({"a": DA_OBS_ENS, "b": DA_OBS_ENS})


EXP_BRIER_ENS_ALL_DS = xr.Dataset(
    {
        "a": xr.DataArray(
            [[(3 / 4) ** 2, (2 / 4 - 1) ** 2, (2 / 3 - 1) ** 2, np.nan, 1]],
            dims=["threshold", "stn"],
            coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
        ).T,
        "b": xr.DataArray(
            [[0, (2 / 4 - 1) ** 2, 0, np.nan, 1]],
            dims=["threshold", "stn"],
            coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
        ).T,
    },
)
FAIR_CORR_ALL_DS = xr.Dataset(
    {
        "a": xr.DataArray(
            [
                [
                    I1 * (M1 - I1) / (M1**2 * (M1 - 1)),
                    I2 * (M2 - I2) / (M2**2 * (M2 - 1)),
                    I3 * (M3 - I3) / (M3**2 * (M3 - 1)),
                    np.nan,
                    0,
                ]
            ],
            dims=["threshold", "stn"],
            coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
        ).T,
        "b": xr.DataArray(
            [
                [
                    I12 * (M1 - I12) / (M1**2 * (M1 - 1)),
                    I2 * (M2 - I2) / (M2**2 * (M2 - 1)),
                    0,
                    np.nan,
                    0,
                ],
            ],
            dims=["threshold", "stn"],
            coords={"stn": [101, 102, 103, 104, 105], "threshold": [1]},
        ).T,
    },
)
EXP_BRIER_ENS_FAIR_ALL_DS = EXP_BRIER_ENS_ALL_DS - FAIR_CORR_ALL_DS
