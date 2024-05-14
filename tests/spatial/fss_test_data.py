"""
Test data for Forecast Skill Score (FSS)    
"""

import numpy as np
import pandas as pd
import xarray as xr


def generate(obs_pdf, fcst_pdf, *, seed=42):
    """
    Generates random 2-D array for obs/fcst representing a 2-D grid
    Args:
        obs_pdf: (mu, stddev)
        fcst_pdf: (mu, stddev)

    where,
        mu := mean
        stddev := standard deviation
    in,
        Normal distribution ~ N(mu, stddev)
    """
    np.random.seed(seed)
    h = 40
    w = 60
    obs = np.random.normal(loc=obs_pdf[0], scale=obs_pdf[1], size=(h, w))
    fcst = np.random.normal(loc=fcst_pdf[0], scale=fcst_pdf[1], size=(h, w))
    return (obs, fcst)


_PERIODS = 10
_TIME_SERIES = pd.date_range(
    start="2022-11-20T01:00:00.000000000",
    freq="h",
    periods=_PERIODS,
)

EXPECTED_TEST_FSS_2D_REDUCE_TIME = xr.DataArray(
    data=[0.941163, 0.906025],
    coords={"lead_time": [1, 2]},
)
EXPECTED_TEST_FSS_2D_PRESERVE_TIME = xr.DataArray(
    data=[0.937116, 0.915296, 0.922467, 0.913091, 0.941321, 0.9137, 0.920675, 0.920409, 0.93085, 0.925647],
    coords={"time": _TIME_SERIES},
)
EXPECTED_TEST_FSS_2D_PRESERVE_ALL = xr.DataArray(
    data=[
        [0.968599, 0.937148, 0.941111, 0.927991, 0.950237, 0.937238, 0.947837, 0.932682, 0.933721, 0.939457],
        [0.907544, 0.891293, 0.903114, 0.897708, 0.931717, 0.892274, 0.893427, 0.906921, 0.92814, 0.910913],
    ],
    coords={"lead_time": [1, 2], "time": _TIME_SERIES},
)

FSS_CURATED_THRESHOLD = 2
FSS_CURATED_WINDOW_SIZE_3X3 = 3
FSS_CURATED_WINDOW_SIZE_4X4 = 4

# fmt: off
FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS = np.array([
    [5, 1, 1, 2, 3],
    [1, 1, 0, 1, 1],
    [1, 3, 1, 3, 1],
    [4, 3, 2, 1, 0],
])

FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST = np.array([
    [4, 0, 1, 3, 4],
    [1, 2, 0, 3, 0],
    [1, 4, 3, 4, 2],
    [5, 4, 3, 3, 2],
])

FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_OBS = np.array([
    [0, 0, 0, 0, 0, 0, 0,],
    [0, 5, 1, 1, 2, 3, 0,],
    [0, 1, 1, 0, 1, 1, 0,],
    [0, 1, 3, 1, 3, 1, 0,],
    [0, 4, 3, 2, 1, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0,],
])

FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_FCST = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 1, 3, 4, 0],
    [0, 1, 2, 0, 3, 0, 0],
    [0, 1, 4, 3, 4, 2, 0],
    [0, 5, 4, 3, 3, 2, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_OBS = np.array([
    [5, 1, 1, 2, 3, 4],
    [1, 1, 0, 1, 1, 4],
    [1, 3, 1, 3, 1, 4],
    [4, 3, 2, 1, 0, 4],
    [1, 1, 1, 1, 1, 4],
])

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_FCST = np.array([
    [4, 0, 1, 3, 4, 3],
    [1, 2, 0, 3, 0, 3],
    [1, 4, 3, 4, 2, 2],
    [5, 4, 3, 3, 2, 2],
    [2, 2, 2, 2, 2, 2],
])

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_OBS = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 1, 1, 2, 3, 4, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 4, 0, 0],
    [0, 0, 1, 3, 1, 3, 1, 4, 0, 0],
    [0, 0, 4, 3, 2, 1, 0, 4, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_FCST = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 1, 3, 4, 3, 0, 0],
    [0, 0, 1, 2, 0, 3, 0, 3, 0, 0],
    [0, 0, 1, 4, 3, 4, 2, 2, 0, 0],
    [0, 0, 5, 4, 3, 3, 2, 2, 0, 0],
    [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
# fmt: on


def _fss_naive_small_data(fcst, obs, ws, th):  # pylint: disable=too-many-locals
    """
    Naive `O(N^2 * W^2)` implementation for FSS calculations

    .. warning::
        To be used for testing and small data sizes only
    """
    (h, w) = fcst.shape
    sum_sq_fcst = 0
    sum_sq_obs = 0
    sum_sq_diff = 0
    for i in range(0, h - ws + 1):
        for j in range(0, w - ws + 1):
            sum_fcst = 0
            sum_obs = 0
            for k in range(i, i + ws):
                for l in range(j, j + ws):
                    sum_fcst += 1 if fcst[k][l] > th else 0
                    sum_obs += 1 if obs[k][l] > th else 0
            sum_sq_fcst += sum_fcst * sum_fcst
            sum_sq_obs += sum_obs * sum_obs
            sum_sq_diff += (sum_fcst - sum_obs) * (sum_fcst - sum_obs)
    fss = 1 - sum_sq_diff / (sum_sq_fcst + sum_sq_obs)
    return fss


EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW = _fss_naive_small_data(
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS,
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST,
    FSS_CURATED_WINDOW_SIZE_3X3,
    FSS_CURATED_THRESHOLD,
)

EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED = _fss_naive_small_data(
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_OBS,
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_FCST,
    FSS_CURATED_WINDOW_SIZE_3X3,
    FSS_CURATED_THRESHOLD,
)

EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW = _fss_naive_small_data(
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_OBS,
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_FCST,
    FSS_CURATED_WINDOW_SIZE_4X4,
    FSS_CURATED_THRESHOLD,
)

EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED = _fss_naive_small_data(
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_OBS,
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_FCST,
    FSS_CURATED_WINDOW_SIZE_4X4,
    FSS_CURATED_THRESHOLD,
)
