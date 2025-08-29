"""
Test data for Fractions Skill Score (FSS)
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

EXPECTED_TEST_FSS_2D_W2X2_T2 = xr.Dataset(
    data_vars={
        "FSS": np.array(0.968514),
        "FBS": np.array(0.041027),
        "FBS_ref": np.array(1.303052),
    }
)
EXPECTED_TEST_FSS_2D_W2X2_T8 = xr.Dataset(
    data_vars={
        "FSS": np.array(0.779375),
        "FBS": np.array(0.040113),
        "FBS_ref": np.array(0.181817),
    }
)
EXPECTED_TEST_FSS_2D_W2X2_T8_UNIFORM_BENCHMARK = xr.Dataset(
    data_vars={
        "FSS": np.array(0.779375),
        "FBS": np.array(0.040113),
        "FBS_ref": np.array(0.181817),
        "UFSS": np.array(0.60075),
    }
)
EXPECTED_TEST_FSS_2D_W2X2_T8_RANDOM_BENCHMARK = xr.Dataset(
    data_vars={
        "FSS": np.array(0.779375),
        "FBS": np.array(0.040113),
        "FBS_ref": np.array(0.181817),
        "FSS_rand": np.array(0.50011418),
    }
)
EXPECTED_TEST_FSS_2D_W5X5_T8_BOTH_BENCHMARK = xr.Dataset(
    data_vars={
        "FSS": np.array(0.94229434),
        "FBS": np.array(0.00630083),
        "FBS_ref": np.array(0.10918917),
        "UFSS": np.array(0.60075),
        "FSS_rand": np.array(0.87023263),
    }
)
EXPECTED_TEST_FSS_2D_W5X5_T2 = xr.Dataset(
    data_vars={
        "FSS": np.array(0.993893),
        "FBS": np.array(0.00752167),
        "FBS_ref": np.array(1.231545),
    }
)
EXPECTED_TEST_FSS_2D_W5X5_T8 = xr.Dataset(
    data_vars={
        "FSS": np.array(0.94229434),
        "FBS": np.array(0.00630083),
        "FBS_ref": np.array(0.10918917),
    }
)
EXPECTED_TEST_FSS_2D_W5X5_T8_ZERO_PADDED = xr.Dataset(
    data_vars={
        "FSS": np.array(0.93058573),
        "FBS": np.array(0.0055472),
        "FBS_ref": np.array(0.0799144),
    }
)
EXPECTED_TEST_FSS_2D_REDUCE_TIME = xr.Dataset(
    {
        "FSS": (("lead_time"), np.array([0.94116329, 0.90602492])),
        "FBS": (("lead_time"), np.array([0.03837719, 0.06008772])),
        "FBS_ref": (("lead_time"), np.array([0.65226608, 0.63940058])),
    },
    coords={"lead_time": [1, 2]},
)
EXPECTED_TEST_FSS_2D_W5X5_T8_REFLECTIVE_PADDED = xr.Dataset(
    data_vars={
        "FSS": np.array(0.91943553),
        "FBS": np.array(0.0091916),
        "FBS_ref": np.array(0.11409),
        "UFSS": np.array(0.60075),
        "FSS_rand": np.array(0.82945736),
    }
)
EXPECTED_TEST_FSS_2D_PRESERVE_TIME = xr.Dataset(
    {
        "FSS": (
            ("time"),
            np.array(
                [
                    0.93711612,
                    0.91529585,
                    0.92246746,
                    0.91309131,
                    0.94132104,
                    0.9136998,
                    0.92067537,
                    0.92040932,
                    0.93084956,
                    0.92564655,
                ]
            ),
        ),
        "FBS": (
            ("time"),
            np.array(
                [
                    0.03929094,
                    0.06304825,
                    0.0500731,
                    0.05774854,
                    0.03490497,
                    0.05537281,
                    0.04550439,
                    0.05116959,
                    0.04477339,
                    0.0504386,
                ]
            ),
        ),
        "FBS_ref": (
            ("time"),
            np.array(
                [
                    0.62481725,
                    0.7443348,
                    0.64583333,
                    0.66447368,
                    0.59484649,
                    0.64163012,
                    0.57364766,
                    0.64290936,
                    0.64747807,
                    0.67836257,
                ]
            ),
        ),
    },
    coords={"time": _TIME_SERIES},
)
EXPECTED_TEST_FSS_2D_PRESERVE_ALL = xr.Dataset(
    {
        "FSS": (
            ("lead_time", "time"),
            np.array(
                [
                    [
                        0.96859903,
                        0.93714822,
                        0.94111111,
                        0.92799134,
                        0.95023697,
                        0.93723849,
                        0.94783715,
                        0.93268187,
                        0.93372093,
                        0.9394572,
                    ],
                    [
                        0.90754396,
                        0.89129315,
                        0.90311419,
                        0.89770822,
                        0.93171666,
                        0.89227421,
                        0.89342693,
                        0.90692124,
                        0.92814043,
                        0.91091314,
                    ],
                ]
            ),
        ),
        "FBS": (
            ("lead_time", "time"),
            np.array(
                [
                    [
                        0.01900585,
                        0.04897661,
                        0.03874269,
                        0.04861111,
                        0.03070175,
                        0.03837719,
                        0.02997076,
                        0.04532164,
                        0.04166667,
                        0.04239766,
                    ],
                    [
                        0.05957602,
                        0.07711988,
                        0.06140351,
                        0.06688596,
                        0.03910819,
                        0.07236842,
                        0.06103801,
                        0.05701754,
                        0.04788012,
                        0.05847953,
                    ],
                ]
            ),
        ),
        "FBS_ref": (
            ("lead_time", "time"),
            np.array(
                [
                    [
                        0.60526316,
                        0.77923977,
                        0.65789474,
                        0.6750731,
                        0.61695906,
                        0.61147661,
                        0.5745614,
                        0.67324561,
                        0.62865497,
                        0.7002924,
                    ],
                    [
                        0.64437135,
                        0.70942982,
                        0.63377193,
                        0.65387427,
                        0.57273392,
                        0.67178363,
                        0.57273392,
                        0.6125731,
                        0.66630117,
                        0.65643275,
                    ],
                ]
            ),
        ),
    },
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

# NEED TO FIX
FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_OBS = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 5, 1, 1, 2, 3, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 3, 1, 3, 1, 0],
    [0, 4, 3, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_FCST = np.array([
    [2, 1, 2, 0, 3, 0, 3],
    [0, 4, 0, 1, 3, 4, 3],
    [2, 1, 2, 0, 3, 0, 3],
    [4, 1, 4, 3, 4, 2, 4],
    [4, 5, 4, 3, 3, 2, 3],
    [4, 1, 4, 3, 4, 2, 4],
])

FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_OBS = np.array([
    [1, 1, 1, 0, 1, 1, 1],
    [1, 5, 1, 1, 2, 3, 2],
    [1, 1, 1, 0, 1, 1, 1],
    [3, 1, 3, 1, 3, 1, 3],
    [3, 4, 3, 2, 1, 0, 1],
    [3, 1, 3, 1, 3, 1, 3],
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

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_FCST = np.array([
    [3, 4, 1, 4, 3, 4, 2, 2, 2, 4],
    [0, 2, 1, 2, 0, 3, 0, 3, 0, 3],
    [1, 0, 4, 0, 1, 3, 4, 3, 4, 3],
    [0, 2, 1, 2, 0, 3, 0, 3, 0, 3],
    [3, 4, 1, 4, 3, 4, 2, 2, 2, 4],
    [3, 4, 5, 4, 3, 3, 2, 2, 2, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 4, 5, 4, 3, 3, 2, 2, 2, 3],
    [3, 4, 1, 4, 3, 4, 2, 2, 2, 4],
])

FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_OBS = np.array([
    [1, 3, 1, 3, 1, 3, 1, 4, 1, 3],
    [0, 1, 1, 1, 0, 1, 1, 4, 1, 1],
    [1, 1, 5, 1, 1, 2, 3, 4, 3, 2],
    [0, 1, 1, 1, 0, 1, 1, 4, 1, 1],
    [1, 3, 1, 3, 1, 3, 1, 4, 1, 3],
    [2, 3, 4, 3, 2, 1, 0, 4, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 4, 1, 1],
    [2, 3, 4, 3, 2, 1, 0, 4, 0, 1],
    [1, 3, 1, 3, 1, 3, 1, 4, 1, 3],
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
    count = 0
    for i in range(0, h - ws + 1):
        for j in range(0, w - ws + 1):
            sum_fcst = 0
            sum_obs = 0
            for k in range(i, i + ws):
                for l in range(j, j + ws):
                    count += 1
                    sum_fcst += 1 if fcst[k][l] > th else 0
                    sum_obs += 1 if obs[k][l] > th else 0
            sum_sq_fcst += sum_fcst * sum_fcst
            sum_sq_obs += sum_obs * sum_obs
            sum_sq_diff += (sum_fcst - sum_obs) * (sum_fcst - sum_obs)
    numer = sum_sq_diff / (count * ws * ws)
    denom = (sum_sq_fcst + sum_sq_obs) / (count * ws * ws)
    fss = 1 - numer / denom
    return fss, numer, denom


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

EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED = _fss_naive_small_data(
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_OBS,
    FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_FCST,
    FSS_CURATED_WINDOW_SIZE_3X3,
    FSS_CURATED_THRESHOLD,
)

EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED = _fss_naive_small_data(
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_OBS,
    FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_FCST,
    FSS_CURATED_WINDOW_SIZE_4X4,
    FSS_CURATED_THRESHOLD,
)
