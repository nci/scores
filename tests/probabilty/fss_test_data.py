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


_periods = 10
_time_series = pd.date_range(
    start="2022-11-20T01:00:00.000000000",
    freq="h",
    periods=_periods,
)

EXPECTED_TEST_FSS_2D_REDUCE_TIME = xr.DataArray(
    data=[0.941905, 0.902831],
    coords={"lead_time": [1, 2]},
)
EXPECTED_TEST_FSS_2D_PRESERVE_TIME = xr.DataArray(
    data=[0.934381, 0.914105, 0.921647, 0.909383, 0.941691, 0.911977, 0.922896, 0.91629, 0.932657, 0.923547],
    coords={"time": _time_series},
)
EXPECTED_TEST_FSS_2D_PRESERVE_ALL = xr.DataArray(
    data=[
        [
            0.96783217,
            0.9387079,
            0.9408707,
            0.92433796,
            0.95317959,
            0.93562874,
            0.95238095,
            0.92948287,
            0.93880389,
            0.94201183,
        ],
        [
            0.90240642,
            0.88742515,
            0.90156144,
            0.8938401,
            0.92916984,
            0.88997214,
            0.89420849,
            0.90215827,
            0.92682927,
            0.90379747,
        ],
    ],
    coords={"lead_time": [1, 2], "time": _time_series},
)
