"""
Contains unit tests for scores.probability.brier_impl
"""

import dask
import dask.array
import numpy as np
import pytest
import xarray as xr

from scores.probability import brier_score

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


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "reduce_dims", "expected_bs"),
    [
        # Reduce all dims
        (
            FCST1,
            OBS1,
            None,
            None,
            xr.DataArray(0.1),
        ),
        # preserve dim "a"
        (
            FCST1,
            OBS1,
            ["a"],
            None,
            xr.DataArray([0.1, 0.1], dims=["a"], coords={"a": [0, 1]}),
        ),
        # doesn't break with all NaNs
        (
            FCST1,
            OBS1 * np.nan,
            ["a"],
            None,
            xr.DataArray([np.nan, np.nan], dims=["a"], coords={"a": [0, 1]}),
        ),
        # Check it works with DataSets
        (
            xr.Dataset({"1": FCST1, "2": 0.5 * FCST1}),
            xr.Dataset({"1": OBS1, "2": OBS1}),
            ["a"],
            None,
            xr.Dataset(
                {
                    "1": xr.DataArray([0.1, 0.1], dims=["a"], coords={"a": [0, 1]}),
                    "2": xr.DataArray([0.125, 0.125], dims=["a"], coords={"a": [0, 1]}),
                }
            ),
        ),
    ],
)
def test_brier_score(fcst, obs, preserve_dims, reduce_dims, expected_bs):
    """
    Tests brier_score.

    Note that the underlying MSE function is tested more thoroughly.
    """
    calculated_bs = brier_score(fcst, obs, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    xr.testing.assert_equal(calculated_bs, expected_bs)


def test_brier_score_dask():
    """
    Tests that the Brier score works with dask
    """
    result = brier_score(FCST1.chunk(), OBS1.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_equal(result, xr.DataArray(0.1))
