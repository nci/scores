"""
Unit tests for scores.probability.rank_hist_impl.py
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover


import pytest
import xarray as xr
from numpy import nan
import numpy as np

import scores.probability.rank_hist_impl as rh



DA_FCST = xr.DataArray(
    data=[
        [[2, 4, 1, 5], [3, 3, 3, 0], [3, nan, 2, 1]],
        [[5, 7, 9, 7], [1, 3, 2, 1], [2, 6, 4, 4]],
        [[4, 3, 6, 5], [2, 5, 3, 7], [1, 4, 2, 6]],
    ],
    dims=["stn", "lead_day", "ens_member"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1, 2], "ens_member": [1, 2, 3, 4]},
)
DA_OBS = xr.DataArray(data=[3, 4, nan], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_VAR = xr.DataArray(
    data=[
        [[0, 0, 1, 0, 0], [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4], [nan, nan, nan, nan, nan]],
        [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1 / 3, 1 / 3, 1 / 3, 0]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "rank"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1, 2], "rank": [1, 2, 3, 4, 5]},
)
DS_FCST = xr.merge([DA_FCST.rename("temp"), DA_FCST.rename("prcp")])
DS_OBS = xr.merge([DA_OBS.rename("temp"), DA_OBS.rename("prcp")])
EXP_VAR2 = xr.merge([EXP_VAR.rename("temp"), EXP_VAR.rename("prcp")])

# reduce all dimensions
EXP_RH1 = (
    xr.DataArray(
        data=[1, 1 / 4 + 1 / 3, 1 + 1 / 4 + 1 / 3, 1 / 4 + 1 / 3, 1 + 1 / 4],
        dims=["rank"],
        coords={"rank": [1, 2, 3, 4, 5]},
    )
    / 5
)
DA_WT = xr.DataArray(data=[2, 1, 4], dims=["stn"], coords={"stn": [101, 102, 103]})
# reduce all dimensions with DA_WT
EXP_RH2 = (
    xr.DataArray(
        data=[1, 2 / 4 + 1 / 3, 2 + 2 / 4 + 1 / 3, 2 / 4 + 1 / 3, 1 + 2 / 4],
        dims=["rank"],
        coords={"rank": [1, 2, 3, 4, 5]},
    )
    / 7
)
# keep lead day dim, no weights
EXP_RH3 = xr.DataArray(
    data=[
        [1 / 2, 0, 1 / 2, 0, 0],
        [0, 1 / 8, 1 / 8, 1 / 8, 5 / 8],
        [0, 1 / 3, 1 / 3, 1 / 3, 0],
    ],
    dims=["lead_day", "rank"],
    coords={"lead_day": [0, 1, 2], "rank": [1, 2, 3, 4, 5]},
)
DS_WT = xr.merge([DA_WT.rename("temp"), DA_WT.rename("prcp")])
EXP_RH4 = xr.merge([EXP_RH2.rename("temp"), EXP_RH2.rename("prcp")])


@pytest.mark.parametrize(
    ("fcst", "obs", "expected"),
    [
        (DA_FCST, DA_OBS, EXP_VAR),
        (DS_FCST, DS_OBS, EXP_VAR2),
        (DA_FCST, DS_OBS, EXP_VAR2),
        (DS_FCST, DA_OBS, EXP_VAR2),
    ],
)
def test__value_at_rank(fcst, obs, expected):
    """Tests that `_value_at_rank` returns as expected."""
    result = rh._value_at_rank(fcst, obs, "ens_member")
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        (DA_FCST, DA_OBS, None, "all", None, EXP_VAR),
        (DA_FCST, DA_OBS, "all", None, None, EXP_RH1),
        (DA_FCST, DA_OBS, "all", None, DA_WT, EXP_RH2),
        (DA_FCST, DA_OBS, "stn", None, None, EXP_RH3),
        (DA_FCST, DA_OBS, None, "lead_day", None, EXP_RH3),
        (DA_FCST, DA_OBS, None, None, DS_WT, EXP_RH4),
        (DS_FCST, DA_OBS, None, None, DS_WT, EXP_RH4),
        (DS_FCST, DS_OBS, None, None, DS_WT, EXP_RH4),
        (DS_FCST, DA_OBS, None, None, DA_WT, EXP_RH4),
    ],
)
def test_rank_histogram(fcst, obs, reduce_dims, preserve_dims, weights, expected):
    """Tests that `_value_at_rank` returns as expected."""
    result = rh.rank_histogram(
        fcst, obs, "ens_member", reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("fcst"),
    [(DA_FCST), (DS_FCST)],
)
def test_rank_histogram_warns(fcst):
    """Tests that rank_histogram warns as expected."""
    with pytest.warns(UserWarning, match="Encountered a NaN in"):
        rh.rank_histogram(fcst, DA_OBS, "ens_member")


def test_rank_histogram_dask():
    """
    Tests that rank_histogram works with dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = rh.rank_histogram(DA_FCST.chunk(), DA_OBS.chunk(), "ens_member")
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, EXP_RH1)
