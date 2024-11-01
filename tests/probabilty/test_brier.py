"""
Contains unit tests for scores.probability.brier_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.probability import brier_score, ensemble_brier_score

from tests.probabilty import brier_test_data as btd


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "reduce_dims", "expected_bs"),
    [
        # Reduce all dims
        (
            btd.FCST1,
            btd.OBS1,
            None,
            None,
            xr.DataArray(0.1),
        ),
        # preserve dim "a"
        (
            btd.FCST1,
            btd.OBS1,
            ["a"],
            None,
            xr.DataArray([0.1, 0.1], dims=["a"], coords={"a": [0, 1]}),
        ),
        # doesn't break with all NaNs
        (
            btd.FCST1,
            btd.OBS1 * np.nan,
            ["a"],
            None,
            xr.DataArray([np.nan, np.nan], dims=["a"], coords={"a": [0, 1]}),
        ),
        # Check it works with DataSets
        (
            xr.Dataset({"1": btd.FCST1, "2": 0.5 * btd.FCST1}),
            xr.Dataset({"1": btd.OBS1, "2": btd.OBS1}),
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

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = brier_score(btd.FCST1.chunk(), btd.OBS1.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, xr.DataArray(0.1))


@pytest.mark.parametrize(
    ("fcst", "obs", "error_msg_snippet"),
    [
        # Fcst > 1
        (btd.FCST1 + 0.0000001, btd.OBS1, r"`fcst` contains values outside of the range \[0, 1\]"),
        # Fcst < 0
        (btd.FCST1 - 0.0000001, btd.OBS1, r"`fcst` contains values outside of the range \[0, 1\]"),
        # Obs = 1/2
        (btd.FCST1, btd.OBS1 / 2, "`obs` contains values that are not in the set {0, 1, np.nan}"),
    ],
)
def test_brier_score_raises(fcst, obs, error_msg_snippet):
    """
    Tests that the Brier score raises the correct errors.
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        brier_score(fcst, obs)
    # Check again but with input data as a DataSet
    with pytest.raises(ValueError, match=error_msg_snippet):
        brier_score(xr.Dataset({"x": fcst}), xr.Dataset({"x": obs}))


@pytest.mark.parametrize(
    ("fcst", "obs", "expected"),
    [
        # FCST doubled
        (btd.FCST1 * 2, btd.OBS1, xr.DataArray(0.2)),
        # OBS halved
        (btd.FCST1, btd.OBS1 / 2, xr.DataArray(0.05)),
    ],
)
def test_brier_doesnt_raise(fcst, obs, expected):
    """
    Tests that the Brier score doesn't raise an error when check_args=False
    """
    result = brier_score(fcst, obs, check_args=False)
    xr.testing.assert_equal(result, expected)

    # Check again but with input data as a DataSet
    result = brier_score(xr.Dataset({"x": fcst}), xr.Dataset({"x": obs}), check_args=False)
    xr.testing.assert_equal(result, xr.Dataset({"x": expected}))


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "ensemble_member_dim",
        "thresholds",
        "preserve_dims",
        "reduce_dims",
        "weights",
        "fair_correction",
        "expected",
    ),
    [
        # Fair=False, single threshold, preserve all
        (
            btd.DA_FCST_ENS,
            btd.DA_OBS_ENS,
            "ens_member",
            1,
            "all",
            None,
            None,
            False,
            btd.EXP_BRIER_ENS_ALL,
        ),
        # Fair=True, single threshold, preserve all
        (
            btd.DA_FCST_ENS,
            btd.DA_OBS_ENS,
            "ens_member",
            1,
            "all",
            None,
            None,
            True,
            btd.EXP_BRIER_ENS_FAIR_ALL,
        ),
        # Test reduce_dim arg
        (
            btd.DA_FCST_ENS,
            btd.DA_OBS_ENS,
            "ens_member",
            1,
            None,
            "stn",
            None,
            True,
            btd.EXP_BRIER_ENS_FAIR_ALL_MEAN,
        ),
        # Fair=False, multiple_thresholds, preserve all
        (
            btd.DA_FCST_ENS,
            btd.DA_OBS_ENS,
            "ens_member",
            [-100, 1, 100],
            "all",
            None,
            None,
            False,
            btd.EXP_BRIER_ENS_ALL_MULTI,
        ),
        # Test with broadcast with a lead day dimension with Fair=True
        (
            btd.DA_FCST_ENS_LT,
            btd.DA_OBS_ENS,
            "ens_member",
            1,
            "all",
            None,
            None,
            True,
            btd.EXP_BRIER_ENS_FAIR_ALL_LT,
        ),
        # Test with weights
        (
            btd.DA_FCST_ENS,
            btd.DA_OBS_ENS,
            "ens_member",
            1,
            None,
            "stn",
            btd.ENS_BRIER_WEIGHTS,
            False,
            btd.EXP_BRIER_ENS_WITH_WEIGHTS,
        ),
        # Test with Datasets
        (
            btd.FCST_ENS_DS,
            btd.OBS_ENS_DS,
            "ens_member",
            1,
            "all",
            None,
            None,
            True,
            btd.EXP_BRIER_ENS_FAIR_ALL_DS,
        ),
    ],
)
def test_ensemble_brier_score(
    fcst, obs, ensemble_member_dim, thresholds, preserve_dims, reduce_dims, weights, fair_correction, expected
):
    """Tests ensemble_brier_score."""
    result = ensemble_brier_score(
        fcst,
        obs,
        ensemble_member_dim,
        thresholds,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        weights=weights,
        fair_correction=fair_correction,
    )
    xr.testing.assert_equal(result, expected)
