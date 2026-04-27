"""
Contains unit tests for scores.probability.auc_impl.roc_auc
"""

import warnings

try:
    import dask
    import dask.array
except ImportError:
    dask = "Unavailable"

import numpy as np
import pytest
import xarray as xr

from scores.probability import roc_auc, roc_curve_data
from tests.plotdata import roc_test_data as rtd


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "reduce_dims", "weights"),
    [
        # preserve_dims=['lead_day']
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, ["lead_day"], None, None),
        # reduce_dims=['letter', 'pet']
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, None, ["letter", "pet"], None),
        # preserve_dims=None, reduce_dims=None
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, None, None, None),
        # Weighting by lead_day, all dims reduced
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, None, None, rtd.LEAD_DAY_WEIGHTS),
        # Weighting by lead_day and reducing all dims except lead_day should produce identical results
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, None, ["letter", "pet"], rtd.LEAD_DAY_WEIGHTS),
        # preserve_dims=['lead_day', 'letter']
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, ["lead_day", "letter"], None, None),
        # preserve_dims=['letter', 'lead_day'] (dim order switched)
        (rtd.FCST_2X3X2_WITH_NAN, rtd.OBS_3X3_WITH_NAN, ["letter", "lead_day"], None, None),
    ],
)
def test_roc_auc(fcst, obs, preserve_dims, reduce_dims, weights):
    """
    Tests roc_auc with a variety of inputs.
    These are regression tests against the expected AUC values computed by roc_curve_data
    which were calculated by hand.
    """
    expected_auc = roc_curve_data(fcst, obs, preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights)
    expected_auc = expected_auc["AUC"]
    expected_auc.attrs = {}
    result = roc_auc(fcst, obs, preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights)
    xr.testing.assert_allclose(result, expected_auc)


@pytest.mark.parametrize("check_args", [True, False])
def test_roc_auc_dask(check_args):
    """Tests that roc_auc works with Dask-backed arrays."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst = rtd.FCST_2X3X2_WITH_NAN.chunk()
    obs = rtd.OBS_3X3_WITH_NAN.chunk()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = roc_curve_data(fcst, obs, preserve_dims=["letter", "lead_day"], check_args=check_args)["AUC"]
    expected.attrs = {}

    if check_args:
        with pytest.warns(UserWarning, match="`fcst` or `obs` is an xarray object backed by a Dask array"):
            result = roc_auc(
                fcst,
                obs,
                preserve_dims=["letter", "lead_day"],
                check_args=check_args,
            )
    else:
        result = roc_auc(
            fcst,
            obs,
            preserve_dims=["letter", "lead_day"],
            check_args=check_args,
        )

    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    expected = expected.compute()
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("fcst", "obs", "error_class", "error_msg_snippet"),
    [
        # fcst has invalid values
        (
            xr.DataArray([43, 100, 1, 0]),
            xr.DataArray([0, 1, 1, 0]),
            ValueError,
            "`fcst` contains values outside of the range [0, 1]",
        ),
        # fcst has invalid values
        (
            xr.DataArray([-0.1, 0.5, 1, 0]),
            xr.DataArray([0, 1, 1, 0]),
            ValueError,
            "`fcst` contains values outside of the range [0, 1]",
        ),
        # obs has invalid values
        (
            xr.DataArray([0, 0.3, 1, 0]),
            xr.DataArray([0, 1, 1, 0.5]),
            ValueError,
            "`obs` contains values that are not in the set {0, 1, np.nan}",
        ),
    ],
)
def test_roc_auc_raises(fcst, obs, error_class, error_msg_snippet):
    """Tests that roc_auc raises the correct error."""
    with pytest.raises(error_class) as exc:
        roc_auc(fcst, obs)
    assert error_msg_snippet in str(exc.value)


def test_roc_auc_preserve_all_dims():
    """preserve_dims='all' raises ValueError."""
    fcst = xr.DataArray([0.9, 0.1], dims=["sample"])
    obs = xr.DataArray([1, 0], dims=["sample"])
    with pytest.raises(ValueError, match="`preserve_dims='all'` is not supported"):
        roc_auc(fcst, obs, preserve_dims="all")


def test_roc_auc_all_same_class():
    """_roc_auc_mann_whitney returns NaN when all obs are the same class"""
    fcst = xr.DataArray([0.9, 0.8, 0.7], dims=["sample"])
    obs = xr.DataArray([1, 1, 1], dims=["sample"])
    result = roc_auc(fcst, obs, check_args=False)
    assert np.isnan(float(result))


def test_roc_auc_zero_pos_weight():
    """_roc_auc_mann_whitney_weighted returns NaN when total positive weight is zero."""
    fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
    obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
    weights = xr.DataArray([0.0, 0.0, 1.0, 1.0], dims=["sample"])  # all positives have zero weight
    result = roc_auc(fcst, obs, weights=weights, check_args=False)
    assert np.isnan(float(result))


def test_roc_auc_dataset_vs_roc_curve_data():
    """
    Tests that roc_auc produces the same result for xr.Dataset inputs as roc_curve_data.
    Since roc_curve_data only works on DataArrays, the expected Dataset is reconstructed
    from per-variable DataArray results.
    """
    fcst_da1 = rtd.FCST_2X3X2_WITH_NAN
    fcst_da2 = 1 - rtd.FCST_2X3X2_WITH_NAN
    obs_da = rtd.OBS_3X3_WITH_NAN

    fcst_ds = xr.Dataset({"var1": fcst_da1, "var2": fcst_da2})
    obs_ds = xr.Dataset({"var1": obs_da, "var2": obs_da})

    result = roc_auc(fcst_ds, obs_ds, preserve_dims=["lead_day"], weights=rtd.LEAD_DAY_WEIGHTS)

    expected_var1 = roc_curve_data(fcst_da1, obs_da, preserve_dims=["lead_day"], weights=rtd.LEAD_DAY_WEIGHTS)["AUC"]
    expected_var1.attrs = {}
    expected_var2 = roc_curve_data(fcst_da2, obs_da, preserve_dims=["lead_day"], weights=rtd.LEAD_DAY_WEIGHTS)["AUC"]
    expected_var2.attrs = {}
    expected_ds = xr.Dataset({"var1": expected_var1, "var2": expected_var2})

    xr.testing.assert_allclose(result, expected_ds)
