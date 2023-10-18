"""
Contains unit tests for scores.probability.roc_impl
"""
import dask
import numpy as np
import pytest
import xarray as xr

from scores.probability import roc_curve_data
from tests.probabilty import roc_test_data as rtd


@pytest.mark.parametrize(
    ("fcst", "obs", "thresholds", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        # preserve_dims=['lead_day']
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            ["lead_day"],
            None,
            None,
            rtd.EXP_ROC_LEADDAY,
        ),
        # reduce_dims=['letter', 'pet']
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            None,
            ["letter", "pet"],
            None,
            rtd.EXP_ROC_LEADDAY,
        ),
        # Weighting by lead day and reducing all dims except `lead_day` should produce identical results
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            None,
            ["letter", "pet"],
            rtd.LEAD_DAY_WEIGHTS,
            rtd.EXP_ROC_LEADDAY,
        ),
        # preserve_dims=None, reduce_dims=None
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            None,
            None,
            None,
            rtd.EXP_ROC_NONE,
        ),
        # preserve_dims=None, reduce_dims=None, weight by lead_day
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            None,
            None,
            rtd.LEAD_DAY_WEIGHTS,
            rtd.EXP_ROC_NONE_WEIGHTED,
        ),
        # Test AUC works with multiple DIMS preserved
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            ["lead_day", "letter"],
            None,
            None,
            rtd.EXP_ROC_MULTI_DIMS,
        ),
        # Test AUC works with multiple DIMS preserved (and dim order switched)
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1],
            ["letter", "lead_day"],
            None,
            None,
            rtd.EXP_ROC_MULTI_DIMS,
        ),
    ],
)
def test_roc_curve_data(fcst, obs, thresholds, preserve_dims, reduce_dims, weights, expected):
    """
    Tests the roc_curve_data
    """
    result = roc_curve_data(
        fcst, obs, thresholds, preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights
    )
    result.broadcast_equals(expected)


def test_roc_curve_data_dask():
    """tests that roc_curve_data works with dask"""
    result = roc_curve_data(
        rtd.FCST_2X3X2_WITH_NAN.chunk(),
        rtd.OBS_3X3_WITH_NAN.chunk(),
        [0, 0.3, 1],
        preserve_dims=["letter", "lead_day"],
        check_args=False,
    )
    assert isinstance(result.POD.data, dask.array.Array)
    assert isinstance(result.POFD.data, dask.array.Array)
    assert isinstance(result.AUC.data, dask.array.Array)

    result = result.compute()

    assert isinstance(result.POD.data, np.ndarray)
    assert isinstance(result.POFD.data, np.ndarray)
    assert isinstance(result.AUC.data, np.ndarray)
    result.broadcast_equals(rtd.EXP_ROC_MULTI_DIMS)


@pytest.mark.parametrize(
    ("fcst", "obs", "thresholds", "preserve_dims", "error_class", "error_msg_snippet"),
    [
        # fcst has invalid values
        (
            xr.DataArray([43, 100, 1, 0]),
            xr.DataArray([0, 1, 1, 0]),
            [0.3],
            None,
            ValueError,
            "`fcst` contains values outside of the range [0, 1]",
        ),
        # obs has invalid values
        (
            xr.DataArray([0, 0.3, 1, 0]),
            xr.DataArray([0, 1, 1, 0.5]),
            [0.3],
            None,
            ValueError,
            "`obs` contains values that are not in the set {0, 1, np.nan}",
        ),
        # 'threshold' in fcst dims
        (
            xr.DataArray([0.5], dims=["threshold"]),
            xr.DataArray(0),
            [0.3],
            None,
            ValueError,
            "'threshold' must not be in the supplied data object dimensions",
        ),
        # put an np.inf in the thresholds
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [-np.inf, 0, 0.3, 1, np.inf],
            None,
            ValueError,
            "`thresholds` contains values outside of the range [0, 1]",
        ),
        # thresholds are not monotonic increasing
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 0.2, 1],
            None,
            ValueError,
            "`thresholds` is not monotonic increasing between 0 and 1",
        ),
    ],
)
def test_roc_curve_data_raises(fcst, obs, thresholds, preserve_dims, error_class, error_msg_snippet):
    """
    Tests that roc_curve_data raises the correct error
    """
    with pytest.raises(error_class) as exc:
        roc_curve_data(fcst, obs, thresholds, preserve_dims=preserve_dims)
    assert error_msg_snippet in str(exc.value)
