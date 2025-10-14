"""
Contains unit tests for scores.probability.roc_impl
"""

try:
    import dask
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.plotdata import roc
from tests.plotdata import roc_test_data as rtd


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
        # check behaviour when 0 is not supplied as a threshold
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0.3, 1],
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
def test_roc(fcst, obs, thresholds, preserve_dims, reduce_dims, weights, expected):
    """
    Tests the roc
    """
    result = roc(fcst, obs, thresholds, preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights)
    result.broadcast_equals(expected)


def test_roc_curve_auc():
    """
    Tests that the bug highlighted in issue #857 is fixed.
    """
    fcst1 = xr.DataArray(data=[0.3, 0.5, 0.7, 0.9, 0.9], dims="time")
    fcst2 = xr.DataArray(data=[0.3, 0.5, 0.7, 1, 1], dims="time")
    obs = xr.DataArray(data=[0, 0, 1, 1, 0], dims="time")

    result_a = roc(fcst1, obs, np.arange(0, 1.05, 0.1))
    result_b = roc(fcst2, obs, np.arange(0, 1.05, 0.1))
    xr.testing.assert_allclose(result_a.AUC, xr.DataArray(0.75))
    xr.testing.assert_allclose(result_b.AUC, xr.DataArray(0.75))


def test_roc_auto_threshold():
    fcst = xr.DataArray(data=[0.1, 0.4, 0.3, 0.9], dims="time")
    obs = xr.DataArray(data=[0, 0, 1, 1], dims="time")
    # Test with check_args=False
    result_no_check = roc(fcst, obs, check_args=False)
    xr.testing.assert_equal(result_no_check, rtd.EXP_ROC_AUTO)
    # Test with check_args=True
    result_check = roc(fcst, obs, check_args=True)
    xr.testing.assert_equal(result_check, rtd.EXP_ROC_AUTO)


def test_roc_dask():
    """tests that roc works with dask"""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = roc(
        rtd.FCST_2X3X2_WITH_NAN.chunk(),
        rtd.OBS_3X3_WITH_NAN.chunk(),
        [0, 0.3, 1],
        preserve_dims=["letter", "lead_day"],
        check_args=False,
    )
    assert isinstance(result.POD.data, dask.array.Array)  # type: ignore
    assert isinstance(result.POFD.data, dask.array.Array)  # type: ignore
    assert isinstance(result.AUC.data, dask.array.Array)  # type: ignore

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
        # put an -np.inf in the thresholds
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [-np.inf, 0, 0.3, 1],
            None,
            ValueError,
            "`thresholds` contains values outside of the range [0, 1]",
        ),
        # put an np.inf in the thresholds
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [0, 0.3, 1, np.inf],
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
        # threshold arg is a str that is not "auto"
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            "manual",
            None,
            ValueError,
            "If `thresholds` is a str, then it must be set to 'auto'",
        ),
        # threshold arg is an empty iterable
        (
            rtd.FCST_2X3X2_WITH_NAN,
            rtd.OBS_3X3_WITH_NAN,
            [],
            None,
            ValueError,
            "`thresholds` must not be empty",
        ),
    ],
)
def test_roc_raises(fcst, obs, thresholds, preserve_dims, error_class, error_msg_snippet):
    """
    Tests that roc raises the correct error
    """
    with pytest.raises(error_class) as exc:
        roc(fcst, obs, thresholds, preserve_dims=preserve_dims)
    assert error_msg_snippet in str(exc.value)


def test_roc_warns():
    """
    Tests that roc raises the correct warning when thresholds are automatically
    generated and the number of thresholds is large.
    """
    fcst = xr.DataArray(data=np.linspace(0, 1, 1001), dims="time")
    obs = fcst * 0
    with pytest.warns(
        UserWarning,
        match=(
            "Number of automatically generated thresholds is very large \\(>1000\\). "
            "If performance is slow, consider supplying thresholds manually as an iterable of floats."
        ),
    ):
        roc(fcst, obs)
