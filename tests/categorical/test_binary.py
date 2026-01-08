"""
Tests scores.categorical.binary
"""

import numpy as np
import pytest
import xarray as xr

from scores.categorical import probability_of_detection, probability_of_false_detection
from scores.utils import dask_available

HAS_DASK = dask_available()

if HAS_DASK:
    import dask.array as da
else:
    da = None

fcst0 = xr.DataArray(data=[[0, 0], [0, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
fcst1 = xr.DataArray(data=[[1, 1], [1, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
fcst_mix = xr.DataArray(data=[[0, 1], [1, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
obs0 = xr.DataArray(data=[[0, 0], [0, 0]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
obs1 = xr.DataArray(data=[[1, 1], [1, 1]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
obs_mix = xr.DataArray(data=[[0, 1], [1, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
fcst_bad = xr.DataArray(data=[[0, 0.2], [0, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
obs_bad = xr.DataArray(data=[[0, 3], [0, np.nan]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})
weight_array = xr.DataArray(data=[[2, 1], [1, 1]], dims=["a", "b"], coords={"a": [100, 200], "b": [500, 600]})

expected_pod0 = xr.DataArray(data=np.nan, name="ctable_probability_of_detection")
expected_pod1 = xr.DataArray(data=1, name="ctable_probability_of_detection")
expected_pod2 = xr.DataArray(data=0, name="ctable_probability_of_detection")
expected_pod3 = xr.DataArray(data=2 / 3, name="ctable_probability_of_detection")
expected_poda = xr.DataArray(data=[0.5, 1], dims="b", coords={"b": [500, 600]}, name="ctable_probability_of_detection")
expected_pod_weighted = xr.DataArray(data=1 / 2, name="ctable_probability_of_detection")


expected_pofd0 = xr.DataArray(data=0, name="ctable_probability_of_false_detection")
expected_pofd1 = xr.DataArray(data=np.nan, name="ctable_probability_of_false_detection")
expected_pofd2 = xr.DataArray(data=1, name="ctable_probability_of_false_detection")
expected_pofd3 = xr.DataArray(data=2 / 3, name="ctable_probability_of_false_detection")
expected_pofda = xr.DataArray(
    data=[0.5, 1], dims="b", coords={"b": [500, 600]}, name="ctable_probability_of_false_detection"
)
expected_pofd_weighted = xr.DataArray(data=1 / 2, name="ctable_probability_of_detection")


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "check_args", "weights", "expected"),
    [
        (fcst0, obs0, None, True, None, expected_pod0),  # Fcst zeros, obs zeros
        (fcst1, obs1, None, True, None, expected_pod1),  # Fcst ones, obs ones
        (fcst1, obs0, None, True, None, expected_pod0),  # Fcst ones, obs zeros
        (fcst0, obs1, None, True, None, expected_pod2),  # Fcst zeros, obs ones
        (fcst_mix, obs1, None, True, None, expected_pod3),  # Fcst mixed, obs ones
        (fcst1, obs_mix, None, True, None, expected_pod1),  # Fcst ones, obs mixed
        (fcst_mix, obs1, "a", True, None, expected_poda),  # Fcst mix, obs ones, only reduce one dim
        (fcst_bad, obs0, None, False, None, expected_pod0),  # Don't check for bad data
        (fcst_mix, obs1, None, True, weight_array, expected_pod_weighted),  # Fcst mixed, obs ones, with weights
        (
            xr.Dataset({"array1": fcst0, "array2": fcst1}),
            xr.Dataset({"array1": obs0, "array2": obs1}),
            None,
            True,
            None,
            xr.Dataset({"array1": expected_pod0, "array2": expected_pod1}),
        ),  # Test with DataSet for inputs
    ],
)
def test_probability_of_detection(
    fcst, obs, reduce_dims, check_args, weights, expected
):  # pylint: disable=too-many-positional-arguments
    """Tests probability_of_detection"""
    result = probability_of_detection(fcst, obs, reduce_dims=reduce_dims, weights=weights, check_args=check_args)
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_pod_dask():
    "Tests that probability_of_detection works with dask"

    result = probability_of_detection(fcst_mix.chunk(), obs1.chunk())
    assert isinstance(result.data, da.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, expected_pod3)


@pytest.mark.parametrize(
    ("fcst", "obs", "error_msg"),
    [
        (fcst_bad, obs0, "`fcst` contains values that are not in the set {0, 1, np.nan}"),
        (fcst0, obs_bad, "`obs` contains values that are not in the set {0, 1, np.nan}"),
    ],
)
def test_probability_of_detection_raises(fcst, obs, error_msg):
    """test probability_of_detection raises"""
    with pytest.raises(ValueError) as exc:
        probability_of_detection(fcst, obs)
    assert error_msg in str(exc.value)


@pytest.mark.parametrize(
    ("fcst", "obs", "reduce_dims", "check_args", "weights", "expected"),
    [
        (fcst0, obs0, None, True, None, expected_pofd0),  # Fcst zeros, obs zeros
        (fcst1, obs1, None, True, None, expected_pofd1),  # Fcst ones, obs ones
        (fcst1, obs0, None, True, None, expected_pofd2),  # Fcst ones, obs zeros
        (fcst0, obs1, None, True, None, expected_pofd1),  # Fcst zeros, obs ones
        (fcst_mix, obs0, None, True, None, expected_pofd3),  # Fcst ones, obs mixed
        (fcst_mix, obs0, "a", True, None, expected_poda),  # Fcst mix, obs ones, only reduce one dim
        (fcst_bad, obs0, None, False, None, expected_pofd0),  # Don't check for bad data
        (fcst_mix, obs0, None, True, weight_array, expected_pofd_weighted),  # Fcst mixed, obs ones, with weights
        (
            xr.Dataset({"array1": fcst0, "array2": fcst1}),
            xr.Dataset({"array1": obs0, "array2": obs1}),
            None,
            True,
            None,
            xr.Dataset({"array1": expected_pofd0, "array2": expected_pofd1}),
        ),  # Test with DataSet for inputs
    ],
)
def test_probability_of_false_detection(
    fcst, obs, reduce_dims, check_args, weights, expected
):  # pylint: disable=too-many-positional-arguments
    """Tests probability_of_false_detection"""
    result = probability_of_false_detection(fcst, obs, reduce_dims=reduce_dims, weights=weights, check_args=check_args)
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_pofd_dask():
    "Tests that probability_of_false_detection works with dask"

    result = probability_of_false_detection(fcst_mix.chunk(), obs0.chunk())
    assert isinstance(result.data, da.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(result, expected_pofd3)


@pytest.mark.parametrize(
    ("fcst", "obs", "error_msg"),
    [
        (fcst_bad, obs0, "`fcst` contains values that are not in the set {0, 1, np.nan}"),
        (fcst0, obs_bad, "`obs` contains values that are not in the set {0, 1, np.nan}"),
    ],
)
def test_probability_of_false_detection_raises(fcst, obs, error_msg):
    """test probability_of_false_detection raises"""
    with pytest.raises(ValueError) as exc:
        probability_of_false_detection(fcst, obs)
    assert error_msg in str(exc.value)
