"""
Unit tests for scores.probability.pit_impl.py

To do
    - _alpha_score needs to use _alpha_score_array
    - update .alpha_score methods to call on left and right
    - uncomment failed unit tests for alpha score and see if they work
    - move test data to test data file when appropriate
"""


try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import re
import warnings

import numpy as np
import pytest
import xarray as xr
from numpy import nan
from scipy.interpolate import interp1d

from scores.probability.pit_impl import (
    Pit,
    Pit_fcst_at_obs,
    _alpha_score,
    _alpha_score_array,
    _construct_hist_values,
    _diagonal_intersection_points,
    _dims_for_mean_with_checks,
    _expected_value,
    _get_pit_x_values,
    _get_plotting_points,
    _get_plotting_points_param,
    _pit_cdfvalues,
    _pit_distribution_for_cdf,
    _pit_distribution_for_ens,
    _pit_distribution_for_jumps,
    _pit_distribution_for_unif,
    _pit_hist_left,
    _pit_hist_right,
    _pit_values_final_processing,
    _pit_values_for_cdf,
    _pit_values_for_cdf_array,
    _pit_values_for_ens,
    _pit_values_for_fcst_at_obs,
    _right_left_checks,
    _value_at_pit_cdf,
    _variance,
    _variance_integral_term,
)
from tests.probabilty import pit_test_data as ptd


def create_dataset(dataarray):
    """
    Creates a dataset with two variables from a data array.
    Used for tests where a dataset is required.
    """
    return xr.merge([dataarray.rename("tas"), dataarray.rename("pr")])


def test_create_dataset():
    """Tests that `create_dataset` returns as expected."""
    da = xr.DataArray(data=[1, 2], dims=["time"], coords={"time": [10, 11]})
    result = create_dataset(da)
    expected = xr.Dataset(
        data_vars={
            "tas": (["time"], [1, 2]),
            "pr": (["time"], [1, 2]),
        },
        coords={"time": [10, 11]},
    )
    xr.testing.assert_equal(expected, result)


# test data using create_dataset
EXP_PCVFJ2 = {"left": create_dataset(ptd.EXP_PCVFJ_LEFT), "right": create_dataset(ptd.EXP_PCVFJ_RIGHT)}
EXP_PCV2 = {"left": create_dataset(ptd.EXP_PCV_LEFT), "right": create_dataset(ptd.EXP_PCV_RIGHT)}
DS_FCST = create_dataset(ptd.DA_FCST)
DS_OBS = create_dataset(ptd.DA_OBS)
EXP_PITCDF_LEFT5 = create_dataset(ptd.EXP_PITCDF_LEFT4)
EXP_PITCDF_RIGHT5 = create_dataset(ptd.EXP_PITCDF_RIGHT4)
EXP_PITCDF5 = {"left": EXP_PITCDF_LEFT5, "right": EXP_PITCDF_RIGHT5}
EXP_PVFAO3 = {"left": create_dataset(ptd.EXP_PVFAO_LEFT0), "right": create_dataset(ptd.EXP_PVFAO_RIGHT0)}
# test data sets for plotting point functions
DS_GPP_LEFT4 = create_dataset(ptd.DA_GPP_LEFT2)
DS_GPP_RIGHT4 = create_dataset(ptd.DA_GPP_RIGHT2)
EXP_GPP_X4 = create_dataset(ptd.EXP_GPP_X2)
EXP_GPP_Y4 = create_dataset(ptd.EXP_GPP_Y2)
EXP_GPP4 = {"x_plotting_position": EXP_GPP_X4, "y_plotting_position": EXP_GPP_Y4}
# test datasets for histogram functions
LIST_CHV2 = [create_dataset(da) for da in ptd.LIST_CHV1]
DS_PH_LEFT = create_dataset(ptd.DA_PH_LEFT)
DS_PH_RIGHT = create_dataset(ptd.DA_PH_RIGHT)
EXP_HV3 = create_dataset(ptd.EXP_HV2)


def test_interp1d():
    """
    scipy.interpolate.interp1d is used twice in pit_impl.py
    to interpolate when x has duplicate values.
    This test ensures that it handles interpolation as is needed for pit_impl.
    """
    # x values are sorted, with some duplicates
    x_vals = [0, 0, 1, 1, 2, 2]
    # y values are assending, possibly with jumps at common x values
    y_vals = [0, 2, 2, 4, 5, 6]
    # evaluated values are with the range of x_vals, but not including x_vals
    eval_vals = [0.5, 1.2]
    result = interp1d(x_vals, y_vals, kind="linear")(eval_vals)
    expected = np.array([2, 4.2])
    np.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected_left", "expected_right"),
    [
        (ptd.DA_PVPF_WTS, ptd.EXP_PVFP_LEFT, ptd.EXP_PVFP_RIGHT),
        (create_dataset(ptd.DA_PVPF_WTS), create_dataset(ptd.EXP_PVFP_LEFT), create_dataset(ptd.EXP_PVFP_RIGHT)),
        (None, ptd.EXP_PVFP_LEFT1, ptd.EXP_PVFP_RIGHT1),
    ],
)
def test__pit_values_final_processing(weights, expected_left, expected_right):
    """
    Tests that `_pit_values_final_processing` returns as expected.
    This test specifically tests that the weighted means are resaled correctly for left
    and right
    """
    expected = {"left": expected_left, "right": expected_right}
    result = _pit_values_final_processing(ptd.DA_PVFP, weights, {"stn"})
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_allclose(expected[key], result[key])


@pytest.mark.parametrize(
    ("fcst_at_obs", "fcst_at_obs_left", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        (ptd.DA_FAO, ptd.DA_FAO_LEFT, None, "all", None, ptd.EXP_PVFAO0),
        (ptd.DA_FAO, None, None, "all", None, ptd.EXP_PVFAO1),
        (ptd.DA_FAO, None, None, "lead_day", ptd.DA_WT, ptd.EXP_PVFAO2),
        (ptd.DA_FAO, None, "stn", None, ptd.DA_WT, ptd.EXP_PVFAO2),
        (create_dataset(ptd.DA_FAO), create_dataset(ptd.DA_FAO_LEFT), None, "all", None, EXP_PVFAO3),
    ],
)
def test__pit_values_for_fcst_at_obs(fcst_at_obs, fcst_at_obs_left, reduce_dims, preserve_dims, weights, expected):
    """Tests that `_pit_values_for_fcst_at_obs` returns as expected."""
    result = _pit_values_for_fcst_at_obs(fcst_at_obs, fcst_at_obs_left, reduce_dims, preserve_dims, weights)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_allclose(expected[key], result[key])


@pytest.mark.parametrize(
    ("fcst_at_obs", "fcst_at_obs_left", "reduce_dims", "preserve_dims", "weights", "expected"),
    [
        (ptd.DA_FAO, ptd.DA_FAO_LEFT, None, "all", None, ptd.EXP_PVFAO0),
        (ptd.DA_FAO, None, None, "all", None, ptd.EXP_PVFAO1),
        (ptd.DA_FAO, None, None, "lead_day", ptd.DA_WT, ptd.EXP_PVFAO2),
        (ptd.DA_FAO, None, "stn", None, ptd.DA_WT, ptd.EXP_PVFAO2),
        (create_dataset(ptd.DA_FAO), create_dataset(ptd.DA_FAO_LEFT), None, "all", None, EXP_PVFAO3),
    ],
)
def test_Pit_fcst_at_obs(fcst_at_obs, fcst_at_obs_left, reduce_dims, preserve_dims, weights, expected):
    """
    Tests that `Pit_fcst_at_obs.__init__` returns as expected. The
    Test is on `.left` and `.right` attributes.
    """
    result = Pit_fcst_at_obs(
        fcst_at_obs,
        fcst_at_obs_left=fcst_at_obs_left,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(expected["left"], result.left)
    xr.testing.assert_allclose(expected["right"], result.right)


def test_Pit_fcst_at_obs_dask():
    """Tests that dask works for `Pit_fcst_at_obs`."""
    pit = Pit_fcst_at_obs(ptd.DA_FAO.chunk(), fcst_at_obs_left=ptd.DA_FAO_LEFT.chunk(), preserve_dims="all")
    left = pit.left
    right = pit.right
    assert isinstance(left.data, dask.array.Array)
    assert isinstance(right.data, dask.array.Array)
    left = left.compute()
    right = right.compute()
    assert isinstance(left.data, (np.ndarray, np.generic))
    assert isinstance(right.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(left, ptd.EXP_PVFAO0["left"])
    xr.testing.assert_equal(right, ptd.EXP_PVFAO0["right"])


def test__pit_values_for_ens():
    """Tests that `_pit_values_for_ens` returns as expected."""
    stns = [100, 101, 102, 103, 104]
    fcst = xr.DataArray(
        data=[
            [  # lead day 1
                [1, 5, 2, 7, 4, 8, 3, 5, 9, 2],
                [0, 0, 1, 5, 3, 7, 4, 0, 8, 0],
                [0, 0, 1, 5, 3, 7, 4, 0, 8, nan],
                [5, 3, 7, 9, 4, 6, 2, 3, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [  # lead day 2
                [1, 0, 2, 0, 4, 0, 3, 5, 9, 0],
                [0, 0, 5, 5, 3, 7, 4, 0, 8, 0],
                [0, 0, 1, 5, 3, 7, 4, 2, 8, 0],
                [5, 3, 0, 9, 4, 6, 2, 3, 1, 2],
                [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            ],
        ],
        dims=["lead_day", "stn", "member"],
        coords={"stn": stns, "lead_day": [0, 1], "member": range(10)},
    )
    obs = xr.DataArray(data=[0, 5, 2, nan, 3], dims=["stn"], coords={"stn": stns})
    expected = xr.DataArray(
        data=[
            [
                [0.0, 0.7, 4 / 9, nan, 1],
                [0.0, 0.6, 0.4, nan, nan],
            ],  # lower  # lead day 1  # lead day 2
            [
                [0.0, 0.8, 4 / 9, nan, 1],
                [0.4, 0.8, 0.5, nan, nan],
            ],  # upper  # lead day 1  # lead day 2
        ],
        dims=["uniform_endpoint", "lead_day", "stn"],
        coords={"uniform_endpoint": ["lower", "upper"], "stn": stns, "lead_day": [0, 1]},
    )
    result = _pit_values_for_ens(fcst, obs, "member")
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (ptd.DA_GPV, ptd.EXP_GPV),  # data array input
        (create_dataset(ptd.DA_GPV), ptd.EXP_GPV),  # dataset input
    ],
)
def test__get_pit_x_values(pit_values, expected):
    """Tests that `_get_pit_x_values` returns as expected."""
    result = _get_pit_x_values(pit_values)
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (ptd.DA_GPV, ptd.EXP_PCVFJ),
        (create_dataset(ptd.DA_GPV), EXP_PCVFJ2),
    ],  # data array input  # dataset input
)
def test__pit_distribution_for_jumps(pit_values, expected):
    """Tests that `_pit_distribution_for_jumps` returns as expected."""
    result = _pit_distribution_for_jumps(pit_values, ptd.EXP_GPV)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (ptd.DA_GPV, ptd.EXP_PCVFU),  # data array input
        (create_dataset(ptd.DA_GPV), create_dataset(ptd.EXP_PCVFU)),  # dataset input
    ],
)
def test__pit_distribution_for_unif(pit_values, expected):
    """Tests that `_pit_distribution_for_unif` returns as expected."""
    result = _pit_distribution_for_unif(pit_values, ptd.EXP_GPV)
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (ptd.DA_GPV, ptd.EXP_PCV),
        (create_dataset(ptd.DA_GPV), EXP_PCV2),
    ],  # data array input  # dataset input
)
def test__pit_cdfvalues(pit_values, expected):
    """Tests that `_pit_cdfvalues` returns as expected."""
    result = _pit_cdfvalues(pit_values)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (None, {"stn", "lead_day", "instrument"}),
        (xr.DataArray([2, 3], dims=["test"], coords={"test": [1, 2]}), {"stn", "lead_day", "instrument", "test"}),
    ],
)
def test__dims_for_mean_with_checks(weights, expected):
    """Tests that `_dims_for_mean_with_checks` returns as expected."""
    result = _dims_for_mean_with_checks(ptd.DA_FCST, ptd.DA_OBS_PVCDF, "ens_member", weights, None, None)
    assert expected == result


@pytest.mark.parametrize(
    ("fcst", "obs", "weights"),
    [
        (
            xr.DataArray(data=[0], dims=["uniform_endpoint"], coords={"uniform_endpoint": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            None,
        ),
        (
            xr.DataArray(data=[0], dims=["x_plotting_position"], coords={"x_plotting_position": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            None,
        ),
        (
            xr.DataArray(data=[0], dims=["lead_day"], coords={"lead_day": [1]}),
            xr.DataArray(data=[1], dims=["pit_x_value"], coords={"pit_x_value": [2]}),
            None,
        ),
        (
            xr.DataArray(data=[0], dims=["lead_day"], coords={"lead_day": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            xr.DataArray(data=[3], dims=["y_plotting_position"], coords={"y_plotting_position": [5]}),
        ),
        (
            xr.DataArray(data=[0], dims=["lead_day"], coords={"lead_day": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            xr.DataArray(data=[3], dims=["plotting_point"], coords={"plotting_point": [5]}),
        ),
        (
            xr.DataArray(data=[0], dims=["bin_left_endpoint"], coords={"bin_left_endpoint": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            None,
        ),
    ],
)
def test__dims_for_mean_with_checks_raises(fcst, obs, weights):
    """Test that `_dims_for_mean_with_checks` raises as expected."""
    with pytest.raises(ValueError, match="The following names are reserved and"):
        _dims_for_mean_with_checks(fcst, obs, None, weights, None, None)


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, "all", None, None, ptd.EXP_PITCDF1),
        (ptd.DA_FCST, ptd.DA_OBS, "lead_day", None, None, ptd.EXP_PITCDF2),
        (ptd.DA_FCST, ptd.DA_OBS, None, "stn", None, ptd.EXP_PITCDF2),
        (ptd.DA_FCST, ptd.DA_OBS, "lead_day", None, ptd.WTS_STN, ptd.EXP_PITCDF3),
        (ptd.DA_FCST, ptd.DA_OBS, None, "stn", ptd.WTS_STN, ptd.EXP_PITCDF3),
        (ptd.DA_FCST, ptd.DA_OBS, None, "all", None, ptd.EXP_PITCDF4),
        (DS_FCST, DS_OBS, None, "all", None, EXP_PITCDF5),  # data set example
        (DS_FCST, ptd.DA_OBS, None, "all", None, EXP_PITCDF5),  # data set/array mix
    ],
)
def test__pit_distribution_for_ens(fcst, obs, preserve_dims, reduce_dims, weights, expected):
    """Tests that `_pit_distribution_for_ens` returns as expected."""
    result = _pit_distribution_for_ens(
        fcst, obs, "ens_member", preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights
    )
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "expected_left", "expected_right"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, "all", ptd.EXP_PITCDF_LEFT1, ptd.EXP_PITCDF_RIGHT1),
        (ptd.DA_FCST, ptd.DA_OBS, None, ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4),
        (
            DS_FCST,
            DS_OBS,
            None,
            create_dataset(ptd.EXP_PITCDF_LEFT4),
            create_dataset(ptd.EXP_PITCDF_RIGHT4),
        ),  # data set example
    ],
)
def test___init__(fcst, obs, preserve_dims, expected_left, expected_right):
    """Tests that `Pit.__init__` returns as expected."""
    result = Pit(fcst, obs, "ens_member", preserve_dims=preserve_dims)
    xr.testing.assert_equal(result.left, expected_left)
    xr.testing.assert_equal(result.right, expected_right)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ptd.DA_GPP_LEFT1, ptd.DA_GPP_RIGHT1, ptd.EXP_GPP1),  # several dimensions, NaN handling
        (ptd.DA_GPP_LEFT2, ptd.DA_GPP_RIGHT2, ptd.EXP_GPP2),  # one dimension
        (ptd.DA_GPP_LEFT2, ptd.DA_GPP_LEFT2, ptd.EXP_GPP3),  # left always equals right
        (DS_GPP_LEFT4, DS_GPP_RIGHT4, EXP_GPP4),  # datasets
    ],
)
def test__get_plotting_points_param(left, right, expected):
    """Tests that `_get_plotting_points_param` returns as expected."""
    result = _get_plotting_points_param(left, right)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ptd.EXP_PITCDF_LEFT2, ptd.EXP_PITCDF_RIGHT2, ptd.EXP_PP1),
        (create_dataset(ptd.EXP_PITCDF_LEFT2), create_dataset(ptd.EXP_PITCDF_RIGHT2), create_dataset(ptd.EXP_PP1)),
    ],
)
def test__get_plotting_points(left, right, expected):
    """Tests that `_get_plotting_points` returns as expected."""
    result = _get_plotting_points(left, right)
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("fcst", "obs", "expected"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, ptd.EXP_PP1),
        (DS_FCST, DS_OBS, create_dataset(ptd.EXP_PP1)),
    ],
)
def test_plotting_points(fcst, obs, expected):
    """Tests that `Pit().plotting_points()` returns as expected."""
    result = Pit(fcst, obs, "ens_member", preserve_dims="lead_day").plotting_points()
    xr.testing.assert_equal(expected, result)


def test_plotting_points_dask():
    """Tests that the `.plotting_points` method works with dask."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST.chunk(), ptd.DA_OBS.chunk(), "ens_member", preserve_dims="lead_day").plotting_points()
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, ptd.EXP_PP1)


def test_plotting_points_parametric():
    """Tests that `Pit().plotting_points_parametric()` returns as expected."""
    result = Pit(ptd.DA_FCST, ptd.DA_OBS, "ens_member").plotting_points_parametric()
    expected = ptd.EXP_PPP
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


def test_plotting_points_parametric_dask():
    """
    Tests that the `.plotting_points_parametric` method works with dask.
    """
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST.chunk(), ptd.DA_OBS.chunk(), "ens_member").plotting_points_parametric()
    assert ptd.EXP_PPP.keys() == result.keys()
    assert isinstance(result["y_plotting_position"].data, dask.array.Array)
    # Note: we don't assert that result["x_plotting_position"] is a dask array, because it is
    # a copy of an index and dask appears not to chunk the index.
    result["y_plotting_position"] = result["y_plotting_position"].compute()
    for key in result.keys():
        xr.testing.assert_allclose(result[key], ptd.EXP_PPP[key])


EXP_VAPC1 = xr.DataArray(
    data=[(0.5 + 1.75 / 3) / 2],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.5]},
)
EXP_VAPC2 = xr.DataArray(
    data=[[(4 / 7 + 1) / 2, nan, 1]],
    dims=["pit_x_value", "stn"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0.65]},
)
EXP_VAPC3 = xr.merge([EXP_VAPC1.rename("tas"), EXP_VAPC1.rename("pr")])


@pytest.mark.parametrize(
    ("left", "right", "point", "expected"),
    [
        (ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4, 0.5, EXP_VAPC1),  # one dimension only
        (ptd.EXP_PCV_LEFT, ptd.EXP_PCV_RIGHT, 0.65, EXP_VAPC2),  # several dimensions, nans
        (EXP_PITCDF_LEFT5, EXP_PITCDF_RIGHT5, 0.5, EXP_VAPC3),  # data sets
    ],
)
def test__value_at_pit_cdf(left, right, point, expected):
    """Tests that `_value_at_pit_cdf` returns as expected."""
    result = _value_at_pit_cdf(left, right, point)
    xr.testing.assert_equal(expected, result)


def test_value_at_pit_cdf_raises():
    """Tests that `_value_at_pit_cdf` raises as expected."""
    with pytest.raises(ValueError, match="`point` must not be a value in"):
        _value_at_pit_cdf(ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4, 0.4)


@pytest.mark.parametrize(
    ("cdf_at_endpoints", "expected"),
    [
        (ptd.LIST_CHV1, ptd.EXP_CHV1),  # data arrays
        (LIST_CHV2, create_dataset(ptd.EXP_CHV1)),  # datasets
    ],
)
def test__construct_hist_values(cdf_at_endpoints, expected):
    """Tests that `_construct_hist_values` returns as expected"""
    result = _construct_hist_values(cdf_at_endpoints, 2)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("pit_left", "pit_right", "expected"),
    [
        (ptd.DA_PH_LEFT, ptd.DA_PH_RIGHT, ptd.EXP_PHL1),  # data arrays
        (DS_PH_LEFT, DS_PH_RIGHT, create_dataset(ptd.EXP_PHL1)),
    ],
)
def test__pit_hist_left(pit_left, pit_right, expected):
    """Tests that `_pit_hist_left` returns as expected"""
    result = _pit_hist_left(pit_left, pit_right, 2)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("pit_left", "pit_right", "expected"),
    [
        (ptd.DA_PH_LEFT, ptd.DA_PH_RIGHT, ptd.EXP_PHR1),  # data arrays
        (DS_PH_LEFT, DS_PH_RIGHT, create_dataset(ptd.EXP_PHR1)),  # datasets
    ],
)
def test__pit_hist_right(pit_left, pit_right, expected):
    """Tests that `_pit_hist_right` returns as expected"""
    result = _pit_hist_right(pit_left, pit_right, 2)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("fcst", "obs", "right", "expected"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, True, ptd.EXP_HV1),  # data arrays
        (ptd.DA_FCST, ptd.DA_OBS, False, ptd.EXP_HV2),  # data arrays
        (DS_FCST, DS_OBS, False, EXP_HV3),  # datasets
    ],
)
def test_hist_values(fcst, obs, right, expected):
    """Tests that `hist_values` method of Pit_for_ensemble returns as expected"""
    result = Pit(fcst, obs, "ens_member").hist_values(5, right=right)
    xr.testing.assert_allclose(expected, result)


def test_hist_values_dask():
    """
    Tests that the `.hist_values` method works with dask.
    Note that dask is used for the Pit() calculation but not for .hist_values
    """
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST.chunk(), ptd.DA_OBS.chunk(), "ens_member").hist_values(5, right=True)
    xr.testing.assert_allclose(result, ptd.EXP_HV1)


@pytest.mark.parametrize(
    ("plotting_points_par", "expected"),
    [(ptd.DICT_DIP, np.array([10 / 35, 21 / 60, 0.6])), (ptd.DICT_DIP2, np.array([]))],
)
def test__diagonal_intersection_points(plotting_points_par, expected):
    """Tests that `_diagonal_intersection_points` returns as expected."""
    result = _diagonal_intersection_points(plotting_points_par)
    np.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ptd.DA_ASA_LEFT, ptd.DA_ASA_RIGHT, ptd.EXP_ASA),
    ],
)
def test__alpha_score_array(left, right, expected):
    """Tests that `_alpha_score_array` returns as expected."""
    result = _alpha_score_array(left, right)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ptd.DA_ASA_LEFT, ptd.DA_ASA_RIGHT, ptd.EXP_ASA),  # data arrays
        (create_dataset(ptd.DA_ASA_LEFT), create_dataset(ptd.DA_ASA_RIGHT), create_dataset(ptd.EXP_ASA)),
    ],
)
def test__alpha_score(left, right, expected):
    """Tests that `_alpha_score` returns as expected."""
    result = _alpha_score(left, right)
    xr.testing.assert_allclose(expected, result)


def test_alpha_score():
    """Tests that the `.alpha_score` method returns as expected."""
    result = Pit(ptd.DA_FCST, ptd.DA_OBS, "ens_member").alpha_score()
    xr.testing.assert_allclose(ptd.EXP_AS1, result)


def test_alpha_score_dask():
    """Tests that the `.alpha_score` method works with dask."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST.chunk(), ptd.DA_OBS.chunk(), "ens_member").alpha_score()
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, ptd.EXP_AS1)


@pytest.mark.parametrize(
    ("plotting_points", "expected"),
    [
        (ptd.DA_AS, ptd.EXP_EV),  # data arrays
        (create_dataset(ptd.DA_AS), create_dataset(ptd.EXP_EV)),
    ],
)
def test__expected_value(plotting_points, expected):
    """Tests that `_expected_value` returns as expected."""
    result = _expected_value(plotting_points)
    xr.testing.assert_allclose(expected, result)


def test_expected_value():
    """Tests that the `.expected_value` method returns as expected."""
    result = Pit(ptd.DA_FCST, ptd.DA_OBS, "ens_member").expected_value()
    xr.testing.assert_allclose(ptd.EXP_EXPVAL, result)


def test_expected_value_dask():
    """Tests that the `.expected_value` method works with dask."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST.chunk(), ptd.DA_OBS.chunk(), "ens_member").expected_value()
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, ptd.EXP_EXPVAL)


@pytest.mark.parametrize(
    ("plotting_points", "expected"),
    [
        (ptd.DA_VIT, ptd.EXP_VIT),  # data arrays
        (create_dataset(ptd.DA_VIT), create_dataset(ptd.EXP_VIT)),
    ],
)
def test__variance_integral_term(plotting_points, expected):
    """Tests that `_expected_value` returns as expected."""
    result = _variance_integral_term(plotting_points)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("plotting_points", "expected"),
    [
        (ptd.DA_VAR, ptd.EXP__VAR),  # data arrays
        (create_dataset(ptd.DA_VAR), create_dataset(ptd.EXP__VAR)),
    ],
)
def test__variance(plotting_points, expected):
    """Tests that `_variance` returns as expected."""
    result = _variance(plotting_points)
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("fcst", "obs", "expected"),
    [
        (ptd.DA_FCST_VAR, ptd.DA_OBS_VAR, ptd.EXP_VAR),  # data arrays
        (create_dataset(ptd.DA_FCST_VAR), create_dataset(ptd.DA_OBS_VAR), create_dataset(ptd.EXP_VAR)),
    ],
)
def test_variance(fcst, obs, expected):
    """Tests that the `.variance` method returns as expected."""
    result = Pit(fcst, obs, "member", preserve_dims="all").variance()
    xr.testing.assert_allclose(expected, result)


def test_variance_dask():
    """Tests that the `.variance` method works with dask."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = Pit(ptd.DA_FCST_VAR.chunk(), ptd.DA_OBS_VAR.chunk(), "member", preserve_dims="all").variance()
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, ptd.EXP_VAR)


def test__pit_values_for_cdf_array():
    """Tests that `_pit_values_for_cdf_array` returns as expected."""
    result = _pit_values_for_cdf_array(ptd.DA_FCST_CDF_LEFT, ptd.DA_FCST_CDF_RIGHT, ptd.DA_OBS_PVCDF, "thld")
    xr.testing.assert_allclose(ptd.EXP__PVCDF, result)


@pytest.mark.parametrize(
    ("fcst_left", "obs", "warning_msg"),
    [
        (ptd.DA_FCST_WARN1, ptd.DA_OBS_WARN1, "greater than the maximum"),
        (ptd.DA_FCST_WARN1, ptd.DA_OBS_WARN2, "less than the minimum"),
        (ptd.DA_FCST_WARN2, ptd.DA_OBS_WARN1, "Some forecast CDF values are NaN"),
    ],
)
def test__pit_values_for_cdf_array_warns(fcst_left, obs, warning_msg):
    """Tests that _pit_values_for_cdf_array warns as expected."""
    with pytest.warns(UserWarning, match=warning_msg):
        _pit_values_for_cdf_array(fcst_left, fcst_left, obs, "thld")


@pytest.mark.parametrize(
    ("fcst_left", "fcst_right", "obs", "expected"),
    [
        (
            create_dataset(ptd.DA_FCST_CDF_LEFT),
            create_dataset(ptd.DA_FCST_CDF_RIGHT),
            create_dataset(ptd.DA_OBS_PVCDF),
            create_dataset(ptd.EXP__PVCDF),
        ),
        (ptd.DA_FCST_CDF_LEFT, ptd.DA_FCST_CDF_RIGHT, ptd.DA_OBS_PVCDF, ptd.EXP__PVCDF),
        (
            ptd.DA_FCST_CDF_LEFT,
            ptd.DA_FCST_CDF_RIGHT,
            create_dataset(ptd.DA_OBS_PVCDF),
            create_dataset(ptd.EXP__PVCDF),
        ),
        (
            create_dataset(ptd.DA_FCST_CDF_LEFT),
            create_dataset(ptd.DA_FCST_CDF_RIGHT),
            ptd.DA_OBS_PVCDF,
            create_dataset(ptd.EXP__PVCDF),
        ),
    ],
)
def test__pit_values_for_cdf(fcst_left, fcst_right, obs, expected):
    """Tests that `_pit_values_for_cdf_dataset` returns as expected."""
    result = _pit_values_for_cdf(fcst_left, fcst_right, obs, "thld")
    xr.testing.assert_allclose(expected, result)


@pytest.mark.parametrize(
    ("fcst", "obs", "fcst_left", "preserve_dims", "expected"),
    [
        (ptd.DA_FCST_CDF_RIGHT1, ptd.DA_OBS_PDCDF, ptd.DA_FCST_CDF_LEFT1, "all", ptd.EXP_PDCDF1),
        (ptd.DA_FCST_CDF_LEFT1, ptd.DA_OBS_PDCDF, None, "all", ptd.EXP_PDCDF2),
        (ptd.DA_FCST_CDF_LEFT1, ptd.DA_OBS_PDCDF, None, None, ptd.EXP_PDCDF3),
        (
            create_dataset(ptd.DA_FCST_CDF_LEFT1),
            ptd.DA_OBS_PDCDF,
            None,
            None,
            {"left": create_dataset(ptd.EXP_PDCDF_LEFT3), "right": create_dataset(ptd.EXP_PDCDF_RIGHT3)},
        ),
        (
            create_dataset(ptd.DA_FCST_CDF_LEFT1),
            create_dataset(ptd.DA_OBS_PDCDF),
            None,
            None,
            {"left": create_dataset(ptd.EXP_PDCDF_LEFT3), "right": create_dataset(ptd.EXP_PDCDF_RIGHT3)},
        ),
    ],
)
def test__pit_distribution_for_cdf(fcst, obs, fcst_left, preserve_dims, expected):
    """Tests that `_pit_distribution_for_cdf` returns as expected."""
    result = _pit_distribution_for_cdf(fcst, obs, "thld", fcst_left=fcst_left, preserve_dims=preserve_dims)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("fcst", "fcst_left"),
    [
        (ptd.DA_FCST_CDF_RIGHT1, ptd.DA_FCST_CDF_LEFT_RAISES),
        (create_dataset(ptd.DA_FCST_CDF_RIGHT1), create_dataset(ptd.DA_FCST_CDF_LEFT1).rename({"tas": "rd"})),
    ],
)
def test__pit_distribution_for_cdf_raises(fcst, fcst_left):
    """Tests that `_pit_distribution_for_cdf` raises as expected."""
    with pytest.raises(ValueError, match="`fcst` and `fcst_left` must have same shape"):
        _pit_distribution_for_cdf(fcst, ptd.DA_OBS_PVCDF, "thld", fcst_left=fcst_left)


def test_Pit__init___raises():
    """Tests that `Pit.__init__` raises as expected."""
    with pytest.raises(ValueError, match='`fcst_type` must be one of "ensemble" or "cdf"'):
        Pit(ptd.DA_FCST, ptd.DA_OBS, "member", fcst_type="PDF")


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "special_fcst_dim",
        "fcst_type",
        "fcst_left",
        "reduce_dims",
        "preserve_dims",
        "weights",
        "expected_left",
        "expected_right",
    ),
    [
        (
            ptd.DA_FCST_CDF_LEFT1,
            ptd.DA_OBS_PDCDF,
            "thld",
            "cdf",
            None,
            "all",
            None,
            None,
            ptd.EXP_PDCDF_LEFT3,
            ptd.EXP_PDCDF_RIGHT3,
        ),
        (
            ptd.DA_FCST_CDF_LEFT1,
            ptd.DA_OBS_PDCDF,
            "thld",
            "cdf",
            None,
            None,
            "all",
            None,
            ptd.EXP_PDCDF_LEFT2,
            ptd.EXP_PDCDF_RIGHT2,
        ),
        (
            ptd.DA_FCST_CDF_RIGHT1,
            ptd.DA_OBS_PDCDF,
            "thld",
            "cdf",
            ptd.DA_FCST_CDF_LEFT1,
            None,
            "all",
            None,
            ptd.EXP_PDCDF_LEFT1,
            ptd.EXP_PDCDF_RIGHT1,
        ),
        (
            ptd.DA_FCST,
            ptd.DA_OBS,
            "ens_member",
            "ensemble",
            None,
            None,
            "all",
            None,
            ptd.EXP_PITCDF_LEFT1,
            ptd.EXP_PITCDF_RIGHT1,
        ),
        (
            ptd.DA_FCST,
            ptd.DA_OBS,
            "ens_member",
            "ensemble",
            None,
            None,
            "lead_day",
            ptd.WTS_STN,
            ptd.EXP_PITCDF_LEFT3,
            ptd.EXP_PITCDF_RIGHT3,
        ),
    ],
)
def test_Pit__init__(
    fcst,
    obs,
    special_fcst_dim,
    fcst_type,
    fcst_left,
    reduce_dims,
    preserve_dims,
    weights,
    expected_left,
    expected_right,
):
    """Tests that `Pit.__init__` returns as expected."""
    result = Pit(
        fcst,
        obs,
        special_fcst_dim,
        fcst_type=fcst_type,
        fcst_left=fcst_left,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_equal(expected_left, result.left)
    xr.testing.assert_equal(expected_right, result.right)


@pytest.mark.parametrize(
    ("right", "left", "threshold_dim", "error_msg"),
    [
        (3 * ptd.DA_RLC1, None, "thld", re.escape("`xr_right` values must be between 0 and 1 inclusive.")),
        (
            create_dataset(3 * ptd.DA_RLC1),
            None,
            "thld",
            re.escape("`xr_right` values must be between 0 and 1 inclusive."),
        ),
        (
            ptd.DA_RLC1,
            None,
            "thld",
            re.escape("coordinates along `xr_right[threshold_dim]` are not strictly increasing"),
        ),
        (
            create_dataset(ptd.DA_RLC1),
            None,
            "thld",
            re.escape("coordinates along `xr_right[threshold_dim]` are not strictly increasing"),
        ),
        (ptd.DA_RLC2, ptd.DA_RLC1, None, re.escape("`xr_right` and `xr_left` must have same shape")),
        (ptd.DA_RLC2, 3 * ptd.DA_RLC2, None, re.escape("`xr_left` values must be between 0 and 1 inclusive.")),
        (ptd.DA_RLC2, ptd.DA_RLC3, None, re.escape("`xr_left` must not exceed `xr_right`")),
        (
            create_dataset(ptd.DA_RLC2),
            create_dataset(3 * ptd.DA_RLC2),
            None,
            re.escape("`xr_left` values must be between 0 and 1 inclusive."),
        ),
        (
            create_dataset(ptd.DA_RLC2),
            create_dataset(ptd.DA_RLC3),
            None,
            re.escape("`xr_left` must not exceed `xr_right`"),
        ),
    ],
)
def test__right_left_checks(right, left, threshold_dim, error_msg):
    """Tests that _right_left_checks raises as expected."""
    with pytest.raises(ValueError, match=error_msg):
        _right_left_checks(right, left, threshold_dim, "xr_right", "xr_left")


@pytest.mark.parametrize(
    ("fcst", "obs", "special_fcst_dim", "fcst_type", "fcst_left", "preserve_dims", "exp_left", "exp_right"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, "ens_member", "ensemble", None, None, ptd.EXP_PITCDF_LEFT4, ptd.EXP_PITCDF_RIGHT4),
        (
            ptd.DA_FCST_CDF_RIGHT1,
            ptd.DA_OBS_PDCDF,
            "thld",
            "cdf",
            ptd.DA_FCST_CDF_LEFT1,
            "all",
            ptd.EXP_PDCDF_LEFT1,
            ptd.EXP_PDCDF_RIGHT1,
        ),
    ],
)
def test_pit__left_right_dask(fcst, obs, special_fcst_dim, fcst_type, fcst_left, preserve_dims, exp_left, exp_right):
    """
    Tests that Pit works with dask. The test is done against Pit().left and Pit().right
    """
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    pit = Pit(
        fcst.chunk(),
        obs.chunk(),
        special_fcst_dim,
        fcst_type=fcst_type,
        fcst_left=fcst_left,
        preserve_dims=preserve_dims,
    )
    left = pit.left
    right = pit.right
    assert isinstance(left.data, dask.array.Array)
    assert isinstance(right.data, dask.array.Array)
    left = left.compute()
    right = right.compute()
    assert isinstance(left.data, (np.ndarray, np.generic))
    assert isinstance(right.data, (np.ndarray, np.generic))
    xr.testing.assert_equal(left, exp_left)
    xr.testing.assert_equal(right, exp_right)


def test_plotting_points2():
    """
    Simple test that `Pit_fcst_at_obs().plotting_points()` returns as expected.
    Note: `.plotting_points()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_plotting_points`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).plotting_points()
    xr.testing.assert_equal(ptd.EXP_FAO_PP, result)


def test_plotting_points_parametric2():
    """
    Simple test that `Pit_fcst_at_obs().plotting_points_parametric()` returns as expected.
    Note: `.plotting_points_parametric()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_plotting_points_parametric`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).plotting_points_parametric()
    expected = ptd.EXP_FAO_PPP
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


def test_hist_values2():
    """
    Simple test that `Pit_fcst_at_obs().hist_values()` returns as expected.
    Note: `.hist_values()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_hist_values`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).hist_values(2)
    xr.testing.assert_equal(ptd.EXP_FAO_HV, result)


def test_alpha_score2():
    """
    Simple test that `Pit_fcst_at_obs().alpha_score()` returns as expected.
    Note: `.alpha_score()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_alpha_score`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).alpha_score()
    xr.testing.assert_equal(xr.DataArray(0.4**2 + 0.1**2 + 0.3**2 + 0.2**2) / 2, result)


def test_expected_value2():
    """
    Simple test that `Pit_fcst_at_obs().expected_value()` returns as expected.
    Note: `.expected_value()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_expected_value`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).expected_value()
    xr.testing.assert_allclose(xr.DataArray(0.6), result)


def test_variance2():
    """
    Simple test that `Pit_fcst_at_obs().variance()` returns as expected.
    Note: `.variance()` code is identical for both `Pit_fcst_at_obs` and `Pit` classes,
    so this test is mainly to identify copy/paste errors and is not as comprehensive
    as `test_variance`.
    """
    result = Pit_fcst_at_obs(ptd.DA_FCST_AT_OBS).variance()
    xr.testing.assert_allclose(xr.DataArray(0.04), result)
