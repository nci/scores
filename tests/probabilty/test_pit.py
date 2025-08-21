"""
Unit tests for scores.probability.pit_impl.py
"""

import numpy as np
import pytest
import xarray as xr
from numpy import nan

from scores.probability.pit_impl import (
    Pit_for_ensemble,
    _construct_hist_values,
    _get_pit_x_values,
    _get_plotting_points_dict,
    _pit_cdfvalues,
    _pit_cdfvalues_for_jumps,
    _pit_cdfvalues_for_unif,
    _pit_dimension_checks,
    _pit_hist_left,
    _pit_hist_right,
    _pit_values_for_ensemble,
    _value_at_pit_cdf,
    pit_cdfvalues,
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


def test__pit_values_for_ensemble():
    """Tests that `_pit_values_for_ensemble` returns as expected."""
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
    result = _pit_values_for_ensemble(fcst, obs, "member")
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
def test__pit_cdfvalues_for_jumps(pit_values, expected):
    """Tests that `_pit_cdfvalues_for_jumps` returns as expected."""
    result = _pit_cdfvalues_for_jumps(pit_values, ptd.EXP_GPV)
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
def test__pit_cdfvalues_for_unif(pit_values, expected):
    """Tests that `_pit_cdfvalues_for_unif` returns as expected."""
    result = _pit_cdfvalues_for_unif(pit_values, ptd.EXP_GPV)
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
def test__pit_dimension_checks_raises(fcst, obs, weights):
    """Test that `_pit_dimension_checks` raises as expected."""
    with pytest.raises(ValueError, match="The following names are reserved and"):
        _pit_dimension_checks(fcst, obs, weights)


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
def test_pit_cdfvalues(fcst, obs, preserve_dims, reduce_dims, weights, expected):
    """Tests that `_pit_cdfvalues` returns as expected."""
    result = pit_cdfvalues(
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
    """Tests that `Pit_for_ensemble.__init__` returns as expected."""
    result = Pit_for_ensemble(fcst, obs, "ens_member", preserve_dims=preserve_dims)
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
def test__get_plotting_points_dict(left, right, expected):
    """Tests that `_get_plotting_points_dict` returns as expected."""
    result = _get_plotting_points_dict(left, right)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


EXP_PITCDF_LEFT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],  # Unif[0, 0.4], Unif[0, 0.8]
        [[0, 0, 0, 1, 1], [nan, nan, nan, nan, nan]],  # Unif[0.6, 0.6], nan
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],  # Unif[0, 0.4], Unif[0, 0.8]
        [[0, 0, 1, 1, 1], [nan, nan, nan, nan, nan]],  # Unif[0.6, 0.6], nan
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_value"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)

EXP_PP1 = xr.DataArray(  # uses EXP_PITCDF_LEFT2, EXP_PITCDF_RIGHT2
    data=[[0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1], [0, 0, 0.5, 0.5, 0.75, 0.75, 1, 1, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0, 0, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1]},
)


@pytest.mark.parametrize(
    ("fcst", "obs", "expected"),
    [
        (ptd.DA_FCST, ptd.DA_OBS, EXP_PP1),
        (DS_FCST, DS_OBS, create_dataset(EXP_PP1)),
    ],
)
def test_plotting_points(fcst, obs, expected):
    """Tests that `Pit_for_ensemble().plotting_points()` returns as expected."""
    result = Pit_for_ensemble(fcst, obs, "ens_member", preserve_dims="lead_day").plotting_points()
    xr.testing.assert_equal(expected, result)


def test_plotting_points_parametric():
    """Tests that `Pit_for_ensemble().plotting_points_parametric()` returns as expected."""
    result = Pit_for_ensemble(ptd.DA_FCST, ptd.DA_OBS, "ens_member").plotting_points_parametric()
    expected = {
        "x_plotting_position": xr.DataArray(
            data=[0, 0.4, 0.6, 0.6, 0.8, 1], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
        ),
        "y_plotting_position": xr.DataArray(
            data=[0, 0.5, 1.75 / 3, 2.75 / 3, 1, 1],
            dims=["plotting_point"],
            coords={"plotting_point": [0, 1, 2, 3, 4, 5]},
        ),
    }
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


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
    result = Pit_for_ensemble(fcst, obs, "ens_member").hist_values(5, right=right)
    xr.testing.assert_allclose(expected, result)
