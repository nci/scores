"""
Unit tests for scopres.probability.pit_impl.py
"""

import pytest
import xarray as xr
from numpy import nan

from scores.probability.pit_impl import (
    Pit_for_ensemble,
    _get_pit_x_values,
    _get_plotting_points,
    _pit_cdfvalues,
    _pit_cdfvalues_for_jumps,
    _pit_cdfvalues_for_unif,
    _pit_dimension_checks,
    _pit_values_for_ensemble,
    pit_cdfvalues,
)


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


DA_GPV = xr.DataArray(
    data=[[0.1, nan, 0.5], [0.8, nan, 0.5]],
    dims=["uniform_endpoint", "stn"],
    coords={"uniform_endpoint": ["lower", "upper"], "stn": [101, 102, 103]},
)
DS_GPV = xr.merge([DA_GPV.rename("tas"), DA_GPV.rename("pr")])
EXP_GPV = xr.DataArray(data=[0, 0.1, 0.5, 0.8, 1], dims=["pit_x_value"], coords={"pit_x_value": [0, 0.1, 0.5, 0.8, 1]})

EXP_PCVFU = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFU2 = xr.merge([EXP_PCVFU.rename("tas"), EXP_PCVFU.rename("pr")])


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (DA_GPV, EXP_GPV),
        (DS_GPV, EXP_GPV),
    ],  # data array input  # dataset input
)
def test__get_pit_x_values(pit_values, expected):
    """Tests that `_get_pit_x_values` returns as expected."""
    result = _get_pit_x_values(pit_values)
    xr.testing.assert_equal(expected, result)


EXP_PCVFJ_LEFT = xr.DataArray(
    data=[[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [0.0, 0, 0, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ_RIGHT = xr.DataArray(
    data=[[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ = {"left": EXP_PCVFJ_LEFT, "right": EXP_PCVFJ_RIGHT}
EXP_PCVFJ2 = {
    "left": xr.merge([EXP_PCVFJ_LEFT.rename("tas"), EXP_PCVFJ_LEFT.rename("pr")]),
    "right": xr.merge([EXP_PCVFJ_RIGHT.rename("tas"), EXP_PCVFJ_RIGHT.rename("pr")]),
}

EXP_PCV_LEFT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 0, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCV_RIGHT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_value"],
    coords={"stn": [101, 102, 103], "pit_x_value": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCV = {"left": EXP_PCV_LEFT, "right": EXP_PCV_RIGHT}
EXP_PCV2 = {
    "left": xr.merge([EXP_PCV_LEFT.rename("tas"), EXP_PCV_LEFT.rename("pr")]),
    "right": xr.merge([EXP_PCV_RIGHT.rename("tas"), EXP_PCV_RIGHT.rename("pr")]),
}


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (DA_GPV, EXP_PCVFJ),
        (DS_GPV, EXP_PCVFJ2),
    ],  # data array input  # dataset input
)
def test__pit_cdfvalues_for_jumps(pit_values, expected):
    """Tests that `_pit_cdfvalues_for_jumps` returns as expected."""
    result = _pit_cdfvalues_for_jumps(pit_values, EXP_GPV)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (DA_GPV, EXP_PCVFU),
        (DS_GPV, EXP_PCVFU2),
    ],  # data array input  # dataset input
)
def test__pit_cdfvalues_for_unif(pit_values, expected):
    """Tests that `_pit_cdfvalues_for_unif` returns as expected."""
    result = _pit_cdfvalues_for_unif(pit_values, EXP_GPV)
    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (DA_GPV, EXP_PCV),
        (DS_GPV, EXP_PCV2),
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
    ],
)
def test__pit_dimension_checks_raises(fcst, obs, weights):
    """Test that `_pit_dimension_checks` raises as expected."""
    with pytest.raises(ValueError, match="are reserved and should not be among"):
        _pit_dimension_checks(fcst, obs, weights)


DA_FCST = xr.DataArray(
    data=[
        [[0.0, 0, 4, 2, 1], [0, 0, 0, 0, 1]],
        [[5, 3, 7, 2, 1], [nan, nan, nan, nan, nan]],
        [[2, 2, 5, 1, 2], [3, 2, 1, 4, 0]],
    ],
    dims=["stn", "lead_day", "ens_member"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "ens_member": [0, 1, 2, 3, 4]},
)
DA_OBS = xr.DataArray(data=[0, 4, nan], dims=["stn"], coords={"stn": [101, 102, 103]})
# keep all dims
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
EXP_PITCDF1 = {"left": EXP_PITCDF_LEFT1, "right": EXP_PITCDF_RIGHT1}
# preserve lead day, weights=None
EXP_PITCDF_LEFT2 = xr.DataArray(
    data=[[0, 0.5, 0.5, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT2 = xr.DataArray(
    data=[[0, 0.5, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF2 = {"left": EXP_PITCDF_LEFT2, "right": EXP_PITCDF_RIGHT2}
# preserve lead day, station weights = [1, 2, 3]
WTS_STN = xr.DataArray(data=[1, 2, 3], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_PITCDF_LEFT3 = xr.DataArray(
    data=[[0, 1 / 3, 1 / 3, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT3 = xr.DataArray(
    data=[[0, 1 / 3, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_value"],
    coords={"lead_day": [0, 1], "pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF3 = {"left": EXP_PITCDF_LEFT3, "right": EXP_PITCDF_RIGHT3}
# reduce all dims, no weights
EXP_PITCDF_LEFT4 = xr.DataArray(
    data=[0, 0.5, 1.75 / 3, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT4 = xr.DataArray(
    data=[0, 0.5, 2.75 / 3, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF4 = {"left": EXP_PITCDF_LEFT4, "right": EXP_PITCDF_RIGHT4}
# data set example
DS_FCST = xr.merge([DA_FCST.rename("tas"), DA_FCST.rename("pr")])
DS_OBS = xr.merge([DA_OBS.rename("tas"), DA_OBS.rename("pr")])
EXP_PITCDF_LEFT5 = xr.merge([EXP_PITCDF_LEFT4.rename("tas"), EXP_PITCDF_LEFT4.rename("pr")])
EXP_PITCDF_RIGHT5 = xr.merge([EXP_PITCDF_RIGHT4.rename("tas"), EXP_PITCDF_RIGHT4.rename("pr")])
EXP_PITCDF5 = {"left": EXP_PITCDF_LEFT5, "right": EXP_PITCDF_RIGHT5}


@pytest.mark.parametrize(
    ("fcst", "obs", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        (DA_FCST, DA_OBS, "all", None, None, EXP_PITCDF1),
        (DA_FCST, DA_OBS, "lead_day", None, None, EXP_PITCDF2),
        (DA_FCST, DA_OBS, None, "stn", None, EXP_PITCDF2),
        (DA_FCST, DA_OBS, "lead_day", None, WTS_STN, EXP_PITCDF3),
        (DA_FCST, DA_OBS, None, "stn", WTS_STN, EXP_PITCDF3),
        (DA_FCST, DA_OBS, None, "all", None, EXP_PITCDF4),
        (DS_FCST, DS_OBS, None, "all", None, EXP_PITCDF5),  # data set example
        (DS_FCST, DA_OBS, None, "all", None, EXP_PITCDF5),  # data set/array mix
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
        (DA_FCST, DA_OBS, "all", EXP_PITCDF_LEFT1, EXP_PITCDF_RIGHT1),
        (DA_FCST, DA_OBS, None, EXP_PITCDF_LEFT4, EXP_PITCDF_RIGHT4),
        (DS_FCST, DS_OBS, None, EXP_PITCDF_LEFT5, EXP_PITCDF_RIGHT5),  # data set example
    ],
)
def test___init__(fcst, obs, preserve_dims, expected_left, expected_right):
    """Tests that `Pit_for_ensemble.__init__` returns as expected."""
    result = Pit_for_ensemble(fcst, obs, "ens_member", preserve_dims=preserve_dims)
    xr.testing.assert_equal(result.left, expected_left)
    xr.testing.assert_equal(result.right, expected_right)


EXP_PLOTTING_POINTS2 = xr.DataArray(
    data=[0, 0, 0.5, 0.5, 1.75 / 3, 2.75 / 3, 1, 1, 1, 1],
    dims=["pit_x_value"],
    coords={"pit_x_value": [0, 0, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1, 1]},
)

# @pytest.mark.parametrize(
#     ("fcst", "obs", "preserve_dims", "expected"),
#     [
#         # (DA_FCST, DA_OBS, "all", ),
#         (DA_FCST, DA_OBS, None, EXP_PLOTTING_POINTS2),
#         # (DS_FCST, DS_OBS, None, ),  # data set example
#     ],
# )
# def test_plotting_points(fcst, obs, preserve_dims, expected):
#     """Tests that `Pit_for_ensemble.plotting_points` returns as expected."""
#     result = Pit_for_ensemble(fcst, obs, "ens_member", preserve_dims=preserve_dims).plotting_points()
#     xr.testing.assert_equal(result, expected)

# case with several dimensions; left and right equal when pit_x_value is 0 or 3, NaNs involved
DA_GPP_LEFT1 = xr.DataArray(
    data=[[[1, 2, 2, 5], [0, 0, 3, 6], [1, 1, 2, 3]], [[1, 2, 2, 5], [0, 0, 3, 6], [nan, nan, nan, nan]]],
    dims=["lead_day", "stn", "pit_x_value"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "pit_x_value": [0, 1, 2, 3]},
)
DA_GPP_RIGHT1 = xr.DataArray(
    data=[[[1, 2, 3, 5], [0, 1, 3, 6], [1, 1, 2, 3]], [[1, 2, 3, 5], [0, 1, 3, 6], [nan, nan, nan, nan]]],
    dims=["lead_day", "stn", "pit_x_value"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "pit_x_value": [0, 1, 2, 3]},
)
EXP_GPP_X1 = xr.DataArray(
    data=[0, 1, 1, 2, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP_Y1 = xr.DataArray(
    data=[
        [[1, 2, 2, 2, 3, 5], [0, 0, 1, 3, 3, 6], [1, 1, 1, 2, 2, 3]],
        [[1, 2, 2, 2, 3, 5], [0, 0, 1, 3, 3, 6], [nan, nan, nan, nan, nan, nan]],
    ],
    dims=["lead_day", "stn", "plotting_point"],
    coords={"lead_day": [0, 1], "stn": [101, 102, 103], "plotting_point": [0, 1, 2, 3, 4, 5]},
)
EXP_GPP1 = {"x_plotting_position": EXP_GPP_X1, "y_plotting_position": EXP_GPP_Y1}
# case with only one dimension; left and right equal when pit_x_value is 2 or 3
DA_GPP_LEFT2 = xr.DataArray(data=[1, 2, 2, 5], dims=["pit_x_value"], coords={"pit_x_value": [0, 1, 2, 3]})
DA_GPP_RIGHT2 = xr.DataArray(data=[2, 2, 3, 5], dims=["pit_x_value"], coords={"pit_x_value": [0, 1, 2, 3]})
EXP_GPP_X2 = xr.DataArray(
    data=[0, 0, 1, 2, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP_Y2 = xr.DataArray(
    data=[1, 2, 2, 2, 3, 5], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3, 4, 5]}
)
EXP_GPP2 = {"x_plotting_position": EXP_GPP_X2, "y_plotting_position": EXP_GPP_Y2}
# when left and right always equal
EXP_GPP_X3 = xr.DataArray(data=[0, 1, 2, 3], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3]})
EXP_GPP_Y3 = xr.DataArray(data=[1, 2, 2, 5], dims=["plotting_point"], coords={"plotting_point": [0, 1, 2, 3]})
EXP_GPP3 = {"x_plotting_position": EXP_GPP_X3, "y_plotting_position": EXP_GPP_Y3}
# dataset
DS_GPP_LEFT4 = xr.merge([DA_GPP_LEFT2.rename("tas"), DA_GPP_LEFT2.rename("pr")])
DS_GPP_RIGHT4 = xr.merge([DA_GPP_RIGHT2.rename("tas"), DA_GPP_RIGHT2.rename("pr")])
EXP_GPP_X4 = xr.merge([EXP_GPP_X2.rename("tas"), EXP_GPP_X2.rename("pr")])
EXP_GPP_Y4 = xr.merge([EXP_GPP_Y2.rename("tas"), EXP_GPP_Y2.rename("pr")])
EXP_GPP4 = {"x_plotting_position": EXP_GPP_X4, "y_plotting_position": EXP_GPP_Y4}


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (DA_GPP_LEFT1, DA_GPP_RIGHT1, EXP_GPP1),  # several dimensions, NaN handling
        (DA_GPP_LEFT2, DA_GPP_RIGHT2, EXP_GPP2),  # one dimension
        (DA_GPP_LEFT2, DA_GPP_LEFT2, EXP_GPP3),  # left always equals right
        (DS_GPP_LEFT4, DS_GPP_RIGHT4, EXP_GPP4),  # datasets
    ],
)
def test__get_plotting_points(left, right, expected):
    """Tests that `_get_plotting_points` returns as expected."""
    result = _get_plotting_points(left, right)
    assert expected.keys() == result.keys()
    for key in result.keys():
        xr.testing.assert_equal(expected[key], result[key])


def test_plotting_points():
    """Tests that `Pit_for_ensemble().plotting_points()` returns as expected."""
    result = Pit_for_ensemble(DA_FCST, DA_OBS, "ens_member").plotting_points()
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
