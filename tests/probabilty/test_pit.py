"""
Unit tests for scopres.probability.pit_impl.py
"""

import pytest
import xarray as xr
from numpy import nan

from scores.probability.pit_impl import (
    _get_pit_x_values,
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
EXP_GPV = xr.DataArray(
    data=[0, 0.1, 0.5, 0.8, 1], dims=["pit_x_values"], coords={"pit_x_values": [0, 0.1, 0.5, 0.8, 1]}
)

EXP_PCVFU = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    dims=["stn", "pit_x_values"],
    coords={"stn": [101, 102, 103], "pit_x_values": [0, 0.1, 0.5, 0.8, 1]},
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
    dims=["stn", "pit_x_values"],
    coords={"stn": [101, 102, 103], "pit_x_values": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ_RIGHT = xr.DataArray(
    data=[[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_values"],
    coords={"stn": [101, 102, 103], "pit_x_values": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCVFJ = {"left": EXP_PCVFJ_LEFT, "right": EXP_PCVFJ_RIGHT}
EXP_PCVFJ2 = {
    "left": xr.merge([EXP_PCVFJ_LEFT.rename("tas"), EXP_PCVFJ_LEFT.rename("pr")]),
    "right": xr.merge([EXP_PCVFJ_RIGHT.rename("tas"), EXP_PCVFJ_RIGHT.rename("pr")]),
}

EXP_PCV_LEFT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 0, 1, 1]],
    dims=["stn", "pit_x_values"],
    coords={"stn": [101, 102, 103], "pit_x_values": [0, 0.1, 0.5, 0.8, 1]},
)
EXP_PCV_RIGHT = xr.DataArray(
    data=[[0.0, 0, 4 / 7, 1, 1], [nan, nan, nan, nan, nan], [0.0, 0, 1, 1, 1]],
    dims=["stn", "pit_x_values"],
    coords={"stn": [101, 102, 103], "pit_x_values": [0, 0.1, 0.5, 0.8, 1]},
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
            xr.DataArray(data=[0], dims=["lead_day"], coords={"lead_day": [1]}),
            xr.DataArray(data=[1], dims=["pit_x_values"], coords={"pit_x_values": [2]}),
            None,
        ),
        (
            xr.DataArray(data=[0], dims=["lead_day"], coords={"lead_day": [1]}),
            xr.DataArray(data=[1], dims=["stn"], coords={"stn": [2]}),
            xr.DataArray(data=[3], dims=["pit_x_values"], coords={"pit_x_values": [5]}),
        ),
    ],
)
def test__pit_dimension_checks_raises(fcst, obs, weights):
    """Test that `_pit_dimension_checks` raises as expected."""
    with pytest.raises(ValueError, match="'uniform_endpoint' or 'pit_x_values' are"):
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
    dims=["stn", "lead_day", "pit_x_values"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT1 = xr.DataArray(
    data=[
        [[0, 1, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],  # Unif[0, 0.4], Unif[0, 0.8]
        [[0, 0, 1, 1, 1], [nan, nan, nan, nan, nan]],  # Unif[0.6, 0.6], nan
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["stn", "lead_day", "pit_x_values"],
    coords={"stn": [101, 102, 103], "lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF1 = {"left": EXP_PITCDF_LEFT1, "right": EXP_PITCDF_RIGHT1}
# preserve lead day, weights=None
EXP_PITCDF_LEFT2 = xr.DataArray(
    data=[[0, 0.5, 0.5, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_values"],
    coords={"lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT2 = xr.DataArray(
    data=[[0, 0.5, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_values"],
    coords={"lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF2 = {"left": EXP_PITCDF_LEFT2, "right": EXP_PITCDF_RIGHT2}
# preserve lead day, station weights = [1, 2, 3]
WTS_STN = xr.DataArray(data=[1, 2, 3], dims=["stn"], coords={"stn": [101, 102, 103]})
EXP_PITCDF_LEFT3 = xr.DataArray(
    data=[[0, 1 / 3, 1 / 3, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_values"],
    coords={"lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT3 = xr.DataArray(
    data=[[0, 1 / 3, 1, 1, 1], [0, 0.5, 0.75, 1, 1]],
    dims=["lead_day", "pit_x_values"],
    coords={"lead_day": [0, 1], "pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF3 = {"left": EXP_PITCDF_LEFT3, "right": EXP_PITCDF_RIGHT3}
# reduce all dims, no weights
EXP_PITCDF_LEFT4 = xr.DataArray(
    data=[0, 0.5, 1.75 / 3, 1, 1],
    dims=["pit_x_values"],
    coords={"pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF_RIGHT4 = xr.DataArray(
    data=[0, 0.5, 2.75 / 3, 1, 1],
    dims=["pit_x_values"],
    coords={"pit_x_values": [0.0, 0.4, 0.6, 0.8, 1]},
)
EXP_PITCDF4 = {"left": EXP_PITCDF_LEFT4, "right": EXP_PITCDF_RIGHT4}
# data set example
DS_FCST = xr.merge([DA_FCST.rename("tas"), DA_FCST.rename("pr")])
DS_OBS = xr.merge([DA_OBS.rename("tas"), DA_OBS.rename("pr")])
EXP_PITCDF5 = {
    "left": xr.merge([EXP_PITCDF_LEFT4.rename("tas"), EXP_PITCDF_LEFT4.rename("pr")]),
    "right": xr.merge([EXP_PITCDF_RIGHT4.rename("tas"), EXP_PITCDF_RIGHT4.rename("pr")]),
}


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
