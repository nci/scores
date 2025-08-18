"""
Unit tests for scopres.probability.pit_impl.py
"""

import pytest
import xarray as xr
from numpy import nan

from scores.probability.pit_impl import (
    _get_pit_x_values,
    _pit_cdfvalues_for_jumps,
    _pit_cdfvalues_for_unif,
    _pit_values_for_ensemble,
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
            [  # lower
                [0.0, 0.7, 4 / 9, nan, 1],  # lead day 1
                [0.0, 0.6, 0.4, nan, nan],  # lead day 2
            ],
            [  # upper
                [0.0, 0.8, 4 / 9, nan, 1],  # lead day 1
                [0.4, 0.8, 0.5, nan, nan],  # lead day 2
            ],
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
        (DA_GPV, EXP_GPV),  # data array input
        (DS_GPV, EXP_GPV),  # dataset input
    ],
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


@pytest.mark.parametrize(
    ("pit_values", "expected"),
    [
        (DA_GPV, EXP_PCVFJ),  # data array input
        (DS_GPV, EXP_PCVFJ2),  # dataset input
    ],
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
        (DA_GPV, EXP_PCVFU),  # data array input
        (DS_GPV, EXP_PCVFU2),  # dataset input
    ],
)
def test__pit_cdfvalues_for_unif(pit_values, expected):
    """Tests that `_pit_cdfvalues_for_unif` returns as expected."""
    result = _pit_cdfvalues_for_unif(pit_values, EXP_GPV)
    xr.testing.assert_equal(expected, result)
