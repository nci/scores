"""
Contains tests for the scores.typing file
"""

import pytest

import numpy as np
import xarray as xr

import scores.typing


@pytest.mark.parametrize(
    "xrlike,check_pass",
    [
        # dataset - true
        (xr.Dataset(dict(x=xr.DataArray([1]))), True),
        # dataarray - true
        (xr.DataArray([1]), True),
        # some integer - false
        (42, False),
        # numpy - false
        (np.array([1]), False),
        # some string - false
        ("hi", False),
        # multiple data arrays - false
        ([xr.DataArray([1]), xr.DataArray([2])], False),
    ],
)
def test_is_xarraylike(xrlike, check_pass):
    assert scores.typing.is_xarraylike(xrlike) == check_pass


@pytest.mark.parametrize(
    "dims,check_pass",
    [
        # all strings - pass
        (["hi", "how", "are", "you"], True),
        # all integers - pass
        ([1, 2, 3], True),
        # single tuple - pass
        ((1, 2, 3), True),
        # multiple tuples - pass - each tuple is a hashable
        ([tuple([1]), tuple([1, 2, 3])], True),
        # nested list - fail
        ([[1], [2, 3]], False),
        # mised nested list - fail
        ([1, [2, 3]], False),
        # single string - fail
        ("hello", False),
    ],
)
def test_is_flexibledimensiontypes(dims, check_pass):
    assert scores.typing.is_flexibledimensiontypes(dims) == check_pass


@pytest.mark.parametrize(
    "list_xrlike,check_pass",
    [
        # all dataarray - pass
        ([xr.DataArray([1]), xr.DataArray([1, 2, 3])], True),
        # all dataset - pass
        ([xr.Dataset(dict(x=xr.DataArray([1]))), xr.Dataset(dict(y=xr.DataArray([2, 1])))], True),
        # singleton xarraylike - pass
        ([xr.DataArray([1])], True),
        ([xr.Dataset(dict(x=xr.DataArray([1])))], True),
        # mixed dataset and dataarray - fail
        ([xr.Dataset(dict(x=xr.DataArray([1]))), xr.DataArray([2])], False),
        # mixed dataarray and random type - fail
        ([np.array([1]), xr.DataArray([2])], False),
        # random type - fail
        ([1, 2, "hi"], False),
    ],
)
def test_all_same_xarraylike(list_xrlike, check_pass):
    assert scores.typing.all_same_xarraylike(list_xrlike) == check_pass
