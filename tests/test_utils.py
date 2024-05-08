"""
Contains tests for the scores.utils file
"""

import numpy as np
import pytest
import xarray as xr

from scores import utils
from scores.utils import DimensionError, check_binary
from scores.utils import gather_dimensions as gd
from tests import utils_test_data


def test_dims_complement():
    """Test `dims_complement` returns as expected."""
    xr_data = utils_test_data.DA_RGB
    expected_dims = sorted(["red", "green", "blue"])
    complement = utils.dims_complement(xr_data)
    assert complement == expected_dims

    expected_dims = sorted(["red", "green"])
    complement = utils.dims_complement(xr_data, dims=["blue"])
    assert complement == expected_dims


@pytest.mark.parametrize(
    ("xr_data", "expected_dims", "mode"),
    [
        # 1-D DataArrays
        (utils_test_data.DA_R, ["red"], "equal"),
        (utils_test_data.DA_R, ["red"], None),
        (utils_test_data.DA_R, ["red"], "subset"),
        (utils_test_data.DA_R, ["red"], "superset"),
        (utils_test_data.DA_R, ["green"], "disjoint"),
        # 2-D DataArrays
        (utils_test_data.DA_RG, ["red", "green"], "equal"),
        (utils_test_data.DA_RG, ["red", "green"], None),
        (utils_test_data.DA_RG, ["red", "green", "blue"], "subset"),
        (utils_test_data.DA_RG, ["red", "green", "blue"], "proper subset"),
        (utils_test_data.DA_RG, ["red"], "superset"),
        (utils_test_data.DA_RG, ["red"], "proper superset"),
        (utils_test_data.DA_RG, ["red", "green"], "superset"),
        # 1-D Datasets
        (utils_test_data.DS_R, ["red"], "equal"),
        (utils_test_data.DS_R, ["red"], None),
        (utils_test_data.DS_R, ["red"], "subset"),
        (utils_test_data.DS_R, ["red"], "superset"),
        (utils_test_data.DS_R, ["green"], "disjoint"),
        # 2-D Datasets
        (utils_test_data.DS_RG, ["red", "green"], "equal"),
        (utils_test_data.DS_RG, ["red", "green"], None),
        (utils_test_data.DS_RG, ["red", "green", "blue"], "subset"),
        (utils_test_data.DS_RG, ["red", "green", "blue"], "proper subset"),
        (utils_test_data.DS_RG, ["red"], "superset"),
        (utils_test_data.DA_RG, ["red"], "proper superset"),
        (utils_test_data.DS_RG, ["red", "green"], "superset"),
        # Datasets with mutiple data variables
        (utils_test_data.DS_RG_R, ["red", "green"], "subset"),
        (utils_test_data.DS_RG_R, ["red"], "superset"),
        (utils_test_data.DS_RG_RG, ["red", "green"], None),
        (utils_test_data.DS_RG_RG, ["red", "green"], "subset"),
        (utils_test_data.DS_RG_RG, ["red", "green"], "equal"),
        (utils_test_data.DS_RG_RG, ["red", "green", "blue"], "subset"),
        (utils_test_data.DS_RG_RG, ["red"], "superset"),
        (utils_test_data.DS_RGB_GB, ["green", "blue"], "superset"),
        # issue #162 - dims accepts any iterable
        (utils_test_data.DS_RGB_GB, {"green", "blue"}, "superset"),
    ],
)
def test_check_dims(xr_data, expected_dims, mode):
    """
    Tests that check_dims passes when expected to
    """
    utils.check_dims(xr_data, expected_dims, mode=mode)


@pytest.mark.parametrize(
    ("xr_data", "expected_dims", "mode", "error_class", "error_msg_snippet"),
    [
        # check 'equal'
        (
            utils_test_data.DA_R,
            ["green"],
            "equal",
            DimensionError,
            "equal to the dimensions ['green'].",
        ),
        (
            utils_test_data.DA_RG,
            ["blue"],
            "equal",
            DimensionError,
            "equal to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_R,
            ["blue"],
            "equal",
            DimensionError,
            "equal to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "equal",
            DimensionError,
            "equal to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RG_R,
            ["red", "green"],
            "equal",
            DimensionError,
            "['red'] of data variable 'DA_R' are not equal to the dimensions ['green', 'red']",
        ),
        (
            utils_test_data.DS_RG_R,
            ["red", "green"],
            None,
            DimensionError,
            "['red'] of data variable 'DA_R' are not equal to the dimensions ['green', 'red']",
        ),
        (
            utils_test_data.DS_RGB_GB,
            ["red", "green", "blue"],
            "equal",
            DimensionError,
            "['green', 'blue'] of data variable 'DA_GB' are not equal to the dimensions ['blue', 'green', 'red']",
        ),
        # check 'subset'
        (
            utils_test_data.DA_R,
            ["green"],
            "subset",
            DimensionError,
            "subset to the dimensions ['green'].",
        ),
        (
            utils_test_data.DA_RG,
            ["blue"],
            "subset",
            DimensionError,
            "subset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_R,
            ["blue"],
            "subset",
            DimensionError,
            "subset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "subset",
            DimensionError,
            "subset to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RGB,
            ["red", "green"],
            "subset",
            DimensionError,
            "subset to the dimensions ['green', 'red'].",
        ),
        # check 'superset'
        (
            utils_test_data.DA_R,
            ["green"],
            "superset",
            DimensionError,
            "superset to the dimensions ['green'].",
        ),
        (
            utils_test_data.DA_RG,
            ["blue", "red", "green"],
            "superset",
            DimensionError,
            "superset to the dimensions ['blue', 'green', 'red'].",
        ),
        (
            utils_test_data.DS_R,
            ["blue"],
            "superset",
            DimensionError,
            "superset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "superset",
            DimensionError,
            "superset to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RGB,
            ["red", "green", "blue", "pink"],
            "superset",
            DimensionError,
            "superset to the dimensions ['blue', 'green', 'pink', 'red'].",
        ),
        (
            utils_test_data.DS_RG_R,
            ["red", "green"],
            "superset",
            DimensionError,
            "['red'] of data variable 'DA_R' are not superset to the dimensions ['green', 'red']",
        ),
        # check 'proper subset'
        # these are the same as subset tests
        (
            utils_test_data.DA_R,
            ["green"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['green'].",
        ),
        (
            utils_test_data.DA_RG,
            ["blue"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_R,
            ["blue"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RGB,
            ["red", "green"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['green', 'red'].",
        ),
        # specifically for proper subset
        (
            utils_test_data.DS_RGB,
            ["red", "green", "blue"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['blue', 'green', 'red'].",
        ),
        (
            utils_test_data.DA_R,
            ["red"],
            "proper subset",
            DimensionError,
            "proper subset to the dimensions ['red'].",
        ),
        # check 'proper superset'
        # these are the same as superset tests
        (
            utils_test_data.DA_R,
            ["green"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['green'].",
        ),
        (
            utils_test_data.DA_RG,
            ["blue", "red", "green"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['blue', 'green', 'red'].",
        ),
        (
            utils_test_data.DS_R,
            ["blue"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['blue'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RGB,
            ["red", "green", "blue", "pink"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['blue', 'green', 'pink', 'red'].",
        ),
        (
            utils_test_data.DS_RG_R,
            ["red", "green"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['green', 'red'].",
        ),
        # specifically for proper superset
        (
            utils_test_data.DA_R,
            ["red"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['red'].",
        ),
        (
            utils_test_data.DS_RG_R,
            ["red"],
            "proper superset",
            DimensionError,
            "['red'] of data variable 'DA_R' are not proper superset to the dimensions ['red']",
        ),
        (
            utils_test_data.DS_RGB,
            ["red", "green", "blue"],
            "proper superset",
            DimensionError,
            "superset to the dimensions ['blue', 'green', 'red'].",
        ),
        (
            utils_test_data.DS_RGB_GB,
            ["green", "blue"],
            "proper superset",
            DimensionError,
            "['green', 'blue'] of data variable 'DA_GB' are not proper superset to the dimensions ['blue', 'green']",
        ),
        # check 'disjoint'
        (
            utils_test_data.DA_R,
            ["red"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['red'].",
        ),
        (
            utils_test_data.DA_R,
            ["red", "green"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['green', 'red'].",
        ),
        (
            utils_test_data.DS_R,
            ["red"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['red'].",
        ),
        (
            utils_test_data.DS_R,
            ["red", "green"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['green', 'red'].",
        ),
        (
            utils_test_data.DA_RG,
            ["red", "blue"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RG,
            ["red", "blue"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['blue', 'red'].",
        ),
        (
            utils_test_data.DS_RG_R,
            ["green"],
            "disjoint",
            DimensionError,
            "disjoint to the dimensions ['green'].",
        ),
        # check the modes
        (utils_test_data.DA_R, ["red"], "frog", ValueError, "No such mode frog,"),
        # check if a non data object is passed
        (
            [5],
            ["red"],
            "equal",
            DimensionError,
            "Supplied object has no dimensions",
        ),
        # duplicate values in dims
        (
            utils_test_data.DA_R,
            ["red", "blue", "red"],
            "equal",
            ValueError,
            "Supplied dimensions ['red', 'blue', 'red'] contains duplicate values.",
        ),
        # can't convert into a set
        (
            utils_test_data.DA_R,
            [["red", "blue", "red"]],
            "subset",
            ValueError,
            "Cannot convert supplied dims [['red', 'blue', 'red']] into a set. ",
        ),
        # if a string is passed
        (
            utils_test_data.DA_R,
            "red",
            "equal",
            TypeError,
            "'red' must be an iterable of strings",
        ),
    ],
)
def test_check_dims_raises(xr_data, expected_dims, mode, error_class, error_msg_snippet):
    """
    Tests that check_dims correctly raises the correct error
    """

    with pytest.raises(error_class) as excinfo:
        utils.check_dims(xr_data, expected_dims, mode=mode)
    assert error_msg_snippet in str(excinfo.value)


def test_gather_dimensions_examples():
    """
    Test the logic for dimension handling with some examples
    """

    fcst_dims_conflict = set(["base_time", "lead_time", "lat", "lon", "all"])
    fcst_dims = set(["base_time", "lead_time", "lat", "lon"])
    obs_dims = []

    # Basic tests on reduction
    assert gd(fcst_dims, obs_dims, reduce_dims="lat") == set(["lat"])
    assert gd(fcst_dims, obs_dims, reduce_dims=["lat", "lon"]) == set(["lat", "lon"])
    assert gd(fcst_dims, obs_dims, reduce_dims=["lat", "lat", "lon"]) == set(["lat", "lon"])

    # Tests if reduce_dims and preserve_dims are both None
    assert gd(fcst_dims, obs_dims) == fcst_dims

    # Reduce every dimension if the string "all" is specified
    assert gd(fcst_dims, obs_dims, reduce_dims="all") == fcst_dims

    # Reduce "all" as a named dimension explicitly
    assert gd(fcst_dims_conflict, obs_dims, reduce_dims=["all"]) == set(["all"])

    # Basic tests on preservation
    assert gd(fcst_dims, obs_dims, preserve_dims="lat") == set(["base_time", "lead_time", "lon"])
    assert gd(fcst_dims, obs_dims, preserve_dims=["lat", "lon"]) == set(["base_time", "lead_time"])
    assert gd(fcst_dims, obs_dims, preserve_dims=["lat", "lat", "lon"]) == set(["base_time", "lead_time"])

    # Preserve every dimension if the string "all" is specified
    assert gd(fcst_dims, obs_dims, preserve_dims="all") == set([])

    # Preserve "all" as a named dimension explicitly
    assert gd(fcst_dims_conflict, obs_dims, preserve_dims=["all"]) == set(["base_time", "lead_time", "lat", "lon"])

    # Test that preserve is the inverse of reduce
    preserve_all = gd(fcst_dims, obs_dims, preserve_dims="all")
    reduce_empty = gd(fcst_dims, obs_dims, reduce_dims=[])

    assert preserve_all == reduce_empty
    assert preserve_all == set([])

    # Single dimensions specified as a string will be packed into a list
    assert gd(fcst_dims, obs_dims, reduce_dims="lead_time") == set(["lead_time"])


def test_gather_dimensions_exceptions():
    """
    Confirm an exception is raised when both preserve and reduce arguments are specified
    """

    fcst_dims_conflict = set(["base_time", "lead_time", "lat", "lon", "all"])
    fcst_dims = set(["base_time", "lead_time", "lat", "lon"])
    obs_dims = []

    # Confirm an exception if both preserve and reduce are specified
    with pytest.raises(ValueError):
        gd(fcst_dims, obs_dims, preserve_dims=[], reduce_dims=[])

    # Attempt to reduce a non-existent dimension
    with pytest.raises(ValueError) as excinfo:
        assert not gd(fcst_dims_conflict, obs_dims, reduce_dims="nonexistent")
    assert str(excinfo.value.args[0]) == utils.ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION

    # Attempt to preserve a non-existent dimension
    with pytest.raises(ValueError) as excinfo:
        assert not gd(fcst_dims_conflict, obs_dims, preserve_dims="nonexistent")
    assert str(excinfo.value.args[0]) == utils.ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION

    # Preserve "all" as a string but named dimension present in data
    with pytest.warns(UserWarning):
        assert gd(fcst_dims_conflict, obs_dims, preserve_dims="all") == set([])

    # Preserve "all" as a string but named dimension present in data
    with pytest.warns(UserWarning):
        assert gd(fcst_dims_conflict, obs_dims, reduce_dims="all") == fcst_dims_conflict


@pytest.mark.parametrize(
    (
        "fcst_dims",
        "obs_dims",
        "weights_dims",
        "reduce_dims",
        "preserve_dims",
        "score_specific_fcst_dims",
        "error_msg_snippet",
    ),
    [
        # checks for score_specific_fcst_dims
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            None,
            None,
            None,
            ["black"],
            "`score_specific_fcst_dims` must be a subset of `fcst` dimensions",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            None,
            None,
            None,
            ["red"],
            "`obs.dims` must not contain any `score_specific_fcst_dims`",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            None,
            None,
            None,
            "red",
            "`obs.dims` must not contain any `score_specific_fcst_dims`",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            None,
            None,
            "green",
            "`weights.dims` must not contain any `score_specific_fcst_dims`",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            "blue",
            None,
            "blue",
            "`reduce_dims` and `preserve_dims` must not contain any `score_specific_fcst_dims`",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            None,
            ["blue"],
            "blue",
            "`reduce_dims` and `preserve_dims` must not contain any `score_specific_fcst_dims`",
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            "yellow",
            None,
            "blue",
            utils.ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION,
        ),
    ],
)
def test_gather_dimensions_score_specific_fcst_dims_exceptions(
    fcst_dims, obs_dims, weights_dims, reduce_dims, preserve_dims, score_specific_fcst_dims, error_msg_snippet
):
    """
    Confirm `gather_dimensions` raises exceptions as expected.
    """

    assert score_specific_fcst_dims is not None

    with pytest.raises(ValueError) as excinfo:
        gd(
            fcst_dims,
            obs_dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
            score_specific_fcst_dims=score_specific_fcst_dims,
        )

    assert error_msg_snippet in str(excinfo.value)


@pytest.mark.parametrize(
    ("fcst_dims", "obs_dims", "weights_dims", "reduce_dims", "preserve_dims", "score_specific_fcst_dims", "expected"),
    [

        (
            utils_test_data.DA_B.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            None,
            None, 
            "blue",
            {"red", "green"},
        ),
        (
            utils_test_data.DA_B.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            None,
            None,
            ["blue"],
            {"red", "green"},
        ),
        (
            utils_test_data.DA_RGB.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            ["green"],
            None,
            "blue",
            {"green"},
        ),
        # test for preserve_dims="all"
        (utils_test_data.DA_RGB.dims, utils_test_data.DA_B.dims, None, None, "all", "red", set([])),
        # three tests for preserve_dims
        (utils_test_data.DA_RGB.dims, utils_test_data.DA_B.dims, None, None, ["green"], "red", {"blue"}),
    ],
)
def test_gather_dimensions_score_specific_fcst_dims_examples(
    fcst_dims, obs_dims, weights_dims, reduce_dims, preserve_dims, score_specific_fcst_dims, expected
):
    """
    Test that `gather_dimensions` gives outputs as expected.
    """
    assert score_specific_fcst_dims is not None

    result = gd(
        fcst_dims,
        obs_dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=score_specific_fcst_dims,
    )
    assert result == expected


def test_tmp_coord_name_namecollision():
    """
    Confirm that asking for multiple names will result in unique names
    """
    names = []
    number_of_names = 3
    data = xr.DataArray(data=[1, 2, 3])
    names = utils.tmp_coord_name(data, count=number_of_names)
    assert len(set(names)) == len(names)
    assert len(names) == number_of_names


def test_tmp_coord_name():
    """
    Tests that `tmp_coord_name` returns as expected.
    """
    data = xr.DataArray(data=[1, 2, 3])
    assert utils.tmp_coord_name(data) == "newdim_0"

    data = xr.DataArray(data=[1, 2, 3], dims=["stn"], coords={"stn": [101, 202, 304]})
    assert utils.tmp_coord_name(data) == "newstnstn"

    data = xr.DataArray(data=[1, 2, 3], dims=["stn"], coords={"stn": [101, 202, 304], "elevation": ("stn", [0, 3, 24])})
    assert utils.tmp_coord_name(data) == "newstnstnelevation"


@pytest.mark.parametrize(
    ("da"),
    [
        (xr.DataArray([0, 1, 2])),
        (xr.DataArray([0, 1, -1])),
        (xr.DataArray([0, 1, 0.5])),
        (xr.DataArray([[0, 1, 1.0000001], [0, 1, 1]])),
    ],
)
def test_check_binary_raises(da):
    """test check_binary raises"""
    with pytest.raises(ValueError) as exc:
        check_binary(da, "my name")
    assert "`my name` contains values that are not in the set {0, 1, np.nan}" in str(exc.value)


@pytest.mark.parametrize(
    ("da"),
    [
        (xr.DataArray([0, 1])),
        (xr.DataArray([0, 0])),
        (xr.DataArray([1, 1])),
        (xr.DataArray([0, 1, np.nan])),
        (xr.DataArray([[0, 1, np.nan], [0, 1, np.nan]])),
    ],
)
def test_check_binary_doesnt_raise(da):
    """test check_binary doesn't raise"""
    check_binary(da, "my name")
