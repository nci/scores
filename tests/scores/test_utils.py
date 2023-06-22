"""
Containts tests for the scores.utils file
"""

import pytest
import utils_test_data  # pylint: disable=import-error

from scores import utils
from scores.utils import DimensionError
from scores.utils import gather_dimensions as gd


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


def test_gather_dimensions():
    """
    Test the logic for dimension handling with some examples
    """

    fcst_dims = set(["base_time", "lead_time", "lat", "lon"])
    obs_dims = []
    weights_dims = []

    assert gd(fcst_dims, obs_dims, weights_dims, preserve_dims="lat") == set(["base_time", "lead_time", "lon"])
    assert gd(fcst_dims, obs_dims, weights_dims, preserve_dims=["lat", "lon"]) == set(["base_time", "lead_time"])

    preserve_all = gd(fcst_dims, obs_dims, weights_dims, preserve_dims="all")
    reduce_empty = gd(fcst_dims, obs_dims, weights_dims, reduce_dims=[])

    assert preserve_all == reduce_empty
    assert preserve_all == []
