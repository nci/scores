"""
Contains tests for the scores.utils file
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from scores import utils
from scores.typing import LiftedDataset, XarrayLike, XarrayTypeMarker
from scores.utils import DimensionError, check_binary, check_weights
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


@pytest.mark.parametrize(
    ("fcst_dims", "obs_dims", "weights_dims", "reduce_dims", "preserve_dims", "score_specific_fcst_dims", "expected"),
    [
        (
            utils_test_data.DA_B.dims,
            utils_test_data.DA_R.dims,
            utils_test_data.DA_G.dims,
            "all",
            None,
            None,
            {"blue", "red", "green"},
        ),
        (utils_test_data.DA_B.dims, utils_test_data.DA_R.dims, utils_test_data.DA_G.dims, None, "all", None, set([])),
    ],
)
def test_gather_dimensions_weights_no_score_specific_examples(
    fcst_dims, obs_dims, weights_dims, reduce_dims, preserve_dims, score_specific_fcst_dims, expected
):
    """
    Test that `gather_dimensions` gives outputs as expected.
    """

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


def test_invalid_numpy_operator():
    """
    Test exception is raised for specifying an invalid operator

    """
    with pytest.raises(ValueError):
        utils.NumpyThresholdOperator(sorted)


def test_check_weights():
    """
    Tests :py:func:`scores.utils.check_weights`.

    Cases:
    - conformant weights - at least one positive, rest can be >= 0
    - any one weight negative - should raise warning
    - all NaNs - should raise warning
    - all zeros - should raise warning
    - implicitly test "context" by setting appropriate context message for the above cases
    """

    def _check_weights_assert_no_warning(_w):
        threw_warning = False
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            try:
                _w = xr.DataArray(_w)
                check_weights(_w, raise_error=False)
            except Warning:
                threw_warning = True
        assert not threw_warning

    def _expect_warning(_w):
        with pytest.warns(UserWarning):
            _w = xr.DataArray(_w)
            check_weights(_w, raise_error=False)

    def _expect_error(_w):
        with pytest.raises(ValueError):
            _w = xr.DataArray(_w)
            check_weights(_w, raise_error=True)

    _check_weights_assert_no_warning(np.array([1.0, 0.0, 0.1]))
    # catching np.inf is not the responsibility of this function, ...
    _check_weights_assert_no_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, np.inf]]))
    # ... neither is np.nan - which is ignored.
    _check_weights_assert_no_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, np.nan]]))

    # --- the rest of these should throw warnings (or error depending on raise_error) ---
    # However inputs with all nans is invalid, ...
    _expect_warning(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
    _expect_error(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
    # ... and negative infinity is not allowed.
    _expect_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, -np.inf]]))
    _expect_error(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, -np.inf]]))
    # In fact any negative number is should fail the check, ...
    _expect_warning(np.array([-1e-7, 0.0, 0.0, 1.0]))
    _expect_error(np.array([-1e-7, 0.0, 0.0, 1.0]))
    # ... even if all of them are negative, and theoretically can be flipped to be conformant,
    # it is not the responsibility of this function to do so.
    _expect_warning(np.array([-1.0, -0.1]))
    _expect_error(np.array([-1.0, -0.1]))

    # --- weights=None should do nothing ---
    assert check_weights(None) is None


# ---
# tests for LiftedDatasetUtils = ldsutils (short form)
_DS_TEST = xr.Dataset({"a": xr.DataArray([1], dims="t"), "b": xr.DataArray([2], dims="t")})


@pytest.mark.parametrize(
    "xr_data, do_lift, expect_error_raised, expect_xr_type",
    [
        # one dataset and one data array => should produce error
        (tuple([xr.DataArray([1], dims="a"), _DS_TEST]), True, True, "invalid"),
        # multiple inputs with data arrays => all data arrays => no error
        (tuple([xr.DataArray([1], dims="a"), xr.DataArray([2], dims="b")]), True, False, "dataarray"),
        # multiple inputs with data arrays => all data arrays => no error
        (tuple([xr.DataArray([1], dims="a"), xr.DataArray([2], dims="b")]), False, True, "dataarray"),
        # single input with dataset => all datasets => no error
        (tuple([_DS_TEST]), True, False, "dataset"),
        # intentionally don't lift the dataset to prompt a type error => should produce error
        (tuple([_DS_TEST]), False, True, "dataset"),
    ],
)
def test_ldsutils_allsametype(xr_data, do_lift, expect_error_raised, expect_xr_type):
    """
    Checks for homogeneity in LiftedDataset (lds) types, and checks that it raises an error if they
    are of different types.
    """
    lds = [LiftedDataset(x) for x in xr_data] if do_lift else xr_data
    if expect_error_raised:
        with pytest.raises(TypeError):
            utils.LiftedDatasetUtils.all_same_type(*lds)
    else:
        ret = utils.LiftedDatasetUtils.all_same_type(*lds)
        if expect_xr_type == "dataarray":
            assert ret == XarrayTypeMarker.DATAARRAY
        else:  # dataset
            assert ret == XarrayTypeMarker.DATASET


def test_ldsutils_lift_fn():
    # pylint: disable=use-dict-literal, too-many-locals
    # most official xarray examples use `.loc[dict(...)]` for slicing
    """
    Checks lift functions: ``LiftedDatasetUtils.lift_fn`` and ``LiftedDatasetUtils.lift_fn_ret``
        - ``lift_fn`` only lifts the fn args
        - ``lift_fn_ret`` lifts both the args and the return type
    """

    # --- helper dummy functions
    # test function that works on `XarrayLike`. Output can be anything - arbitrarily chosen string
    def _test_func(x: XarrayLike, y: XarrayLike, z: int, *, a: XarrayLike, b: XarrayLike, c: int) -> XarrayLike:
        return (x * z + y) - (a + b * c)

    _lifted_test_func = utils.LiftedDatasetUtils.lift_fn(_test_func)
    _lifted_test_func_ret = utils.LiftedDatasetUtils.lift_fn_ret(_test_func)

    # --- prepare dataarray testcase
    test_da_x: xr.DataArray = xr.DataArray([1, 2], dims=["x"])
    test_da_y: xr.DataArray = xr.DataArray([3, 4], dims=["x"])
    test_da_a: xr.DataArray = xr.DataArray([5, 6], dims=["x"])
    test_da_b: xr.DataArray = xr.DataArray([7, 8], dims=["x"])
    scalar_z: int = 20
    scalar_c: int = 4
    da_test = _test_func(test_da_x, test_da_y, scalar_z, a=test_da_a, b=test_da_b, c=scalar_c)

    # this is not the actual test - just a safety check
    # elem 1: 20 + 3 - 5 - 28 = -10
    # elem 2: 40 + 4 - 6 - 32 =   6
    da_expected = xr.DataArray([-10, 6], dims=["x"])
    assert da_expected.identical(da_test)

    # --- FUNCTION LIFTING TESTS FOR DATAARRAY --> START
    # actual tests - result is still a data array, not a lifted data array
    # lift_fn: return type should be preserved, result should match expected
    lds_da_x, lds_da_y, lds_da_a, lds_da_b = map(LiftedDataset, [test_da_x, test_da_y, test_da_a, test_da_b])
    da_res = _lifted_test_func(lds_da_x, lds_da_y, scalar_z, a=lds_da_a, b=lds_da_b, c=scalar_c)
    da_res.identical(da_expected)
    assert isinstance(da_res, xr.DataArray)

    # lift_fn_res: return type should be lifted to LiftedDataset
    da_res = _lifted_test_func_ret(lds_da_x, lds_da_y, scalar_z, a=lds_da_a, b=lds_da_b, c=scalar_c)
    assert isinstance(da_res, LiftedDataset)
    assert da_res.is_dataarray()
    assert da_res.raw().identical(da_expected)
    # <--  END

    # --- prepare dataset testcase
    test_ds_x: xr.DataArray = xr.Dataset(dict(y=test_da_x))
    test_ds_y: xr.DataArray = xr.Dataset(dict(y=test_da_y))
    test_ds_a: xr.DataArray = xr.Dataset(dict(y=test_da_a))
    test_ds_b: xr.DataArray = xr.Dataset(dict(y=test_da_b))
    ds_test = _test_func(test_ds_x, test_ds_y, scalar_z, a=test_ds_a, b=test_ds_b, c=scalar_c)
    ds_expected = xr.Dataset(dict(y=da_expected))
    # this is not the actual test - just a safety check
    assert ds_expected.identical(ds_test)

    # --- FUNCTION LIFTING TESTS FOR DATASET --> START
    # actual tests - result is still a data array, not a lifted data array
    # lift_fn: return type should be preserved, result should match expected
    lds_ds_x, lds_ds_y, lds_ds_a, lds_ds_b = map(LiftedDataset, [test_ds_x, test_ds_y, test_ds_a, test_ds_b])
    ds_res = _lifted_test_func(lds_ds_x, lds_ds_y, scalar_z, a=lds_ds_a, b=lds_ds_b, c=scalar_c)
    assert ds_res.identical(ds_expected)
    assert isinstance(ds_res, xr.Dataset)

    # lift_fn_res: return type should be lifted to LiftedDataset
    ds_res = _lifted_test_func_ret(lds_ds_x, lds_ds_y, scalar_z, a=lds_ds_a, b=lds_ds_b, c=scalar_c)
    assert isinstance(ds_res, LiftedDataset)
    assert ds_res.is_dataset()
    assert ds_res.raw().identical(ds_expected)
    # <--  END


def test_ldsutils_lift_fn_invalid_ret():
    """
    Test against a dummy function that simply returns a string.
        - Neither function should care about the input arguments
        - ``lift_fn_ret`` should raise an error because it can't lift a string to a
          ``LiftedDataset``
    """

    # dummy function that returns a string
    def _test_func(x) -> str:
        # pylint: disable=unused-argument
        return "hi"

    # lift_fn should not do anything regardless of whether any of the args or return type is
    # compatible.
    _lifted_test_func = utils.LiftedDatasetUtils.lift_fn(_test_func)
    ret = _lifted_test_func(0)
    assert ret == "hi"

    # lift_fn_ret should raise an error because of incompatible return type
    _lifted_test_func_ret = utils.LiftedDatasetUtils.lift_fn_ret(_test_func)

    # TypeError: because the output return type is a string
    with pytest.raises(TypeError):
        _lifted_test_func_ret(0)

    # TypeError: as before
    # UserWarning: should NOT occur because the input is a lifted dataset
    with pytest.raises(TypeError):
        _lifted_test_func_ret(LiftedDataset(xr.DataArray([1, 2], dims="a")))

    # TypeError: as before
    # UserWarning: should NOT occur because the input is a lifted dataset
    with pytest.raises(TypeError):
        _lifted_test_func_ret(
            LiftedDataset(
                xr.Dataset(
                    dict(x=xr.DataArray([1, 2], dims="a")),
                ),
            ),
        )
