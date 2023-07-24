"""
Contains unit tests for scores.categorical
"""
import multicategorical_test_data as mtd
import pytest
import xarray as xr

from scores.categorical import firm
from scores.categorical.multicategorical_impl import _single_category_score
from scores.utils import DimensionError


def test_dummy_test():
    assert 1 == 1


@pytest.mark.parametrize(
    ("fcst", "obs", "categorical_threshold", "discount_distance", "expected"),
    [
        # Threshold 5, discount = 0, preserve all dims
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 0, mtd.EXP_SC_CASE0),
        # Threshold -200, discount = 0, preserve 1 dim
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, -200, 0, mtd.EXP_SC_CASE1),
        # Threshold 200, discount = 0, preserve 1 dim
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 200, 0, mtd.EXP_SC_CASE1),
        # Threshold 5, discount = 7, preserve all dims.
        # discount_distance is maximum for both false alarms and misses
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 7, mtd.EXP_SC_CASE2),
        # Threshold 5, discount = 0.5, preserve all dims.
        # discount_distance is minimum for both false alarms and misses
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 0.5, mtd.EXP_SC_CASE3),
    ],
)
def test__single_category_score(fcst, obs, categorical_threshold, discount_distance, expected):
    """Tests _single_category_score"""
    risk_parameter = 0.7

    calculated = _single_category_score(
        fcst,
        obs,
        risk_parameter,
        categorical_threshold,
        discount_distance,
    )
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "risk_parameters",
        "categorical_thresholds",
        "weights",
        "reduce_dims",
        "preserve_dims",
        "discount_distance",
        "expected",
    ),
    [
        # Test for single category case identical to CASE0 in
        # test__single_category_score
        (
            mtd.DA_FCST_SC,
            mtd.DA_OBS_SC,
            0.7,
            [5],
            [1],
            None,
            ["i", "j", "k"],
            0.0,
            mtd.EXP_SC_CASE0,
        ),
        # Test for single category with discount distance. Identical to CASE3
        # in test__single_category_score
        (
            mtd.DA_FCST_SC,
            mtd.DA_OBS_SC,
            0.7,
            [5],
            [1],
            None,
            ["i", "j", "k"],
            0.5,
            mtd.EXP_SC_CASE3,
        ),
        # Test for single category case on slightly bigger dataset
        # Preserve dimensions
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [5],
            [1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE0,
        ),
        # Single category, only keep one dimension
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [5],
            [1],
            None,
            ["i"],
            0,
            mtd.EXP_FIRM_CASE1,
        ),
        # Single category, no dimensions
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [5],
            [1],
            None,
            None,
            0,
            mtd.EXP_FIRM_CASE2,
        ),
        # Single category, no dimensions
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [5],
            [1],
            "all",
            None,
            0,
            mtd.EXP_FIRM_CASE2,
        ),
        # 2 categories, same weight
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            [1, 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE3,
        ),
        # 3 categories, same weight
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5, 8],
            [1, 1, 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE4,
        ),
        # 2 categories, 2 weights
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            [2, 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE4,
        ),
        # 2 categories, 2 weights that are xr.DataArrays
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            mtd.LIST_WEIGHTS_FIRM0,
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE4,
        ),
        # 2 categories, 2 weights with the first a xr.DataArray and the second
        # a float
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            [mtd.LIST_WEIGHTS_FIRM0[0], 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE4,
        ),
        # 2 categories, 2 weights that are xr.DataArrays with different values
        # for different coords
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            mtd.LIST_WEIGHTS_FIRM1,
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE5,
        ),
        # 2 categories, 2 weights that are xr.DataArrays with different values
        # for different coords, with NaN in weights
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            [0, 5],
            mtd.LIST_WEIGHTS_FIRM2,
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE6,
        ),
    ],
)
def test_firm(
    fcst,
    obs,
    risk_parameters,
    categorical_thresholds,
    weights,
    reduce_dims,
    preserve_dims,
    discount_distance,
    expected,
):
    """Tests firm"""
    calculated = firm(
        fcst,
        obs,
        risk_parameters,
        categorical_thresholds,
        weights,
        discount_distance,
        reduce_dims,
        preserve_dims,
    )
    if preserve_dims != None:
        calculated = calculated.transpose(*preserve_dims)
    xr.testing.assert_allclose(
        calculated,
        expected,
        atol=0.001,
    )


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "risk_parameters",
        "categorical_thresholds",
        "weights",
        "preserve_dims",
        "discount_distance",
        "error_type",
        "error_msg_snippet",
    ),
    [
        # len(categorical_thresholds) is 0
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [],
            [],
            ["i", "j", "k"],
            0,
            ValueError,
            "`categorical_thresholds` must have at least",
        ),
        # weights and categories don't match. risk_parameters is a float
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [1],
            [1, 2],
            ["i", "j", "k"],
            0,
            ValueError,
            "`categorical_thresholds` and `weights`",
        ),
        # risk_parameter = 0
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.0,
            [5],
            [1],
            ["i", "j", "k"],
            0,
            ValueError,
            "0 < `risk_parameter` < 1 must",
        ),
        # risk_parameter = 1
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.0,
            [5],
            [1],
            ["i", "j", "k"],
            0,
            ValueError,
            "0 < `risk_parameter` < 1 must",
        ),
        # negative weight
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5, 6],
            [1, -1],
            ["i", "j", "k"],
            0,
            ValueError,
            "`weights` must be > 0",
        ),
        # negative weight with weights being a xr.DataArray
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5, 6],
            mtd.LIST_WEIGHTS_FIRM3,
            ["i", "j", "k"],
            0,
            ValueError,
            "value was found in index 0 of `weights",
        ),
        # zero weight
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5, 6],
            [1, 0],
            ["i", "j", "k"],
            0,
            ValueError,
            "`weights` must be > 0",
        ),
        # zero weight with weights being a xr.DataArray
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5, 6],
            mtd.LIST_WEIGHTS_FIRM4,
            ["i", "j", "k"],
            0,
            ValueError,
            "No values <= 0 are allowed in `weights`",
        ),
        # bad dims in weights
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5, 6],
            mtd.LIST_WEIGHTS_FIRM5,
            ["i", "j", "k"],
            0,
            DimensionError,
            "of data object are not subset to",
        ),
        # negative discount distance
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5],
            [1],
            ["i", "j", "k"],
            -1,
            ValueError,
            "`discount_distance` must be >= 0",
        ),
    ],
)
def test_firm_raises(
    fcst,
    obs,
    risk_parameters,
    categorical_thresholds,
    weights,
    preserve_dims,
    discount_distance,
    error_type,
    error_msg_snippet,
):
    """
    Tests that the firm raises the correct errors
    """
    with pytest.raises(error_type, match=error_msg_snippet):
        firm(
            fcst,
            obs,
            risk_parameters,
            categorical_thresholds,
            weights,
            discount_distance,
            None,
            preserve_dims,
        )
