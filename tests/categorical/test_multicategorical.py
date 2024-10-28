"""
Contains unit tests for scores.categorical
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.categorical import firm
from scores.categorical.multicategorical_impl import (
    _single_category_score,
    risk_matrix_score,
    _risk_matrix_score,
    risk_matrix_weights_to_array,
)
from scores.utils import DimensionError
from tests.categorical import multicategorical_test_data as mtd


@pytest.mark.parametrize(
    ("fcst", "obs", "categorical_threshold", "discount_distance", "threshold_assignment", "expected"),
    [
        # Threshold 5, discount = 0, preserve all dims
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 0, "lower", mtd.EXP_SC_CASE0),
        # Threshold -200, discount = 0, preserve 1 dim
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, -200, 0, "lower", mtd.EXP_SC_CASE1),
        # Threshold 200, discount = 0, preserve 1 dim
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 200, 0, "lower", mtd.EXP_SC_CASE1),
        # Threshold 5, discount = 7, preserve all dims.
        # discount_distance is maximum for both false alarms and misses
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 7, "lower", mtd.EXP_SC_CASE2),
        # Threshold 5, discount = 0.5, preserve all dims.
        # discount_distance is minimum for both false alarms and misses
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, 5, 0.5, "lower", mtd.EXP_SC_CASE3),
        # Test lower/right assignment
        (mtd.DA_FCST_SC2, mtd.DA_OBS_SC2, 2, None, "lower", mtd.EXP_SC_CASE4),
        # Test upper/left assignment
        (mtd.DA_FCST_SC2, mtd.DA_OBS_SC2, 2, None, "upper", mtd.EXP_SC_CASE5),
        # Threshold xr.Datarray, discount = 0, preserve all dims
        (mtd.DA_FCST_SC, mtd.DA_OBS_SC, mtd.DA_THRESHOLD_SC, 0, "lower", mtd.EXP_SC_CASE6),
    ],
)

# pylint: disable=too-many-positional-arguments
def test__single_category_score(fcst, obs, categorical_threshold, discount_distance, threshold_assignment, expected):
    """Tests _single_category_score"""
    risk_parameter = 0.7

    calculated = _single_category_score(
        fcst,
        obs,
        risk_parameter,
        categorical_threshold,
        discount_distance=discount_distance,
        threshold_assignment=threshold_assignment,
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
        # 2 categories defined with xr.DataArrays for threhsolds that don't vary by coord
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            mtd.DA_THRESHOLD_FIRM,
            [1, 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE3,
        ),
        # 2 categories defined with xr.DataArrays for threhsolds that do vary
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.7,
            mtd.DA_THRESHOLD_FIRM2,
            [1, 1],
            None,
            ["i", "j", "k"],
            0,
            mtd.EXP_FIRM_CASE3,
        ),
    ],
)

# pylint: disable=too-many-positional-arguments
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
        discount_distance=discount_distance,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    if preserve_dims is not None:
        calculated = calculated.transpose(*preserve_dims)
    xr.testing.assert_allclose(
        calculated,
        expected,
        atol=0.001,
    )


def test_firm_dask():
    """Tests firm works with dask"""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run dask tests")  # pragma: no cover

    calculated = firm(
        mtd.DA_FCST_FIRM.chunk(),
        mtd.DA_OBS_FIRM.chunk(),
        0.7,
        [0, 5],
        mtd.LIST_WEIGHTS_FIRM2,
        discount_distance=0,
        reduce_dims=None,
        preserve_dims=["i", "j", "k"],
    )

    calculated = calculated.transpose("i", "j", "k")

    assert isinstance(calculated.firm_score.data, dask.array.Array)
    calculated = calculated.compute()
    assert isinstance(calculated.firm_score.data, np.ndarray)
    xr.testing.assert_allclose(
        calculated,
        mtd.EXP_FIRM_CASE6,
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
        "threshold_assignment",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
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
            "upper",
            ValueError,
            "`discount_distance` must be >= 0",
        ),
        # wrong threshold assignment
        (
            mtd.DA_FCST_FIRM,
            mtd.DA_OBS_FIRM,
            0.5,
            [5],
            [1],
            ["i", "j", "k"],
            0.0,
            "up",
            ValueError,
            """ `threshold_assignment` must be either \"upper\" or \"lower\" """,
        ),
    ],
)

# pylint: disable=too-many-positional-arguments
def test_firm_raises(
    fcst,
    obs,
    risk_parameters,
    categorical_thresholds,
    weights,
    preserve_dims,
    discount_distance,
    threshold_assignment,
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
            discount_distance=discount_distance,
            reduce_dims=None,
            preserve_dims=preserve_dims,
            threshold_assignment=threshold_assignment,
        )


@pytest.mark.parametrize(
    ("fcst", "obs", "decision_weights", "severity_dim", "prob_threshold_dim", "threshold_assignment", "expected"),
    [
        # test "lower" (0.4 prob threshold), plus nan behaviour
        (mtd.DA_RMS_FCST, mtd.DA_RMS_OBS, mtd.DA_RMS_WT0, "sev", "prob", "lower", mtd.EXP_RMS_CASE0),
        # test "upper" (0.4 prob threshold)
        (mtd.DA_RMS_FCST, mtd.DA_RMS_OBS, mtd.DA_RMS_WT0, "sev", "prob", "upper", mtd.EXP_RMS_CASE1),
        # Sydney example from paper, weights 1, variety of possible obs
        (mtd.DA_RMS_FCST1, mtd.DA_RMS_OBS1, mtd.DA_RMS_WT1, "sev", "prob", "lower", mtd.EXP_RMS_CASE2),
        # Sydney example from paper, escalation weights, variety of possible obs
        (mtd.DA_RMS_FCST1, mtd.DA_RMS_OBS1, mtd.DA_RMS_WT2, "sev", "prob", "lower", mtd.EXP_RMS_CASE3),
        # Sydney example from paper, escalation weights, sev coords transposed in weight matrix
        (mtd.DA_RMS_FCST1, mtd.DA_RMS_OBS1, mtd.DA_RMS_WT2A, "sev", "prob", "lower", mtd.EXP_RMS_CASE3),
    ],
)
def test__risk_matrix_score(
    fcst, obs, decision_weights, severity_dim, prob_threshold_dim, threshold_assignment, expected
):
    """Tests _risk_matrix_score"""
    calculated = _risk_matrix_score(
        fcst,
        obs,
        decision_weights,
        severity_dim,
        prob_threshold_dim,
        threshold_assignment=threshold_assignment,
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    ("weights", "preserve_dims", "expected"),
    [
        # Sydney example from paper, escalation weights, no mean score
        (None, "all", mtd.EXP_RMS_CASE3A),
        # Sydney example from paper, escalation weights, unweighted mean score
        (None, ["forecaster"], mtd.EXP_RMS_CASE3B),
        # Sydney example from paper, escalation weights, weighted mean score
        (mtd.DA_RMS_WEIGHTS_SYD, ["forecaster"], mtd.EXP_RMS_CASE3C),
    ],
)
def test_risk_matrix_score(weights, preserve_dims, expected):
    """Tests risk_matrix_score weighted means"""
    calculated = risk_matrix_score(
        mtd.DA_RMS_FCST1,
        mtd.DA_RMS_OBS1,
        mtd.DA_RMS_WT2A,
        "sev",
        "prob",
        threshold_assignment="lower",
        weights=weights,
        preserve_dims=preserve_dims,
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "decision_weights",
        "severity_dim",
        "prob_threshold_dim",
        "threshold_assignment",
        "weights",
        "error_msg_snippet",
    ),
    [
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "severity_cat",
            "prob",
            "upper",
            None,
            "`severity_dim` must be a dimension of `fcst`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS.rename({"sev": "severity"}),
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            None,
            "`severity_dim` must be a dimension of `obs`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0.rename({"sev": "severity"}),
            "sev",
            "prob",
            "upper",
            None,
            "`severity_dim` must be a dimension of `decision_weights`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            mtd.DA_RMS_WEIGHTS.rename({"obs_case": "sev"}),
            "`severity_dim` must not be a dimension of `weights`",
        ),
        (
            mtd.DA_RMS_FCST.rename({"day": "prob"}),
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            None,
            "`prob_threshold_dim` must not be a dimension of `fcst`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS.rename({"day": "prob"}),
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            None,
            "`prob_threshold_dim` must not be a dimension of `obs`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "probability",
            "upper",
            None,
            "`prob_threshold_dim` must be a dimension of `decision_weights`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            mtd.DA_RMS_WEIGHTS.rename({"obs_case": "prob"}),
            "`prob_threshold_dim` must not be a dimension of `weights`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT3,
            "sev",
            "prob",
            "upper",
            None,
            "`decision_weights` must have exactly 2 dimensions: `severity_dim` and `prob_threshold_dim`",
        ),
        (  # some forecast values greater than 1
            100 * mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT2,
            "sev",
            "prob",
            "upper",
            None,
            "values in `fcst` must lie in the closed interval ",
        ),
        (  # some forecast values negative
            -mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT2,
            "sev",
            "prob",
            "upper",
            None,
            "values in `fcst` must lie in the closed interval ",
        ),
        (  # obs values are 0, 2, or nan
            mtd.DA_RMS_FCST,
            2 * mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT2,
            "sev",
            "prob",
            "upper",
            None,
            "values in `obs` can only be 0, 1 or nan",
        ),
        (  # obs values are 1.5, 0.5 or nan
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS + 0.5,
            mtd.DA_RMS_WT2,
            "sev",
            "prob",
            "upper",
            None,
            "values in `obs` can only be 0, 1 or nan",
        ),
        (  # some probability thresholds at least 1
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT4,
            "sev",
            "prob",
            "upper",
            None,
            "`prob_threshold_dim` coordinates must be strictly between 0 and 1",
        ),
        (  # some probability thresholds no greater than 0
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT5,
            "sev",
            "prob",
            "upper",
            None,
            "`prob_threshold_dim` coordinates must be strictly between 0 and 1",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT6,
            "sev",
            "prob",
            "upper",
            None,
            "`severity_dim` coordinates do not match in `decision_weights` and `fcst`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS2,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "upper",
            None,
            "`severity_dim` coordinates do not match in `decision_weights` and `obs`",
        ),
        (
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            "mid",
            None,
            """ `threshold_assignment` must be either \"upper\" or \"lower\" """,
        ),
    ],
)
def test_risk_matrix_score_raises(
    fcst,
    obs,
    decision_weights,
    severity_dim,
    prob_threshold_dim,
    threshold_assignment,
    weights,
    error_msg_snippet,
):
    """
    Tests that the risk_matrix_score raises the correct errors
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        risk_matrix_score(
            fcst,
            obs,
            decision_weights,
            severity_dim,
            prob_threshold_dim,
            threshold_assignment=threshold_assignment,
            weights=weights,
            reduce_dims=None,
            preserve_dims=None,
        )


@pytest.mark.parametrize(
    (
        "reduce_dims",
        "preserve_dims",
        "error_msg_snippet",
    ),
    [
        (
            ["sev", "stn"],
            None,
            "You are requesting to reduce a dimension which does not appear in your data",
        ),
        (
            None,
            ["sev", "stn"],
            "You are requesting to preserve a dimension which does not appear in your data",
        ),
    ],
)
def test_risk_matrix_score_raises2(
    reduce_dims,
    preserve_dims,
    error_msg_snippet,
):
    """
    Tests that the risk_matrix_score raises the correct errors when `gather_dimensions` is called
    and involves the severity or prob threshold dims.
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        risk_matrix_score(
            mtd.DA_RMS_FCST,
            mtd.DA_RMS_OBS,
            mtd.DA_RMS_WT0,
            "sev",
            "prob",
            threshold_assignment="lower",
            weights=None,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )


@pytest.mark.parametrize(
    ("prob_threshold_coords"),
    [
        ([0.1, 0.7, 0.4, 0.2]),
        ([0.4, 0.7, 0.1, 0.2]),
    ],
)
def test_risk_matrix_weights_to_array(prob_threshold_coords):
    """Tests risk_matrix_weights_to_array"""
    expected = mtd.EXP_DECISION_WEIGHT
    calculated = risk_matrix_weights_to_array(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        "sev",
        [0, 1, 2],
        "prob",
        prob_threshold_coords,
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    (
        "weight_matrix",
        "severity_coords",
        "prob_threshold_coords",
        "error_msg_snippet",
    ),
    [
        (
            np.array([1, 2, 3, 4]),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`weight_matrix` must be two dimensional",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b"],
            [0.1, 0.3, 0.7],
            "number of `prob_threshold_coords` must equal number of rows of `weight_matrix`",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b", "c"],
            [0.1, 0.3],
            "number of `severity_coords` must equal number of columns of `weight_matrix`",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b"],
            [0.1, 1.3],
            "`prob_threshold_coords` must strictly between 0 and 1",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b"],
            [0.0, 0.3],
            "`prob_threshold_coords` must strictly between 0 and 1",
        ),
    ],
)
def test_risk_matrix_weights_to_array_raises(weight_matrix, severity_coords, prob_threshold_coords, error_msg_snippet):
    """
    Tests that the risk_matrix_score raises the correct errors when `gather_dimensions` is called
    and involves the severity or prob threshold dims.
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        risk_matrix_weights_to_array(weight_matrix, "sev", severity_coords, "prob", prob_threshold_coords)
