"""
Contains unit tests for scores.emerging
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.emerging.risk_matrix import (
    _risk_matrix_score,
    _scaling_to_weight_matrix,
    matrix_weights_to_array,
    risk_matrix_score,
    weights_from_warning_scaling,
)

# mtd is used as the abbreviation in preparation for migration to multicategorical_test_data.py
from tests.emerging import emerging_test_data as mtd


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
    ("weights", "preserve_dims", "reduce_dims", "expected"),
    [
        # Sydney example from paper, escalation weights, no mean score
        (None, "all", None, mtd.EXP_RMS_CASE3A),
        # Sydney example from paper, escalation weights, unweighted mean score
        (None, ["forecaster"], None, mtd.EXP_RMS_CASE3B),
        # Sydney example from paper, escalation weights, unweighted mean score
        (None, None, ["obs_case"], mtd.EXP_RMS_CASE3B),
        # Sydney example from paper, escalation weights, weighted mean score
        (mtd.DA_RMS_WEIGHTS_SYD, ["forecaster"], None, mtd.EXP_RMS_CASE3C),
    ],
)
def test_risk_matrix_score(weights, preserve_dims, reduce_dims, expected):
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
        reduce_dims=reduce_dims,
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


def test_risk_matrix_score_datasets():
    """
    Tests that risk_matrix_score returns as expected for dataset inputs.
    """
    expected = xr.Dataset({"a": mtd.EXP_RMS_CASE3A, "b": mtd.EXP_RMS_CASE3A})
    calculated = risk_matrix_score(
        xr.Dataset({"a": mtd.DA_RMS_FCST1, "b": mtd.DA_RMS_FCST1}),
        xr.Dataset({"a": mtd.DA_RMS_OBS1, "b": mtd.DA_RMS_OBS1}),
        mtd.DA_RMS_WT2A,
        "sev",
        "prob",
        threshold_assignment="lower",
        weights=None,
        preserve_dims="all",
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    ("fcst", "obs"),
    [
        (mtd.DA_RMS_FCST1.chunk(), mtd.DA_RMS_OBS1.chunk()),
        (mtd.DA_RMS_FCST1, mtd.DA_RMS_OBS1.chunk()),
        (mtd.DA_RMS_FCST1.chunk(), mtd.DA_RMS_OBS1),
    ],
)
def test_risk_matrix_score_dask(fcst, obs):
    """Tests `risk_matrix_score` works with dask."""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    expected = mtd.EXP_RMS_CASE3A
    result = risk_matrix_score(
        fcst, obs, mtd.DA_RMS_WT2A, "sev", "prob", threshold_assignment="lower", weights=None, preserve_dims="all"
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute().transpose(*expected.dims)
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, expected)


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
        (  # some forecast values negative, fcst is a dataset
            -xr.Dataset({"a": mtd.DA_RMS_FCST, "b": mtd.DA_RMS_FCST}),
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
        (  # obs values are 1.5, 0.5 or nan, obs is a dataset
            mtd.DA_RMS_FCST,
            xr.Dataset({"a": mtd.DA_RMS_OBS + 0.5, "b": mtd.DA_RMS_OBS}),
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
def test_matrix_weights_to_array(prob_threshold_coords):
    """Tests matrix_weights_to_array"""
    expected = mtd.EXP_DECISION_WEIGHT
    calculated = matrix_weights_to_array(
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
            "`matrix_weights` must be two dimensional",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b"],
            [0.1, 0.3, 0.7],
            "number of `prob_threshold_coords` must equal number of rows of `matrix_weights`",
        ),
        (
            np.array([[1, 2], [3, 4]]),
            ["a", "b", "c"],
            [0.1, 0.3],
            "number of `severity_coords` must equal number of columns of `matrix_weights`",
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
def test_matrix_weights_to_array_raises(weight_matrix, severity_coords, prob_threshold_coords, error_msg_snippet):
    """
    Tests that matrix_weights_to_array raises as expected.
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        matrix_weights_to_array(weight_matrix, "sev", severity_coords, "prob", prob_threshold_coords)


@pytest.mark.parametrize(
    ("scaling_matrix", "assessment_weights", "expected"),
    [
        (  # LONG-RANGE scaling, equal assessment weights
            np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]),
            [1.0, 1, 1],
            np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        ),
        (  # LONG-RANGE scaling, unequal assessment weights
            np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]),
            [2.0, 1, 1],
            np.array([[0, 0, 0], [0, 0, 0], [2.0, 0, 0]]),
        ),
        (  # LONG-RANGE scaling, equal assessment weights, column and row length differ
            np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]),
            [1.0, 1, 1],
            np.array([[0.0, 0, 0], [1, 0, 0]]),
        ),
        (  # Sligh variation on MID-RANGE scaling, equal weights
            np.array([[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 1, 2], [0, 0, 0, 0]]),
            [1.0, 1, 1],
            np.array([[0.0, 0, 1], [0, 1, 0], [1, 0, 1]]),
        ),
        (  # GOLDING scaling (good tester of algorithm steps), equal weights
            np.array([[0, 1, 2, 3], [0, 1, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]),
            [1.0, 1, 1],
            np.array([[0.0, 0, 1], [1, 0, 0], [0, 2, 0]]),
        ),
        (  # GOLDING scaling, unequal weights
            np.array([[0, 1, 2, 3], [0, 1, 2, 2], [0, 0, 2, 2], [0, 0, 0, 0]]),
            [1.0, 2, 3],
            np.array([[0.0, 0, 3], [1, 0, 0], [0, 3, 0]]),
        ),
        (  # scaling using 5 warning levels
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 2, 4], [0, 0, 2, 2, 3], [0, 0, 0, 0, 0]]),
            [1.0, 1, 1, 1],
            np.array([[0.0, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 1]]),
        ),
    ],
)
def test__scaling_to_weight_matrix(scaling_matrix, assessment_weights, expected):
    """Tests _scaling_to_weight_matrix"""
    calculated = _scaling_to_weight_matrix(scaling_matrix, assessment_weights)
    np.testing.assert_array_equal(calculated, expected, strict=True)


def test_weights_from_warning_scaling():
    """Tests weights_from_warning_scaling"""
    expected = mtd.DA_RMS_WT2
    # SHORT-RANGE scaling
    scaling_matrix = np.array([[0, 2, 3, 3], [0, 1, 2, 3], [0, 1, 1, 2], [0, 0, 0, 0]])
    calculated = weights_from_warning_scaling(
        scaling_matrix, [1, 2, 3], "sev", [1, 2, 3], "prob", (0.1, 0.4, 0.7)
    ).transpose(*expected.dims)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    (
        "scaling_matrix",
        "assessment_weights",
        "severity_coords",
        "prob_threshold_coords",
        "error_msg_snippet",
    ),
    [
        (
            np.array([[[1], [2]], [[3], [4]]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`scaling_matrix` should be two dimensional",
        ),
        (
            np.array([[0, 3, 4], [0, 2, 1.1], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`scaling_matrix` should only have have integer entries",
        ),
        (
            np.array([[0, 3, 4], [0, 2, -1], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`scaling_matrix` should only have non-negative integer values",
        ),
        (
            np.array([[1, 3, 4], [0, 2, 1], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "The first column of `scaling_matrix` should consist of zeros only",
        ),
        (
            np.array([[0, 3, 4], [0, 2, 1], [0, 0, 1]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "The last row of `scaling_matrix` should consist of zeros only",
        ),
        (
            np.array([[0, 3, 2], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`scaling_matrix` should be non-decreasing along each row",
        ),
        (
            np.array([[0, 1, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3, 0.5],
            "`scaling_matrix` should be non-increasing along each column",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2],
            [0.1, 0.3, 0.5],
            "Length of `prob_threshold_coords` should be one less than rows of `scaling_matrix`",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 1),
            [1, 2, 3],
            [0.1, 0.3],
            "Length of `severity_coords` should be one less than columns of `scaling_matrix`",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1),
            [1, 2],
            [0.1, 0.3],
            "length of `assessment_weights` must be at least the highest value in `scaling_matrix`",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 2, 4),
            [1, 2],
            [0, 0.3],
            "`prob_threshold_coords` must strictly between 0 and 1",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1, 1, 2, 4),
            [1, 2],
            [0.1, 1],
            "`prob_threshold_coords` must strictly between 0 and 1",
        ),
        (
            np.array([[0, 2, 3], [0, 2, 2], [0, 0, 0]]),
            (1.4, 0, 2),
            [1, 2],
            [0.1, 0.3],
            "values in `assessment_weights` must be positive",
        ),
    ],
)
def test_weights_from_warning_scaling_raises(
    scaling_matrix, assessment_weights, severity_coords, prob_threshold_coords, error_msg_snippet
):
    """
    Tests that weights_from_warning_scaling raises as expected.
    """
    with pytest.raises(ValueError, match=error_msg_snippet):
        weights_from_warning_scaling(
            scaling_matrix, assessment_weights, "sev", severity_coords, "prob", prob_threshold_coords
        )
