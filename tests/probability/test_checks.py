"""
This module contains unit tests for scores.probability.checks
"""

import pytest

import scores
from tests.probability import cdf_test_data


@pytest.mark.parametrize(
    (
        "cdf",
        "threshold_dim",
        "tolerance",
        "error_msg_snippet",
    ),
    [
        (
            cdf_test_data.DA_DECREASING_CDFS1,
            "y",
            0,
            "'y' is not a dimension of `cdf`",
        ),
        (
            cdf_test_data.DA_DECREASING_CDFS1,
            "x",
            -1,
            "`tolerance` must be nonnegative.",
        ),
        (
            cdf_test_data.DA_CDF_ENVELOPE2,
            "y",
            0,
            "'y' is not a dimension of `cdf`",
        ),
        (
            cdf_test_data.DA_CDF_ENVELOPE1,
            "x",
            0,
            "CDFs should have no NaNs or be all NaN along `threshold_dim`",
        ),
        (
            cdf_test_data.DA_NAN_DECREASING_CDFS2,
            "x",
            0,
            "Coordinates along 'x' dimension should be increasing.",
        ),
    ],
)
def test_check_nan_decreasing_inputs(cdf, threshold_dim, tolerance, error_msg_snippet):
    """Tests that `_check_nan_decreasing_inputs` raises an exception as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        scores.probability.checks.check_nan_decreasing_inputs(cdf, threshold_dim, tolerance)
