# pylint disable: too-many-arguments
"""
Contains unit tests for scores.probability.crps
"""

from typing import Iterable, Optional  # pylint: disable=unused-import

import crps_test_data  # pylint: disable=import-error
import dask
import numpy as np
import pytest
from assertions import assert_dataarray_equal  # pylint: disable=import-error
from assertions import assert_dataset_equal  # pylint: disable=import-error

from scores.probability import (
    adjust_fcst_for_crps,
    crps_cdf,
    crps_cdf_brier_decomposition,
)
from scores.probability.crps_impl import (
    crps_cdf_exact,
    crps_cdf_reformat_inputs,
    crps_cdf_trapz,
    crps_step_threshold_weight,
)


@pytest.mark.parametrize(
    ("weight_upper", "expected"),
    [
        (True, crps_test_data.EXP_STEP_WEIGHT_UPPER),
        (False, crps_test_data.EXP_STEP_WEIGHT_LOWER),
    ],
)
def test_crps_stepweight(
    weight_upper,
    expected,
):
    """Tests `crps_step_threshold_weight` with a variety of inputs."""
    result = crps_step_threshold_weight(
        crps_test_data.DA_STEP_WEIGHT,
        "x",
        [1, 2, 3, 4, 5, 6],
        True,
        0.2,
        weight_upper,
    )
    assert_dataarray_equal(result, expected, decimals=7)


def test_crps_cdf_exact():
    """Tests `crps_cdf_exact`."""
    result = crps_cdf_exact(
        crps_test_data.DA_FCST_CRPS_EXACT,
        crps_test_data.DA_OBS_CRPS_EXACT,
        crps_test_data.DA_WT_CRPS_EXACT,
        "x",
        include_components=True,
    )
    assert_dataset_equal(result, crps_test_data.EXP_CRPS_EXACT, decimals=7)

    result2 = crps_cdf_exact(
        crps_test_data.DA_FCST_CRPS_EXACT,
        crps_test_data.DA_OBS_CRPS_EXACT,
        crps_test_data.DA_WT_CRPS_EXACT,
        "x",
        include_components=False,
    )

    assert list(result2.data_vars) == ["total"]


def test_crps_cdf_exact_dask():
    """Tests `crps_cdf_exact` works with Dask."""
    result = crps_cdf_exact(
        crps_test_data.DA_FCST_CRPS_EXACT.chunk(),
        crps_test_data.DA_OBS_CRPS_EXACT.chunk(),
        crps_test_data.DA_WT_CRPS_EXACT,
        "x",
        include_components=True,
    )
    assert isinstance(result.total.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.total.data, np.ndarray)
    assert_dataset_equal(result, crps_test_data.EXP_CRPS_EXACT, decimals=7)

    result2 = crps_cdf_exact(
        crps_test_data.DA_FCST_CRPS_EXACT.chunk(),
        crps_test_data.DA_OBS_CRPS_EXACT.chunk(),
        crps_test_data.DA_WT_CRPS_EXACT.chunk(),
        "x",
        include_components=False,
    )
    assert isinstance(result2.total.data, dask.array.Array)
    result2 = result2.compute()
    assert isinstance(result2.total.data, np.ndarray)
    assert list(result2.data_vars) == ["total"]


@pytest.mark.parametrize(
    ("weight_upper", "expected"),
    [
        (True, crps_test_data.EXP_STEP_WEIGHT_UPPER),
        (False, crps_test_data.EXP_STEP_WEIGHT_LOWER),
    ],
)
def test_crps_stepweight2(
    weight_upper,
    expected,
):
    """Tests `crps_step_threshold_weight` with a variety of inputs."""
    result = crps_step_threshold_weight(
        crps_test_data.DA_STEP_WEIGHT,
        "x",
        [1, 2, 3, 4, 5, 6],
        True,
        0.2,
        weight_upper,
    )
    assert_dataarray_equal(result, expected, decimals=7)


def test_crps_cdf_trapz():
    """Tests `crps_cdf_trapz`. Uses dense interpolation to get an approximate (to 4 dec pl) result."""
    result = crps_cdf_trapz(
        crps_test_data.DA_FCST_CRPS_DENSE,
        crps_test_data.DA_OBS_CRPS_DENSE,
        crps_test_data.DA_WT_CRPS_DENSE,
        "x",
        include_components=True,
    )
    assert_dataset_equal(result, crps_test_data.EXP_CRPS_EXACT, decimals=4)

    result2 = crps_cdf_trapz(
        crps_test_data.DA_FCST_CRPS_DENSE,
        crps_test_data.DA_OBS_CRPS_DENSE,
        crps_test_data.DA_WT_CRPS_DENSE,
        "x",
        include_components=False,
    )
    assert_dataset_equal(result, crps_test_data.EXP_CRPS_EXACT, decimals=4)
    assert list(result2.data_vars) == ["total"]


@pytest.mark.parametrize(
    (
        "threshold_weight",
        "additional_thresholds",
        "expected",
    ),
    [
        (
            crps_test_data.DA_WT_REFORMAT1,
            None,
            crps_test_data.EXP_REFORMAT1,
        ),
        (
            crps_test_data.DA_WT_REFORMAT1,
            [0, 1.5],
            crps_test_data.EXP_REFORMAT2,
        ),
        (
            None,
            None,
            crps_test_data.EXP_REFORMAT3,
        ),
    ],
)
def test_crps_cdf_reformat_inputs(
    threshold_weight,
    additional_thresholds,
    expected,
):
    """Tests `crps_cdf_reformat_inputs` with a variety of inputs."""
    result = crps_cdf_reformat_inputs(
        crps_test_data.DA_FCST_REFORMAT1,
        crps_test_data.DA_OBS_REFORMAT1,
        "x",
        threshold_weight,
        additional_thresholds,
        "linear",
        "forward",
    )
    assert len(result) == len(expected)

    for res, exp in zip(result, expected):
        assert_dataarray_equal(res, exp, decimals=7)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "threshold_dim",
        "threshold_weight",
        "fcst_fill_method",
        "threshold_weight_fill_method",
        "integration_method",
        "dims",
        "error_msg_snippet",
    ),
    [
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "y",
            None,
            "linear",
            None,
            "exact",
            None,
            "'y' is not a dimension of `fcst`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            crps_test_data.DA_WT_CHECK_CRPS1,
            "linear",
            "forward",
            "exact",
            None,
            "'x' is not a dimension of `threshold_weight`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "station",
            None,
            "linear",
            None,
            "exact",
            None,
            "'station' is a dimension of `obs`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_CHECK_CRPS,
            "x",
            None,
            "linear",
            None,
            "exact",
            None,
            "Dimensions of `obs` must be a subset of dimensions of `fcst`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            crps_test_data.DA_WT_CHECK_CRPS2,
            "linear",
            "backward",
            "exact",
            None,
            "Dimensions of `threshold_weight` must be a subset of dimensions of `fcst`",
        ),
        # TODO: Revisit if still needed after more handling in gather_dimensions
        # (
        #     crps_test_data.DA_FCST_REFORMAT1,
        #     crps_test_data.DA_OBS_REFORMAT1,
        #     "x",
        #     None,
        #     "linear",
        #     None,
        #     "exact",
        #     ["y", "x"],
        #     "`dims` must be a subset of `fcst` dimensions",
        # ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            None,
            "fat",
            None,
            "exact",
            None,
            "`fcst_fill_method` must be 'linear', 'step', 'forward' or 'backward'",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            crps_test_data.DA_WT_REFORMAT1,
            "linear",
            "fat",
            "exact",
            None,
            "`threshold_weight_fill_method` must be 'linear', 'step', 'forward' or 'backward'",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            crps_test_data.DA_WT_REFORMAT1,
            "linear",
            "step",
            "waffly",
            None,
            "`integration_method` must be 'exact' or 'trapz'",
        ),
        (
            crps_test_data.DA_FCST_CHECK_CRPS,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            crps_test_data.DA_WT_REFORMAT1,
            "linear",
            "step",
            "exact",
            None,
            "`threshold_dim` in `fcst` must have at least 2 values to calculate CRPS",
        ),
        (
            crps_test_data.DA_FCST_CHECK_CRPS2,
            crps_test_data.DA_OBS_CHECK_CRPS2,
            "x",
            None,
            "linear",
            "step",
            "exact",
            None,
            "`threshold_dim` coordinates in `fcst` must be increasing",
        ),
        (
            crps_test_data.DA_FCST_CHECK_CRPS2A,
            crps_test_data.DA_OBS_CHECK_CRPS2,
            "x",
            crps_test_data.DA_FCST_CHECK_CRPS2,  # weight
            "linear",
            "step",
            "exact",
            None,
            "`threshold_dim` coordinates in `threshold_weight` must be increasing",
        ),
        (
            crps_test_data.DA_FCST_CHECK_CRPS2A,
            crps_test_data.DA_OBS_CHECK_CRPS2,
            "x",
            crps_test_data.DA_WT_CHECK_CRPS3,  # weight
            "linear",
            "step",
            "exact",
            None,
            "`threshold_weight` has negative values",
        ),
    ],
)
# pylint: disable=too-many-arguments
def test_crps_cdf_raises(
    fcst,
    obs,
    threshold_dim,
    threshold_weight,
    fcst_fill_method,
    threshold_weight_fill_method,
    integration_method,
    dims,
    error_msg_snippet,
):
    """Check that `crps` raises exceptions as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        crps_cdf(
            fcst,
            obs,
            threshold_dim,
            threshold_weight,
            [],
            True,
            fcst_fill_method,
            threshold_weight_fill_method,
            integration_method,
            dims,
        )


@pytest.mark.parametrize(
    (
        "fcst",
        "threshold_weight",
        "propagate_nan",
        "integration_method",
        "dims",
        "expected_and_dec",  # tuple: expected and decimals
    ),
    [
        (  # exact, preserve station dim
            crps_test_data.DA_FCST_CRPS,
            crps_test_data.DA_WT_CRPS,
            True,
            "exact",
            ["station"],
            (crps_test_data.EXP_CRPS1, 7),
        ),
        (  # exact, preserve no dims
            crps_test_data.DA_FCST_CRPS,
            crps_test_data.DA_WT_CRPS,
            True,
            "exact",
            None,
            (crps_test_data.EXP_CRPS2, 7),
        ),
        (  # trapz, preserve station dim
            crps_test_data.DA_FCST_CRPS_DENSE,
            crps_test_data.DA_WT_CRPS_DENSE,
            True,
            "trapz",
            ["station"],
            (crps_test_data.EXP_CRPS1, 4),
        ),
        (  # exact, preserve station dim, weight is None
            crps_test_data.DA_FCST_CRPS,
            None,
            True,
            "exact",
            ["station"],
            (crps_test_data.EXP_CRPS3, 7),
        ),
        (  # exact, preserve station dim, weight is None, don't propagate nan
            crps_test_data.DA_FCST_CRPS,
            None,
            False,
            "exact",
            ["station"],
            (crps_test_data.EXP_CRPS4, 7),
        ),
        (  # exact, preserve station dim, don't propagate nan
            crps_test_data.DA_FCST_CRPS,
            crps_test_data.DA_WT_CRPS,
            False,
            "exact",
            ["station"],
            (crps_test_data.EXP_CRPS5, 7),
        ),
    ],
)
# pylint: disable=too-many-arguments
def test_crps_cdf(
    fcst,
    threshold_weight,
    propagate_nan,
    integration_method,
    dims,
    expected_and_dec,
):
    """Tests `crps` with a variety of inputs."""
    result = crps_cdf(
        fcst,
        crps_test_data.DA_OBS_CRPS,
        "x",
        threshold_weight=threshold_weight,
        additional_thresholds=None,
        propagate_nans=propagate_nan,
        fcst_fill_method="linear",
        threshold_weight_fill_method="forward",
        integration_method=integration_method,
        preserve_dims=dims,
        include_components=True,
    )

    assert_dataset_equal(result, expected_and_dec[0], decimals=expected_and_dec[1])


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "decreasing_tolerance",
        "expected",
    ),
    [
        (
            crps_test_data.DA_FCST_ADJUST1,
            crps_test_data.DA_OBS_ADJUST1,
            0,
            crps_test_data.EXP_FCST_ADJUST1,
        ),
        (  # exact, preserve station dim
            crps_test_data.DA_FCST_ADJUST2,
            crps_test_data.DA_OBS_ADJUST2,
            10,
            crps_test_data.EXP_FCST_ADJUST2,
        ),
    ],
)
def test_adjust_fcst_for_crps(
    fcst,
    obs,
    decreasing_tolerance,
    expected,
):
    """Tests `adjust_fcst_for_crps` with a variety of inputs."""
    result = adjust_fcst_for_crps(fcst, "x", obs, decreasing_tolerance)
    assert_dataarray_equal(result, expected, decimals=7)


@pytest.mark.parametrize(
    (
        "threshold_dim",
        "decreasing_tolerance",
        "error_msg_snippet",
    ),
    [
        (
            "aj",
            0,
            "'aj' is not a dimension of `fcst`",
        ),
        (
            "x",
            -10,
            "`decreasing_tolerance` must be nonnegative",
        ),
    ],
)
def test_adjust_fcst_raises(
    threshold_dim,
    decreasing_tolerance,
    error_msg_snippet,
):
    """Check that `adjust_fcst_for_crps` raises exceptions as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        adjust_fcst_for_crps(
            crps_test_data.DA_FCST_ADJUST1,
            threshold_dim,
            crps_test_data.DA_OBS_ADJUST1,
            decreasing_tolerance,
        )


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "threshold_dim",
        "fcst_fill_method",
        "dims",
        "error_msg_snippet",
    ),
    [
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "y",
            "linear",
            None,
            "'y' is not a dimension of `fcst`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "station",
            "linear",
            None,
            "'station' is a dimension of `obs`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_CHECK_CRPS,
            "x",
            "linear",
            None,
            "Dimensions of `obs` must be a subset of dimensions of `fcst`",
        ),
        (
            crps_test_data.DA_FCST_REFORMAT1,
            crps_test_data.DA_OBS_REFORMAT1,
            "x",
            "fat",
            None,
            "`fcst_fill_method` must be 'linear', 'step', 'forward' or 'backward'",
        ),
        (
            crps_test_data.DA_FCST_CHECK_CRPS2,
            crps_test_data.DA_OBS_CHECK_CRPS2,
            "x",
            "linear",
            None,
            "`threshold_dim` coordinates in `fcst` must be increasing",
        ),
    ],
)
# pylint: disable=too-many-arguments
def test_crps_cdf_brier_raises(
    fcst,
    obs,
    threshold_dim,
    fcst_fill_method,
    dims,
    error_msg_snippet,
):
    """Check that `crps_cdf_brier_decomposition` raises exceptions as expected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        crps_cdf_brier_decomposition(fcst, obs, threshold_dim, fcst_fill_method=fcst_fill_method, reduce_dims=dims)


@pytest.mark.parametrize(
    ("dims", "expected"),
    [
        (None, crps_test_data.EXP_CRPS_BD1),
        (["station"], crps_test_data.EXP_CRPS_BD2),
    ],
)
def test_crps_cdf_brier_decomposition(dims, expected):
    """Tests `crps_cdf_brier_decomposition` with a variety of inputs."""
    result = crps_cdf_brier_decomposition(
        crps_test_data.DA_FCST_CRPS_BD, crps_test_data.DA_OBS_CRPS_BD, "x", preserve_dims=dims
    )
    assert_dataset_equal(result, expected, decimals=7)
