# pylint disable: too-many-arguments
"""
Contains unit tests for scores.probability.crps
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.probability import (
    adjust_fcst_for_crps,
    crps_cdf,
    crps_cdf_brier_decomposition,
    crps_for_ensemble,
    interval_tw_crps_for_ensemble,
    tail_tw_crps_for_ensemble,
    tw_crps_for_ensemble,
)
from scores.probability.crps_impl import (
    crps_cdf_exact,
    crps_cdf_reformat_inputs,
    crps_cdf_trapz,
    crps_step_threshold_weight,
)
from tests.probabilty import crps_test_data


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
        threshold_values=[1, 2, 3, 4, 5, 6],
        steppoints_in_thresholds=True,
        steppoint_precision=0.2,
        weight_upper=weight_upper,
    )
    xr.testing.assert_allclose(result, expected)


def test_crps_cdf_exact():
    """Tests `crps_cdf_exact`."""
    result = crps_cdf_exact(
        crps_test_data.DA_FCST_CRPS_EXACT,
        crps_test_data.DA_OBS_CRPS_EXACT,
        crps_test_data.DA_WT_CRPS_EXACT,
        "x",
        include_components=True,
    )
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPS_EXACT)

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

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

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
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPS_EXACT)

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


def test_crps_cdf_trapz():
    """Tests `crps_cdf_trapz`. Uses dense interpolation to get an approximate (to 4 dec pl) result."""
    result = crps_cdf_trapz(
        crps_test_data.DA_FCST_CRPS_DENSE,
        crps_test_data.DA_OBS_CRPS_DENSE,
        crps_test_data.DA_WT_CRPS_DENSE,
        "x",
        include_components=True,
    )
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPS_EXACT, atol=4)

    result2 = crps_cdf_trapz(
        crps_test_data.DA_FCST_CRPS_DENSE,
        crps_test_data.DA_OBS_CRPS_DENSE,
        crps_test_data.DA_WT_CRPS_DENSE,
        "x",
        include_components=False,
    )
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPS_EXACT, atol=4)
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
        threshold_weight=threshold_weight,
        additional_thresholds=additional_thresholds,
        fcst_fill_method="linear",
        threshold_weight_fill_method="forward",
    )
    assert len(result) == len(expected)

    for res, exp in zip(result, expected):
        xr.testing.assert_allclose(res, exp)


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
        # FIXLATER: Revisit if still needed after more handling in gather_dimensions
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
            threshold_dim=threshold_dim,
            threshold_weight=threshold_weight,
            additional_thresholds=[],
            propagate_nans=True,
            fcst_fill_method=fcst_fill_method,
            threshold_weight_fill_method=threshold_weight_fill_method,
            integration_method=integration_method,
            preserve_dims=dims,
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
        threshold_dim="x",
        threshold_weight=threshold_weight,
        additional_thresholds=None,
        propagate_nans=propagate_nan,
        fcst_fill_method="linear",
        threshold_weight_fill_method="forward",
        integration_method=integration_method,
        preserve_dims=dims,
        include_components=True,
    )
    xr.testing.assert_allclose(result, expected_and_dec[0], atol=expected_and_dec[1])


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
    result = adjust_fcst_for_crps(fcst, "x", obs, decreasing_tolerance=decreasing_tolerance)
    xr.testing.assert_allclose(result, expected)


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
            decreasing_tolerance=decreasing_tolerance,
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
        crps_cdf_brier_decomposition(
            fcst, obs, threshold_dim=threshold_dim, fcst_fill_method=fcst_fill_method, reduce_dims=dims
        )


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
        crps_test_data.DA_FCST_CRPS_BD, crps_test_data.DA_OBS_CRPS_BD, threshold_dim="x", preserve_dims=dims
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("fcst", "obs", "method", "weight", "preserve_dims", "decomposition", "expected"),
    [
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            None,
            "all",
            False,
            crps_test_data.EXP_CRPSENS_ECDF,
        ),
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "fair",
            None,
            "all",
            False,
            crps_test_data.EXP_CRPSENS_FAIR,
        ),
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            crps_test_data.DA_WT_CRPSENS,
            None,
            False,
            crps_test_data.EXP_CRPSENS_WT,
        ),
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            None,
            "all",
            False,
            crps_test_data.EXP_CRPSENS_ECDF_DS,
        ),
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "fair",
            None,
            "all",
            False,
            crps_test_data.EXP_CRPSENS_FAIR_DS,
        ),
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            crps_test_data.DS_WT_CRPSENS,
            None,
            False,
            crps_test_data.EXP_CRPSENS_WT_DS,
        ),
        (
            crps_test_data.DA_FCST_CRPSENS_LT,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            None,
            "all",
            False,
            crps_test_data.EXP_CRPSENS_ECDF_BC,
        ),
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            None,
            "all",
            True,
            crps_test_data.EXP_CRPSENS_ECDF_DECOMPOSITION,
        ),
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            None,
            "all",
            True,
            crps_test_data.EXP_CRPSENS_ECDF_DECOMPOSITION_DS,
        ),
    ],
)
def test_crps_for_ensemble(fcst, obs, method, weight, preserve_dims, decomposition, expected):
    """Tests `crps_for_ensemble` returns as expected."""
    result = crps_for_ensemble(
        fcst,
        obs,
        "ens_member",
        method=method,
        weights=weight,
        preserve_dims=preserve_dims,
        decomposition=decomposition,
    )
    xr.testing.assert_allclose(result, expected)


def test_crps_for_ensemble_raises():
    """Tests `crps_for_ensemble` raises exception as expected."""
    with pytest.raises(ValueError) as excinfo:
        crps_for_ensemble(
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ens_member",
            method="unfair",
        )
    assert "`method` must be one of 'ecdf' or 'fair'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("fcst", "obs"),
    [
        (crps_test_data.DA_FCST_CRPSENS.chunk(), crps_test_data.DA_OBS_CRPSENS.chunk()),
        (crps_test_data.DA_FCST_CRPSENS, crps_test_data.DA_OBS_CRPSENS.chunk()),
        (crps_test_data.DA_FCST_CRPSENS.chunk(), crps_test_data.DA_OBS_CRPSENS),
    ],
)
def test_crps_for_ensemble_dask(fcst, obs):
    """Tests `crps_for_ensemble` works with dask."""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = crps_for_ensemble(
        fcst=fcst,
        obs=obs,
        ensemble_member_dim="ens_member",
        method="ecdf",
        preserve_dims="all",
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPSENS_ECDF)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "method",
        "tail",
        "threshold",
        "preserve_dims",
        "reduce_dims",
        "weights",
        "decomposition",
        "expected",
    ),
    [
        # Test ECDF upper
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "upper",
            1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DA,
        ),
        # Test broadcasting
        (
            crps_test_data.DA_FCST_CRPSENS_LT,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "upper",
            1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_BC,
        ),
        # Test fair lower
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "fair",
            "upper",
            1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_FAIR_DA,
        ),
        # Test ECDF lower
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "lower",
            1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_LOWER_TAIL_CRPSENS_ECDF_DA,
        ),
        # Test fair lower
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "fair",
            "lower",
            1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_LOWER_TAIL_CRPSENS_FAIR_DA,
        ),
        # test that it equals the standard CRPS when the tail contains all threshold
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "lower",
            np.inf,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_CRPSENS_ECDF,
        ),
        # test that both the weights and reduce dims args work
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "lower",
            np.inf,
            None,
            "stn",
            crps_test_data.DA_WT_CRPSENS,
            False,
            crps_test_data.EXP_CRPSENS_WT,
        ),
        # test that passing an xarray object for the threshold arg works
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "upper",
            crps_test_data.DA_T_TWCRPSENS,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DA,
        ),
        # test that passing in xr.Datasets with an xr.Dataset for the threshold arg works
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            "upper",
            crps_test_data.DS_T_TWCRPSENS,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DS,
        ),
        # test that passing in xr.Datasets with an xr.DataArray for the threshold arg works
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            "upper",
            crps_test_data.DA_T_TWCRPSENS,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DS,
        ),
        # Test decomposition with DataArrays
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            "upper",
            1,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA,
        ),
        # Test decomposition with Datasets
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            "upper",
            1,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DS,
        ),
    ],
)
def test_tail_tw_crps_for_ensemble(
    fcst, obs, method, tail, threshold, preserve_dims, reduce_dims, weights, decomposition, expected
):
    """Tests tail_tw_crps_for_ensembles"""
    result = tail_tw_crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim="ens_member",
        threshold=threshold,
        method=method,
        tail=tail,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        weights=weights,
        decomposition=decomposition,
    )
    xr.testing.assert_allclose(result, expected)


def test_tail_tw_crps_for_ensemble_dask():
    """Tests `tail_tw_crps_for_ensemble` works with dask."""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    # Check that it works with xr.Datarrays
    result = tail_tw_crps_for_ensemble(
        fcst=crps_test_data.DA_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DA_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        threshold=1,
        method="ecdf",
        tail="upper",
        preserve_dims="all",
        reduce_dims=None,
        weights=None,
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DA)

    # Check that it works with xr.Datasets
    result_ds = tail_tw_crps_for_ensemble(
        fcst=crps_test_data.DS_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DS_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        threshold=1,
        method="ecdf",
        tail="upper",
        preserve_dims="all",
        reduce_dims=None,
        weights=None,
    )
    assert isinstance(result_ds["a"].data, dask.array.Array)
    result_ds = result_ds.compute()
    assert isinstance(result_ds["a"].data, np.ndarray)
    xr.testing.assert_allclose(result_ds, crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DS)


def test_tail_tw_crps_for_ensemble_raises():
    with pytest.raises(ValueError, match="'middle' is not one of 'upper' or 'lower'"):
        result = tail_tw_crps_for_ensemble(
            fcst=crps_test_data.DA_FCST_CRPSENS,
            obs=crps_test_data.DA_OBS_CRPSENS,
            ensemble_member_dim="ens_member",
            threshold=1,
            method="ecdf",
            tail="middle",
        )


def v_func1(x):
    """For testing tw_crps_for_ensembles. The equivalent of a tail weight for thresholds 1 and higher"""
    return np.maximum(x, 1)


def v_func2(x):
    """For testing tw_crps_for_ensembles. The equivalent of the unweighted CRPS"""
    return x


def v_func3(x):
    """For testing tw_crps_for_ensembles. The equivalent of a tail weight for thresholds that vary across a dimension"""
    return np.maximum(x, crps_test_data.DA_T_TWCRPSENS)


def v_func4(x):
    """For testing tw_crps_for_ensembles. The equivalent of a tail weight for thresholds that vary across a dimension with a xr.dataset"""
    return np.maximum(x, crps_test_data.DS_T_TWCRPSENS)


@pytest.mark.parametrize(
    ("fcst", "obs", "method", "v_func", "preserve_dims", "reduce_dims", "weights", "decomposition", "expected"),
    [
        # test ecdf
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DA,
        ),
        # test broadcast
        (
            crps_test_data.DA_FCST_CRPSENS_LT,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_BC,
        ),
        # test fair
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "fair",
            v_func1,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_FAIR_DA,
        ),
        # test that it equals the standard CRPS when the tail contains all threshold
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func2,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_CRPSENS_ECDF,
        ),
        # # test that both the weights and reduce dims args work
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func2,
            None,
            "stn",
            crps_test_data.DA_WT_CRPSENS,
            False,
            crps_test_data.EXP_CRPSENS_WT,
        ),
        # test that it works when threshold vary across a dimension
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func3,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DA,
        ),
        # test that it works when threshold vary across a dimension with xr.Datasets
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            v_func4,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DS,
        ),
        # test that it works when threshold vary across a dimension with xr.Dataset, except
        # with an xr.DataArray in the v_func
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            v_func3,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_THRES_CRPSENS_DS,
        ),
        # Test decomposition with DataArrays
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            v_func1,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DA,
        ),
        # Test decomposition with Datasets
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            v_func1,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DECOMP_DS,
        ),
    ],
)
def test_tw_crps_for_ensemble(fcst, obs, method, v_func, preserve_dims, reduce_dims, weights, decomposition, expected):
    """Tests tw_crps_for_ensembles"""

    result = tw_crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim="ens_member",
        chaining_func=v_func,
        method=method,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        weights=weights,
        decomposition=decomposition,
    )
    xr.testing.assert_allclose(result, expected)


def test_tw_crps_for_ensemble_dask():
    """Tests `tw_crps_for_ensemble` works with dask."""
    result = tw_crps_for_ensemble(
        fcst=crps_test_data.DA_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DA_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        chaining_func=v_func1,
        method="ecdf",
        preserve_dims="all",
        reduce_dims=None,
        weights=None,
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DA)

    # Check that it works with xr.Datasets
    result_ds = tw_crps_for_ensemble(
        fcst=crps_test_data.DS_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DS_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        chaining_func=v_func1,
        method="ecdf",
        preserve_dims="all",
        reduce_dims=None,
        weights=None,
    )
    assert isinstance(result_ds["a"].data, dask.array.Array)
    result_ds = result_ds.compute()
    assert isinstance(result_ds["a"].data, np.ndarray)
    xr.testing.assert_allclose(result_ds, crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF_DS)


@pytest.mark.parametrize(
    (
        "fcst",
        "obs",
        "method",
        "lower_threshold",
        "upper_threshold",
        "preserve_dims",
        "reduce_dims",
        "weights",
        "decomposition",
        "expected",
    ),
    [
        # Test interval
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            2,
            5,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DA,
        ),
        # test that it equals the standard CRPS when the interval contains all threshold with ecdf
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            -np.inf,
            np.inf,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_CRPSENS_ECDF,
        ),
        # test that it equals the standard CRPS when the interval contains all threshold with fair
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "fair",
            -np.inf,
            np.inf,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_CRPSENS_FAIR,
        ),
        # Test broadcast
        (
            crps_test_data.DA_FCST_CRPSENS_LT,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            -np.inf,
            np.inf,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_CRPSENS_ECDF_BC,
        ),
        # Test with weights and reduce dims
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            -np.inf,
            np.inf,
            None,
            "stn",
            crps_test_data.DA_WT_CRPSENS,
            False,
            crps_test_data.EXP_CRPSENS_WT,
        ),
        # test that passing an xarray object for the threshold args works
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            crps_test_data.DA_LI_TWCRPSENS,
            crps_test_data.DA_UI_TWCRPSENS,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_VAR_INTERVAL_CRPSENS_ECDF_DA,
        ),
        # Test that a float for lower_threshold and an xr.DataArray for upper_threshold works
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            2,
            crps_test_data.DA_UI_CONS_TWCRPSENS,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DA,
        ),
        # Test that an xr.DataArray for lower_threshold and a float for upper_threshold works
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            crps_test_data.DA_LI_CONS_TWCRPSENS,
            5,
            "all",
            None,
            None,
            False,
            crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DA,
        ),
        # test that passing in xr.Datasets for fcst, obs and weights works
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            -np.inf,
            np.inf,
            None,
            None,
            crps_test_data.DS_WT_CRPSENS,
            False,
            crps_test_data.EXP_CRPSENS_WT_DS,
        ),
        # Test decomposition with DataArrays
        (
            crps_test_data.DA_FCST_CRPSENS,
            crps_test_data.DA_OBS_CRPSENS,
            "ecdf",
            2,
            5,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DA,
        ),
        # Test decomposition with Datasets
        (
            crps_test_data.DS_FCST_CRPSENS,
            crps_test_data.DS_OBS_CRPSENS,
            "ecdf",
            2,
            5,
            "all",
            None,
            None,
            True,
            crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DECOMP_DS,
        ),
    ],
)
def test_interval_tw_crps_for_ensemble(
    fcst, obs, method, lower_threshold, upper_threshold, preserve_dims, reduce_dims, weights, decomposition, expected
):
    """Tests interval_tw_crps_for_ensembles"""
    result = interval_tw_crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim="ens_member",
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        method=method,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        weights=weights,
        decomposition=decomposition,
    )

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("lower_threshold", "upper_threshold"),
    [
        (1, 1),
        (2, 1),
        (xr.DataArray(data=[1, np.nan]), xr.DataArray(data=[1, np.nan])),
        (xr.DataArray(data=[2, np.nan]), xr.DataArray(data=[1, np.nan])),
        (1, xr.DataArray(data=[1, np.nan])),
        (xr.DataArray(data=[1, np.nan]), 1),
    ],
)
def test_interval_tw_crps_for_ensemble_raises(lower_threshold, upper_threshold):
    """Tests if interval_tw_crps_for_ensemble raises an error when lower_threshold >= upper_threshold"""
    with pytest.raises(ValueError) as excinfo:
        interval_tw_crps_for_ensemble(
            fcst=crps_test_data.DA_FCST_CRPSENS,
            obs=crps_test_data.DA_OBS_CRPSENS,
            ensemble_member_dim="ens_member",
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            method="ecdf",
        )
    assert "`lower_threshold` must be less than `upper_threshold`" in str(excinfo.value)


def test_interval_tw_crps_for_ensemble_dask():
    """Tests `interval_tw_crps_for_ensemble` works with dask."""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    # Check that it works with xr.Datarrays
    result = interval_tw_crps_for_ensemble(
        fcst=crps_test_data.DA_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DA_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        lower_threshold=2,
        upper_threshold=5,
        method="ecdf",
        preserve_dims="all",
        reduce_dims=None,
        weights=None,
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, crps_test_data.EXP_INTERVAL_CRPSENS_ECDF_DA)

    # Check that it works with xr.Datasets
    result_ds = interval_tw_crps_for_ensemble(
        fcst=crps_test_data.DS_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DS_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        lower_threshold=-np.inf,
        upper_threshold=np.inf,
        method="ecdf",
        preserve_dims=None,
        reduce_dims=None,
        weights=crps_test_data.DS_WT_CRPSENS.chunk(),
    )
    assert isinstance(result_ds["a"].data, dask.array.Array)
    result_ds = result_ds.compute()
    assert isinstance(result_ds["a"].data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result_ds, crps_test_data.EXP_CRPSENS_WT_DS)
