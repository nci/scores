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
    tail_twcrps_for_ensemble,
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


def test_crps_for_ensemble():
    """Tests `crps_for_ensemble` returns as expected."""
    result_ecdf = crps_for_ensemble(
        crps_test_data.DA_FCST_CRPSENS, crps_test_data.DA_OBS_CRPSENS, "ens_member", method="ecdf", preserve_dims="all"
    )
    result_fair = crps_for_ensemble(
        crps_test_data.DA_FCST_CRPSENS, crps_test_data.DA_OBS_CRPSENS, "ens_member", method="fair", preserve_dims="all"
    )
    result_weighted_mean = crps_for_ensemble(
        crps_test_data.DA_FCST_CRPSENS,
        crps_test_data.DA_OBS_CRPSENS,
        "ens_member",
        method="ecdf",
        weights=crps_test_data.DA_WT_CRPSENS,
    )
    xr.testing.assert_allclose(result_ecdf, crps_test_data.EXP_CRPSENS_ECDF)
    xr.testing.assert_allclose(result_fair, crps_test_data.EXP_CRPSENS_FAIR)
    xr.testing.assert_allclose(result_weighted_mean, crps_test_data.EXP_CRPSENS_WT)


def test_crps_for_ensemble_raises():
    """Tests `crps_for_ensemble` raises exception as expected."""
    with pytest.raises(ValueError) as excinfo:
        crps_for_ensemble(xr.DataArray(data=[1]), xr.DataArray(data=[1]), "ens_member", method="unfair")
    assert "`method` must be one of 'ecdf' or 'fair'" in str(excinfo.value)


def test_crps_for_ensemble_dask():
    """Tests `crps_for_ensemble` works with dask."""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = crps_for_ensemble(
        fcst=crps_test_data.DA_FCST_CRPSENS.chunk(),
        obs=crps_test_data.DA_OBS_CRPSENS.chunk(),
        ensemble_member_dim="ens_member",
        method="ecdf",
        preserve_dims="all",
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, crps_test_data.EXP_CRPSENS_ECDF)


@pytest.mark.parametrize(
    ("method", "tail", "threshold", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        ("ecdf", "upper", 1, "all", None, None, crps_test_data.EXP_UPPER_TAIL_CRPSENS_ECDF),
        ("fair", "upper", 1, "all", None, None, crps_test_data.EXP_UPPER_TAIL_CRPSENS_FAIR),
        ("ecdf", "lower", 1, "all", None, None, crps_test_data.EXP_LOWER_TAIL_CRPSENS_ECDF),
        ("fair", "lower", 1, "all", None, None, crps_test_data.EXP_LOWER_TAIL_CRPSENS_FAIR),
        # test that it equals the standard CRPS when the tail contains all threshold
        ("ecdf", "lower", np.inf, "all", None, None, crps_test_data.EXP_CRPSENS_ECDF),
        # test that both the weights and reduce dims args work
        ("ecdf", "lower", np.inf, None, "stn", crps_test_data.DA_WT_CRPSENS, crps_test_data.EXP_CRPSENS_WT),
        # test that passing an xarray object for the threshold arg works
        ("ecdf", "upper", crps_test_data.DA_T_TWCRPSENS, "all", None, None, crps_test_data.EXP_VAR_THRES_CRPSENS),
    ],
)
def test_tail_twcrps_for_ensemble(method, tail, threshold, preserve_dims, reduce_dims, weights, expected):
    """Tests tail_twcrps_for_ensembles"""
    result = tail_twcrps_for_ensemble(
        fcst=crps_test_data.DA_FCST_CRPSENS,
        obs=crps_test_data.DA_OBS_CRPSENS,
        ensemble_member_dim="ens_member",
        threshold=threshold,
        method=method,
        tail=tail,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(result, expected)


def test_tail_twcrps_for_ensemble_raises():
    with pytest.raises(ValueError, match="'middle' is not one of 'upper' or 'lower'"):
        result = tail_twcrps_for_ensemble(
            fcst=crps_test_data.DA_FCST_CRPSENS,
            obs=crps_test_data.DA_OBS_CRPSENS,
            ensemble_member_dim="ens_member",
            threshold=1,
            method="ecdf",
            tail="middle",
        )
