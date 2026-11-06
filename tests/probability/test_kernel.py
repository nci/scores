"""Tests for the core kernel functions."""

import numpy as np
import pytest
import xarray as xr

from scores.probability.kernel import (
    KernelTerms,
    _check_estimator,
    _combine_kernel_terms,
    _crps_terms,
    _spread_denominator,
    _spread_term,
)

EXP_SPREAD_DENOM_FAIR = xr.DataArray(np.array([np.nan, 4, np.nan]), dims=["station"], coords={"station": [0, 1, 2]})
EXP_SPREAD_DENOM_ECDF = xr.DataArray(np.array([np.nan, 8, np.nan]), dims=["station"], coords={"station": [0, 1, 2]})
EXP_SPREAD_TERM_ECDF = xr.DataArray(
    np.array([np.nan, np.nan, 1 / 8, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]}
)
EXP_SPREAD_TERM_FAIR = xr.DataArray(
    np.array([np.nan, np.nan, 1 / 4, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]}
)


@pytest.mark.parametrize(("method", "expected"), [("ecdf", EXP_SPREAD_DENOM_ECDF), ("fair", EXP_SPREAD_DENOM_FAIR)])
def test__spread_denominator(method, expected):
    """Tests _spread_denominator."""
    n_members = xr.DataArray(np.array([0, 2, np.nan]), dims=["station"], coords={"station": [0, 1, 2]})
    result = _spread_denominator(n_members, method)
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(("method", "expected"), [("ecdf", EXP_SPREAD_TERM_ECDF), ("fair", EXP_SPREAD_TERM_FAIR)])
def test_spread_term(method, expected):
    """Tests _spread_term."""

    # obs_term not used in _spread_term, so we can set it to any value
    terms = KernelTerms(
        obs_term=xr.DataArray(np.array([0, 1, 2, 3]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
        spread_sum=xr.DataArray(np.array([1, 2, 1, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
        n_members=xr.DataArray(np.array([0, np.nan, 2, 2]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
    )
    result = _spread_term(terms, method)
    xr.testing.assert_equal(result, expected)


def test__combine_kernel_terms():
    """Tests _combine_kernel_terms."""
    terms = KernelTerms(
        obs_term=xr.DataArray(np.array([1, 2, 3, 4]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
        spread_sum=xr.DataArray(np.array([1, 2, 1, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
        n_members=xr.DataArray(np.array([0, np.nan, 2, 2]), dims=["station"], coords={"station": [0, 1, 2, 3]}),
    )
    result_ecdf = _combine_kernel_terms(terms, "ecdf")
    result_fair = _combine_kernel_terms(terms, "fair")
    expected_ecdf = terms.obs_term - EXP_SPREAD_TERM_ECDF
    expected_fair = terms.obs_term - EXP_SPREAD_TERM_FAIR
    xr.testing.assert_equal(result_ecdf, expected_ecdf)
    xr.testing.assert_equal(result_fair, expected_fair)


def test__check_estimator():
    """Tests _check_estimator."""
    with pytest.raises(ValueError):
        _check_estimator("invalid")


def test__crps_terms():
    """
    Tests _crps_terms.

    Note that this is more thoroughly tested in test_crps.py
    """
    fcst = xr.DataArray(
        np.array([[0, 2, np.nan], [1, np.nan, np.nan]]),
        dims=["ens_member", "station"],
        coords={"station": [0, 1, 2], "ens_member": [0, 1]},
    )
    obs = xr.DataArray(np.array([0, 1, 2]), dims=["station"], coords={"station": [0, 1, 2]})
    result = _crps_terms(fcst, obs, "ens_member")

    expected = KernelTerms(
        obs_term=xr.DataArray(np.array([0.5, 1, np.nan]), dims=["station"], coords={"station": [0, 1, 2]}),
        spread_sum=xr.DataArray(np.array([2, 0, 0]), dims=["station"], coords={"station": [0, 1, 2]}),
        n_members=xr.DataArray(np.array([2, 1, 0]), dims=["station"], coords={"station": [0, 1, 2]}),
    )
    assert len(result) == 3
    xr.testing.assert_equal(result.obs_term, expected.obs_term)
    xr.testing.assert_equal(result.spread_sum, expected.spread_sum)
    xr.testing.assert_equal(result.n_members, expected.n_members)
