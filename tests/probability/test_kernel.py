"""Tests for the core kernel functions."""

import xarray as xr
import numpy as np

from scores.probability.kernel import _spread_denominator, _spread_term, KernelTerms

import pytest

EXP_SPREAD_DENOM_FAIR = xr.DataArray(np.array([np.nan, 4, np.nan]), dims=["station"], coords={"station": [0, 1, 2]})
EXP_SPREAD_DENOM_ECDF = xr.DataArray(np.array([np.nan, 8, np.nan]), dims=["station"], coords={"station": [0, 1, 2]})
EXP_SPREAD_TERM_ECDF = xr.DataArray(np.array([np.nan, np.nan, 1/8, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]})
EXP_SPREAD_TERM_FAIR = xr.DataArray(np.array([np.nan, np.nan, 1/4, np.nan]), dims=["station"], coords={"station": [0, 1, 2, 3]})

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
        n_members=xr.DataArray(np.array([0, np.nan, 2, 2]), dims=["station"], coords={"station": [0, 1, 2, 3]})
    )
    result = _spread_term(terms, method)
    xr.testing.assert_equal(result, expected)
