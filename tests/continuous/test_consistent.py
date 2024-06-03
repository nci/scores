"""
Tests for scores.continuous.consistent_impl
"""

import pytest
import xarray as xr
from numpy import nan
from xarray.testing import assert_allclose

from scores.continuous.consistent_impl import (
    check_alpha,
    check_huber_param,
    consistent_expectile_score,
    consistent_huber_score,
    consistent_quantile_score,
)

DA_FCST = xr.DataArray(
    data=[[3.0, 1.0, nan], [-4.0, 0.0, 1.0]],
    dims=["date", "station"],
    coords=dict(date=["1", "2"], station=[100, 101, 102]),
)

DA_OBS = xr.DataArray(
    data=[[nan, 3.0, 5.0], [-4.0, 10.0, -1.0], [3.0, 2.0, -0.2]],
    dims=["date", "station"],
    coords=dict(date=["1", "2", "3"], station=[100, 101, 102]),
)

ALPHA = 0.3

EXP_EXPECTILE_SCORE1 = xr.DataArray(
    data=[[nan, ALPHA * 4.0, nan], [0.0, ALPHA * 100, (1 - ALPHA) * 4]],
    dims=["date", "station"],
    coords=dict(date=["1", "2"], station=[100, 101, 102]),
)

EXP_EXPECTILE_SCORE2 = xr.DataArray(
    data=[0.0, ALPHA * (4 + 100) / 2, (1 - ALPHA) * 4],
    dims=["station"],
    coords=dict(station=[100, 101, 102]),
)

TUNING_PARAM = 2.0

EXP_HUBER_SCORE1 = xr.DataArray(
    data=[[nan, 2.0, nan], [0.0, 18.0, 2.0]],
    dims=["date", "station"],
    coords=dict(date=["1", "2"], station=[100, 101, 102]),
)

EXP_HUBER_SCORE2 = xr.DataArray(
    data=[2.0, 20 / 3],
    dims=["date"],
    coords=dict(date=["1", "2"]),
)

EXP_HUBER_SCORE3 = xr.DataArray(
    data=22 / 4,
)

EXP_QUANTILE_SCORE1 = xr.DataArray(
    data=[[nan, ALPHA * 2.0, nan], [0.0, ALPHA * 10, (1 - ALPHA) * 2]],
    dims=["date", "station"],
    coords=dict(date=["1", "2"], station=[100, 101, 102]),
)

EXP_QUANTILE_SCORE2 = xr.DataArray(
    data=[ALPHA * 2.0, (ALPHA * 10 + (1 - ALPHA) * 2) / 3],
    dims=["date"],
    coords=dict(date=["1", "2"]),
)


def squared_loss(x):
    """squared loss function"""
    return x**2


def squared_loss_prime(x):
    """derivative of `squared_loss`"""
    return 2 * x


def simple_linear(x):
    """simple increasing function"""
    return x


@pytest.mark.parametrize(
    ("preserve_dims", "expected"),
    [
        (["date", "station"], EXP_EXPECTILE_SCORE1),
        (["station"], EXP_EXPECTILE_SCORE2),
    ],
)
def test_consistent_expectile_score(preserve_dims, expected):
    """Tests that `consistent_expectile_score` gives results as expected."""
    result = consistent_expectile_score(
        DA_FCST, DA_OBS, alpha=ALPHA, phi=squared_loss, phi_prime=squared_loss_prime, preserve_dims=preserve_dims
    )
    assert_allclose(result, expected)


@pytest.mark.parametrize("alpha", [1.0, 0.0])
def test_check_alpha(alpha):
    """Tests that `check_alpha` raises as expected."""
    with pytest.raises(ValueError, match="`alpha` must be strictly between 0 and 1"):
        check_alpha(alpha)


def test_check_huber_param():
    """Tests that `check_huber_param` raises as expected."""
    with pytest.raises(ValueError, match="`huber_param` must be positive"):
        check_huber_param(0.0)


@pytest.mark.parametrize(
    ("preserve_dims", "expected"),
    [
        (["date", "station"], EXP_HUBER_SCORE1),
        (["date"], EXP_HUBER_SCORE2),
        (None, EXP_HUBER_SCORE3),
    ],
)
def test_consistent_huber_score(preserve_dims, expected):
    """Tests that `consistent_huber_score` gives results as expected."""
    result = consistent_huber_score(
        DA_FCST,
        DA_OBS,
        huber_param=TUNING_PARAM,
        phi=squared_loss,
        phi_prime=squared_loss_prime,
        preserve_dims=preserve_dims,
    )
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("preserve_dims", "expected"),
    [(["date", "station"], EXP_QUANTILE_SCORE1), (["date"], EXP_QUANTILE_SCORE2)],
)
def test_consistent_quantile_score(preserve_dims, expected):
    """Tests that `consistent_quantile_score` gives results as expected."""
    result = consistent_quantile_score(DA_FCST, DA_OBS, alpha=ALPHA, g=simple_linear, preserve_dims=preserve_dims)
    assert_allclose(result, expected)
