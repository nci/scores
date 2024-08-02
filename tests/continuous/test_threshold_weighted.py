# pylint: disable=use-dict-literal
"""
Tests for scores.continuous.threshold_weighted_impl
"""

import dask
import numpy as np
import pytest
import xarray as xr
from numpy import nan

from scores.continuous import mae, mse, quantile_score
from scores.continuous.threshold_weighted_impl import (
    _auxiliary_funcs,
    _g_j_rect,
    _g_j_trap,
    _phi_j_prime_rect,
    _phi_j_prime_trap,
    _phi_j_rect,
    _phi_j_trap,
    tw_absolute_error,
    tw_expectile_score,
    tw_huber_loss,
    tw_quantile_score,
    tw_squared_error,
)

DA_FCST = xr.DataArray(
    data=[[[3.0, 1.0, nan, 2], [3.0, 1.0, nan, 2]], [[-4.0, 0.0, 1.0, 2], [-4.0, 0.0, 1.0, 2]]],
    dims=["date", "lead_day", "station"],
    coords=dict(
        date=["1", "2"],
        lead_day=[1, 1],
        station=[100, 101, 102, 0],
    ),
)

DA_OBS = xr.DataArray(
    data=[[nan, 3.0, 5.0], [-4.0, 10.0, -1.0], [3.0, 2.0, -0.2]],
    dims=["date", "station"],
    coords=dict(date=["1", "2", "3"], station=[100, 101, 102]),
)

DA_X1 = xr.DataArray([nan, -2.0, 1.0, 5.0])

EXP_G_J_RECT1 = xr.DataArray([nan, 0.0, 2.0, 3.0])  # a = -1, b = 2

DA_X2 = xr.DataArray(
    data=[[nan, -2.0, 1.0, 11.0]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101, 102, 103]),
)

DA_A = xr.DataArray(
    data=[-1.0, 0.0, -1.0, 10.0],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103]),
)

DA_B = xr.DataArray(
    data=[2.0, 2.0, 0.0, 12.0],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103]),
)

EXP_G_J_RECT2 = xr.DataArray(
    data=[[nan, 0.0, 1.0, 1.0]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101, 102, 103]),
)

EXP_PHI_J_RECT1 = xr.DataArray([nan, 0.0, 8.0, 54.0])  # a = -1, b = 2

EXP_PHI_J_RECT2 = xr.DataArray(
    data=[[nan, 0.0, 6.0, 2.0]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101, 102, 103]),
)

EXP_PHI_J_PRIME_RECT1 = 4 * EXP_G_J_RECT1

DA_X3 = xr.DataArray([nan, -3.0, 0.0, 2.0, 5.0, 10.0])
# s = 5
# a = -2, b = 1, c = 5, d = 8
EXP_G_J_TRAP1 = xr.DataArray([nan, 0.0, 2 / 3, 2.5, 7 - 3 / 2, 7.0])

DA_A_TRAP = xr.DataArray(
    data=[[0, 10]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

DA_B_TRAP = xr.DataArray(
    data=[[2, 20]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

DA_C_TRAP = xr.DataArray(
    data=[[5, 25]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

DA_D_TRAP = xr.DataArray(
    data=[[6, 30]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

DA_X_TRAP = xr.DataArray(
    data=[[3, 15]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

EXP_G_J_TRAP2 = xr.DataArray(
    data=[[2.0, 1.25]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

EXP_PHI_J_TRAP1 = xr.DataArray([nan, 0.0, 16 / 9, 14, 6 + 140 - 2 * 126 / 3, 280 - 2 * 126 / 3])

EXP_PHI_J_TRAP2 = xr.DataArray(
    data=[[6 + 8 / 3, 25 / 3]],
    dims=["date", "station"],
    coords=dict(date=["01"], station=[100, 101]),
)

EXP_PHI_J_PRIME_TRAP = 4 * EXP_G_J_TRAP1

DA_A_INF = xr.DataArray([-np.inf, -1], dims=["station"], coords=dict(station=[100, 101]))

DA_B_INF = xr.DataArray([-100, np.inf], dims=["station"], coords=dict(station=[100, 101]))

DA_A_FINITE = xr.DataArray([-101, -1], dims=["station"], coords=dict(station=[100, 101]))

DA_B_FINITE = xr.DataArray([-100, 11], dims=["station"], coords=dict(station=[100, 101]))


DA_FCST1 = xr.DataArray(
    data=[[3.0, 1.0, nan, 1.3, 8.5], [-4.0, 0.0, 1.0, -3.6, -11.23]],
    dims=["date", "station"],
    coords=dict(date=["1", "2"], station=[100, 101, 102, 103, 104]),
)

DA_OBS1 = xr.DataArray(
    data=[
        [nan, 3.0, 5.0, 34.5, -28.1],
        [-4.0, 10.0, -1.0, 0.001, 1.3],
        [3.0, 2.0, -0.2, 1.0, nan],
    ],
    dims=["date", "station"],
    coords=dict(date=["1", "2", "3"], station=[100, 101, 102, 103, 104]),
)

DA_ENDPT1 = xr.DataArray(
    data=[0, 1, 4, -1, 5],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103, 104]),
)

DA_ENDPT2 = xr.DataArray(
    data=[2, 5, 7, 0, 10],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103, 104]),
)

DA_INF = xr.DataArray(
    data=[np.inf, np.inf, np.inf, np.inf, np.inf],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103, 104]),
)
WEIGHTS = xr.DataArray(
    data=[1, 2, np.nan, 4, 0],
    dims=["station"],
    coords=dict(station=[100, 101, 102, 103, 104]),
)
HUBER_PARAM = 0.4
# Calculated using the Jive MSHL function
EXP_HL = HUBER_PARAM * (
    xr.DataArray(
        data=[[nan, 1.8, nan, 33.0, 36.4], [0.0, 9.8, 1.8, 3.401, 12.33]],
        dims=["date", "station"],
        coords=dict(date=["1", "2"], station=[100, 101, 102, 103, 104]),
    )
)


@pytest.mark.parametrize(
    (
        "scoring_func",
        "interval_where_one",
        "interval_where_positive",
        "kwargs",
        "error_msg_snippet",
    ),
    [
        (
            tw_squared_error,
            (0, 2, 4),
            None,
            {},
            "`interval_where_one` must have length 2",
        ),
        (
            tw_squared_error,
            (0, 2),
            (-10, 10, 50),
            {},
            "`interval_where_positive` must be length 2 when not `None`",
        ),
        (
            tw_absolute_error,
            (0, 2, 4),
            None,
            {},
            "`interval_where_one` must have length 2",
        ),
        (
            tw_absolute_error,
            (0, 2),
            (-10, 10, 50),
            {},
            "`interval_where_positive` must be length 2 when not `None`",
        ),
        (
            tw_quantile_score,
            (0, 2, 4),
            None,
            {"alpha": 0.3},
            "`interval_where_one` must have length 2",
        ),
        (
            tw_quantile_score,
            (0, 2),
            (-10, 10, 50),
            {"alpha": 0.3},
            "`interval_where_positive` must be length 2 when not `None`",
        ),
        (
            tw_expectile_score,
            (0, 2, 4),
            None,
            {"alpha": 0.3},
            "`interval_where_one` must have length 2",
        ),
        (
            tw_expectile_score,
            (0, 2),
            (-10, 10, 50),
            {"alpha": 0.3},
            "`interval_where_positive` must be length 2 when not `None`",
        ),
        (
            tw_quantile_score,
            (0, 1),
            None,
            {"alpha": 0},
            "`alpha` must be strictly between 0 and 1",
        ),
        (
            tw_quantile_score,
            (0, 1),
            None,
            {"alpha": 1},
            "`alpha` must be strictly between 0 and 1",
        ),
        (
            tw_expectile_score,
            (0, 1),
            None,
            {"alpha": 0},
            "`alpha` must be strictly between 0 and 1",
        ),
        (
            tw_expectile_score,
            (0, 1),
            None,
            {"alpha": 1},
            "`alpha` must be strictly between 0 and 1",
        ),
        (
            tw_huber_loss,
            (0, 1),
            None,
            {"huber_param": 0},
            "`huber_param` must be positive",
        ),
        (
            tw_huber_loss,
            (1, 0),
            None,
            {"huber_param": 1},
            "left endpoint of `interval_where_one` must be strictly less than right endpoint",
        ),
        (
            tw_huber_loss,
            (1, 0),
            (-2, 2),
            {"huber_param": 1},
            "left endpoint of `interval_where_one` must be strictly less than right endpoint",
        ),
        (
            tw_huber_loss,
            (DA_B, DA_A),
            None,
            {"huber_param": 1},
            "left endpoint of `interval_where_one` must be strictly less than right endpoint",
        ),
        (
            tw_huber_loss,
            (DA_B, DA_A),
            None,
            {"huber_param": 1},
            "left endpoint of `interval_where_one` must be strictly less than right endpoint",
        ),
        (
            tw_huber_loss,
            (-2, 0),
            (-np.inf, 6),
            {"huber_param": 1},
            "can only be infinite when",
        ),
        (
            tw_huber_loss,
            (xr.DataArray([0, 0]), xr.DataArray([1, 1])),
            (xr.DataArray([-np.inf, -1]), xr.DataArray([2, 2])),
            {"huber_param": 1},
            "can only be infinite when",
        ),
        (
            tw_huber_loss,
            (-2, 0),
            (-4, np.inf),
            {"huber_param": 1},
            "can only be infinite when",
        ),
        (
            tw_huber_loss,
            (xr.DataArray([0, 0]), xr.DataArray([1, 1])),
            (xr.DataArray([-1, -1]), xr.DataArray([np.inf, 2])),
            {"huber_param": 1},
            "can only be infinite when",
        ),
        (
            tw_huber_loss,
            (0, 1),
            (0, 2),
            {"huber_param": 1},
            "left endpoint of `interval_where_positive` must be less than",
        ),
        (
            tw_huber_loss,
            (xr.DataArray([0, 1]), xr.DataArray([20, 10])),
            (xr.DataArray([-1, 1]), xr.DataArray([22, 11])),
            {"huber_param": 1},
            "left endpoint of `interval_where_positive` must be less than",
        ),
        (
            tw_huber_loss,
            (0, 2),
            (-1, 2),
            {"huber_param": 1},
            "right endpoint of `interval_where_positive` must be greater than",
        ),
        (
            tw_huber_loss,
            (xr.DataArray([0, 0]), xr.DataArray([2, 2])),
            (xr.DataArray([-1, -1]), xr.DataArray([2, 2])),
            {"huber_param": 1},
            "right endpoint of `interval_where_positive` must be greater than",
        ),
    ],
)
def test_threshold_weighted_score_raises(
    scoring_func,
    interval_where_one,
    interval_where_positive,
    kwargs,
    error_msg_snippet,
):
    """Tests that the threshold weighted scoring rules raise errors as exepected."""
    with pytest.raises(ValueError, match=error_msg_snippet):
        scoring_func(
            DA_FCST,
            DA_OBS,
            interval_where_one=interval_where_one,
            interval_where_positive=interval_where_positive,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("a", "b", "x", "expected"),
    [
        (-1.0, 2.0, DA_X1, EXP_G_J_RECT1),  # a, b float; all cases tested
        (DA_A, DA_B, DA_X2, EXP_G_J_RECT2),  # a, b array
    ],
)
def test__g_j_rect(a, b, x, expected):
    """Tests that `_g_j_rect` gives results as expected."""
    result = _g_j_rect(a, b, x)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("a", "b", "x", "expected"),
    [
        (-1.0, 2.0, DA_X1, EXP_PHI_J_RECT1),  # a, b float; all cases tested
        (DA_A, DA_B, DA_X2, EXP_PHI_J_RECT2),  # a, b array
    ],
)
def test__phi_j_rect(a, b, x, expected):
    """Tests that `_phi_j_rect` gives results as expected."""
    result = _phi_j_rect(a, b, x)
    xr.testing.assert_allclose(result, expected)


def test__phi_j_prime_rect():
    """Tests that `_phi_j_prime_rect` gives results as expected."""
    result = _phi_j_prime_rect(-1.0, 2.0, DA_X1)
    xr.testing.assert_allclose(result, EXP_PHI_J_PRIME_RECT1)


@pytest.mark.parametrize(
    ("a", "b", "c", "d", "x", "expected"),
    [
        (-2, 1, 5, 8, DA_X3, EXP_G_J_TRAP1),  # a, b, c, d float; all cases tested
        (
            DA_A_TRAP,
            DA_B_TRAP,
            DA_C_TRAP,
            DA_D_TRAP,
            DA_X_TRAP,
            EXP_G_J_TRAP2,
        ),  # endpoints are arrays
    ],
)
def test__g_j_trap(a, b, c, d, x, expected):
    """Tests that `_g_j_trap` gives results as expected."""
    result = _g_j_trap(a, b, c, d, x)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("a", "b", "c", "d", "x", "expected"),
    [
        (-2, 1, 5, 8, DA_X3, EXP_PHI_J_TRAP1),  # endpts float; all cases tested
        (
            DA_A_TRAP,
            DA_B_TRAP,
            DA_C_TRAP,
            DA_D_TRAP,
            DA_X_TRAP,
            EXP_PHI_J_TRAP2,
        ),  # endpoints are arrays
    ],
)
def test__phi_j_trap(a, b, c, d, x, expected):
    """Tests that `_phi_j_trap` gives results as expected."""
    result = _phi_j_trap(a, b, c, d, x)
    xr.testing.assert_allclose(result, expected)


def test__phi_j_prime_trap():
    """Tests that `_phi_j_prime_trap` gives results as expected."""
    result = _phi_j_prime_trap(-2, 1, 5, 8, DA_X3)
    xr.testing.assert_allclose(result, EXP_PHI_J_PRIME_TRAP)


@pytest.mark.parametrize(
    ("interval_where_one", "a", "b"),
    [
        ((1, 4), 1, 4),
        ((-np.inf, 5), -6, 5),
        ((-6, np.inf), -6, 11),
        ((DA_A_INF, DA_B_INF), DA_A_FINITE, DA_B_FINITE),
    ],
)
def test__auxiliary_funcs1(interval_where_one, a, b):
    """
    Tests that `_auxiliary_funcs` gives expected results for "rectangular" weights.
    """
    g, phi, phi_prime = _auxiliary_funcs(
        xr.DataArray([-5, 4], dims=["station"], coords=dict(station=[100, 101])),
        xr.DataArray([0, 10], dims=["station"], coords=dict(station=[100, 101])),
        interval_where_one,
        None,
    )

    xvalues = np.linspace(-10, 12, 100)
    x = xr.DataArray(data=xvalues, dims=["x_dim"], coords=dict(x_dim=xvalues))
    xr.testing.assert_allclose(g(x), _g_j_rect(a, b, x))
    xr.testing.assert_allclose(phi(x), _phi_j_rect(a, b, x))
    xr.testing.assert_allclose(phi_prime(x), _phi_j_prime_rect(a, b, x))


@pytest.mark.parametrize(
    ("interval_where_one", "interval_where_positive", "a", "b", "c", "d"),
    [
        ((1, 4), (-1, 5), -1, 1, 4, 5),
        ((-np.inf, 4), (-np.inf, 5), -7, -6, 4, 5),
        ((-1, np.inf), (-3, np.inf), -3, -1, 11, 12),
    ],
)
def test__auxiliary_funcs2(interval_where_one, interval_where_positive, a, b, c, d):
    """
    Tests that `_auxiliary_funcs` gives expected results for "trapezoidal" weights.
    """
    g, phi, phi_prime = _auxiliary_funcs(
        xr.DataArray([0, 10]),
        xr.DataArray([-5, 9]),
        interval_where_one,
        interval_where_positive,
    )
    x = xr.DataArray(np.linspace(-10, 12, 50))
    xr.testing.assert_allclose(g(x), _g_j_trap(a, b, c, d, x))
    xr.testing.assert_allclose(phi(x), _phi_j_trap(a, b, c, d, x))
    xr.testing.assert_allclose(phi_prime(x), _phi_j_prime_trap(a, b, c, d, x))


# Test all of the threshold weighted scores
@pytest.mark.parametrize(
    ("scoring_func", "kwargs", "expected"),
    [
        (tw_squared_error, {}, mse(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"])),
        (tw_absolute_error, {}, mae(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"])),
        (
            tw_quantile_score,
            {"alpha": 0.3},
            quantile_score(DA_FCST1, DA_OBS1, alpha=0.3, preserve_dims=["date", "station"]),
        ),
        (
            tw_expectile_score,
            {"alpha": 0.5},
            0.5 * mse(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"]),
        ),
        (
            tw_huber_loss,
            {"huber_param": HUBER_PARAM},
            EXP_HL,
        ),
    ],
)
def test_threshold_weighted_scores1(scoring_func, kwargs, expected):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if `interval_where_one=(-np.inf, np.inf)`
    then the threshold weighted score should be the same as the original score.
    """
    result = scoring_func(
        DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, np.inf), preserve_dims=["date", "station"], **kwargs
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs"),
    [
        (tw_squared_error, {}),
        (tw_absolute_error, {}),
        (tw_quantile_score, {"alpha": 0.3}),
        (tw_expectile_score, {"alpha": 0.3}),
        (tw_huber_loss, {"huber_param": HUBER_PARAM}),
    ],
)
def test_threshold_weighted_scores2(scoring_func, kwargs):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if the sum of scores whose weights
    sum to 1 is the same as the score with a weight of 1 everywhere.
    Tests for rectangular weights.
    """
    kwargs["preserve_dims"] = ["date", "station"]

    score1 = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, 0), **kwargs)
    score2 = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(0, np.inf), **kwargs)
    score = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, np.inf), **kwargs)
    xr.testing.assert_allclose(score1 + score2, score)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs"),
    [
        (tw_squared_error, {}),
        (tw_absolute_error, {}),
        (tw_quantile_score, {"alpha": 0.3}),
        (tw_expectile_score, {"alpha": 0.3}),
        (tw_huber_loss, {"huber_param": HUBER_PARAM}),
    ],
)
def test_threshold_weighted_scores3(scoring_func, kwargs):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if the sum of scores whose weights
    sum to 1 is the same as the score with a weight of 1 everywhere.
    Tests for rectangular weights, weights vary with station dimension.
    """

    kwargs["preserve_dims"] = ["date", "station"]

    score1 = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-DA_INF, DA_ENDPT1), **kwargs)
    score2 = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(DA_ENDPT1, DA_INF), **kwargs)
    score = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-DA_INF, DA_INF), **kwargs)
    xr.testing.assert_allclose(score1 + score2, score)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs"),
    [
        (tw_squared_error, {}),
        (tw_absolute_error, {}),
        (tw_quantile_score, {"alpha": 0.3}),
        (tw_expectile_score, {"alpha": 0.3}),
        (tw_huber_loss, {"huber_param": HUBER_PARAM}),
    ],
)
def test_threshold_weighted_scores4(scoring_func, kwargs):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if the sum of scores whose weights
    sum to 1 is the same as the score with a weight of 1 everywhere.
    Tests for trapezoidal weights.
    """

    kwargs["preserve_dims"] = ["date", "station"]
    point1 = -4
    point2 = 4
    score1 = scoring_func(
        DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, point1), **kwargs, interval_where_positive=(-np.inf, point2)
    )
    score2 = scoring_func(
        DA_FCST1, DA_OBS1, interval_where_one=(point2, np.inf), **kwargs, interval_where_positive=(point1, np.inf)
    )
    score = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, np.inf), **kwargs)
    xr.testing.assert_allclose(score1 + score2, score)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs"),
    [
        (tw_squared_error, {}),
        (tw_absolute_error, {}),
        (tw_quantile_score, {"alpha": 0.3}),
        (tw_expectile_score, {"alpha": 0.3}),
        (tw_huber_loss, {"huber_param": HUBER_PARAM}),
    ],
)
def test_threshold_weighted_scores5(scoring_func, kwargs):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if the sum of scores whose weights
    sum to 1 is the same as the score with a weight of 1 everywhere.
    Tests for trapezoidal weights, with weights varying by stn dimension.
    """

    kwargs["preserve_dims"] = ["date", "station"]
    score1 = scoring_func(
        DA_FCST1,
        DA_OBS1,
        interval_where_one=(-DA_INF, DA_ENDPT1),
        **kwargs,
        interval_where_positive=(-DA_INF, DA_ENDPT2),
    )
    score2 = scoring_func(
        DA_FCST1, DA_OBS1, interval_where_one=(DA_ENDPT2, DA_INF), **kwargs, interval_where_positive=(DA_ENDPT1, DA_INF)
    )
    score = scoring_func(DA_FCST1, DA_OBS1, interval_where_one=(-np.inf, np.inf), **kwargs)
    xr.testing.assert_allclose(score1 + score2, score)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs"),
    [
        (tw_squared_error, {}),
        (tw_absolute_error, {}),
        (tw_quantile_score, {"alpha": 0.3}),
        (tw_expectile_score, {"alpha": 0.3}),
        (tw_huber_loss, {"huber_param": HUBER_PARAM}),
    ],
)
def test_threshold_weighted_scores6(scoring_func, kwargs):
    """
    Tests that each threshold weight score gives results as expected when the `reduce_dims`
    arg is used.
    """
    score1 = scoring_func(
        DA_FCST1,
        DA_OBS1,
        interval_where_one=(DA_ENDPT2, DA_INF),
        **kwargs,
        interval_where_positive=(DA_ENDPT1, DA_INF),
        preserve_dims=["date", "station"],
    )
    score2 = scoring_func(
        DA_FCST1,
        DA_OBS1,
        interval_where_one=(DA_ENDPT2, DA_INF),
        **kwargs,
        interval_where_positive=(DA_ENDPT1, DA_INF),
        reduce_dims=["date", "station"],
    )
    xr.testing.assert_allclose(np.mean(score1), score2)


@pytest.mark.parametrize(
    ("scoring_func", "kwargs", "expected"),
    [
        (tw_squared_error, {}, mse(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"])),
        (tw_absolute_error, {}, mae(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"])),
        (
            tw_quantile_score,
            {"alpha": 0.3},
            quantile_score(DA_FCST1, DA_OBS1, alpha=0.3, preserve_dims=["date", "station"]),
        ),
        (tw_expectile_score, {"alpha": 0.5}, 0.5 * mse(DA_FCST1, DA_OBS1, preserve_dims=["date", "station"])),
        (
            tw_huber_loss,
            {"huber_param": HUBER_PARAM},
            EXP_HL,
        ),
    ],
)
def test_threshold_weighted_scores_dask(scoring_func, kwargs, expected):
    """
    Tests that each threshold weight score gives results as expected.
    This test is based on the fact that if `interval_where_one=(-np.inf, np.inf)`
    then the threshold weighted score should be the same as the original score.
    """
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover
    result = scoring_func(
        DA_FCST1.chunk(),
        DA_OBS1.chunk(),
        interval_where_one=(-DA_INF.chunk(), DA_INF.chunk()),
        preserve_dims=["date", "station"],
        **kwargs,
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, expected)
