"""Tests for murphy metrics and thetas generation code."""
import re
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from scores.continuous import murphy_score, murphy_thetas
from scores.continuous.murphy_impl import _expectile_thetas, _huber_thetas, _quantile_thetas
from scores.continuous import murphy_impl as murphy


FCST = xr.DataArray(
    dims=("lead_day", "station_number", "valid_15z_date"),
    data=[[[0.0], [5.0]], [[10.0], [15.0]]],
    coords={
        "lead_day": [1, 2],
        "station_number": [100001, 10000],
        "valid_15z_date": [datetime(2017, 1, 1, 15, 0)],
    },
)
OBS = xr.DataArray(
    dims=("valid_15z_date", "station_number"),
    data=[[4.0, 2.0]],
    coords={
        "station_number": [10000, 100001],  # Different ordering from FCST
        "valid_15z_date": [datetime(2017, 1, 1, 15, 0)],
    },
)
EXPECTED_QUANTILE = xr.Dataset(
    coords={"theta": [0.0, 2.0, 10.0], "lead_day": [1, 2]},
    data_vars={
        "total": xr.DataArray(
            dims=("lead_day", "theta"),
            data=[[0.25, 0.0, 0.0], [0.0, 0.25, 0.25]],
        ),
        "underforecast": xr.DataArray(
            dims=("lead_day", "theta"),
            data=[[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
        "overforecast": xr.DataArray(
            dims=("lead_day", "theta"),
            data=[[0.0, 0.0, 0.0], [0.0, 0.25, 0.25]],
        ),
    },
)
EXPECTED_HUBER = xr.Dataset(
    coords={"theta": [0.0, 2.0, 10.0], "lead_day": [1, 2]},
    data_vars={
        "total": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.5, 0.0], [0.0, 0.0], [0.0, 0.75]],
        ),
        "underforecast": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ),
        "overforecast": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.75]],
        ),
    },
)
EXPECTED_EXPECTILE = xr.Dataset(
    coords={"theta": [0.0, 2.0, 10.0], "lead_day": [1, 2]},
    data_vars={
        "total": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.5, 0.0], [0.0, 0.0], [0.0, 1.5]],
        ),
        "underforecast": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ),
        "overforecast": xr.DataArray(
            dims=("theta", "lead_day"),
            data=[[0.0, 0.0], [0.0, 0.0], [0.0, 1.5]],
        ),
    },
)


def patch_scoring_func(monkeypatch, score_function, thetas):
    """Monkeypatch a scoring function, return the mock object."""
    over = _rel_test_array(np.array([[999, 999, np.nan, np.nan]] * len(thetas)).T, theta=thetas)
    under = _rel_test_array(np.array([[np.nan, 888, np.nan, 888]] * len(thetas)).T, theta=thetas)
    mock_rel_fc_func = Mock(return_value=[over, under])
    monkeypatch.setattr(
        murphy,
        score_function,
        mock_rel_fc_func,
    )
    return mock_rel_fc_func


@pytest.mark.parametrize(
    ("functional", "score_function"),
    [
        (murphy.QUANTILE, "_quantile_elementary_score"),
        (murphy.HUBER, "_huber_elementary_score"),
        (murphy.EXPECTILE, "_expectile_elementary_score"),
    ],
)
def test_murphy_score_operations(functional, score_function, monkeypatch):
    """murphy_score makes the expected operations on the scoring function output."""
    fcst = _test_array([1.0, 2.0, 3.0, 4.0])
    obs = _test_array([0.0, np.nan, 0.6, 137.4])
    thetas = [0.0, 2.0, 10.0]
    mock_rel_fc_func = patch_scoring_func(monkeypatch, score_function, thetas)

    result = murphy.murphy_score(
        fcst=fcst,
        obs=obs,
        thetas=thetas,
        functional=functional,
        alpha=0.5,
        huber_a=3,
        decomposition=True,
        preserve_dims=fcst.dims,
    )

    expected = xr.Dataset.from_dict(
        {
            "dims": ("station_number", "theta"),
            "data_vars": {
                "total": {
                    "data": [
                        [999.0, 999.0, 999.0],
                        [np.nan, np.nan, np.nan],
                        [0.0, 0.0, 0.0],
                        [888.0, 888.0, 888.0],
                    ],
                    "dims": ("station_number", "theta"),
                },
                "underforecast": {
                    "data": [
                        [0.0, 0.0, 0.0],
                        [np.nan, np.nan, np.nan],
                        [0.0, 0.0, 0.0],
                        [888.0, 888.0, 888.0],
                    ],
                    "dims": ("station_number", "theta"),
                },
                "overforecast": {
                    "data": [
                        [999.0, 999.0, 999.0],
                        [np.nan, np.nan, np.nan],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    "dims": ("station_number", "theta"),
                },
            },
            "coords": {
                "station_number": {
                    "dims": ("station_number",),
                    "data": [46012, 46126, 46128, 46129],
                },
                "theta": {
                    "dims": ("theta",),
                    "data": thetas,
                },
            },
        }
    )

    xr.testing.assert_identical(result, expected)
    mock_rel_fc_func.assert_called_once()


@pytest.mark.parametrize(
    ("functional", "expected"),
    [
        (murphy.QUANTILE, EXPECTED_QUANTILE),
        (murphy.HUBER, EXPECTED_HUBER),
        (murphy.EXPECTILE, EXPECTED_EXPECTILE),
    ],
)
def test_murphy_score(functional, expected):
    """murphy_score returns the expected object."""
    thetas = [0.0, 2.0, 10.0]

    result = murphy.murphy_score(
        fcst=FCST,
        obs=OBS,
        thetas=thetas,
        functional=functional,
        alpha=0.5,
        huber_a=3.0,
        decomposition=True,
        preserve_dims=["lead_day"],
    )

    xr.testing.assert_identical(result, expected)


def test_murphy_score_mean(monkeypatch):
    """
    murphy_score returns the mean of the result if both reduce_dims and
    preserve_dims are None.
    """
    fcst = _test_array([1.0, 2.0, 3.0, 4.0])
    obs = _test_array([0.0, np.nan, 0.6, 137.4])
    thetas = [0.0, 2.0, 10.0]
    _ = patch_scoring_func(monkeypatch, "_quantile_elementary_score", thetas)

    result = murphy.murphy_score(
        fcst=fcst,
        obs=obs,
        thetas=thetas,
        functional=murphy.QUANTILE,
        alpha=0.5,
        huber_a=3,
        decomposition=True,
    )

    expected = xr.Dataset.from_dict(
        {
            "dims": ("theta"),
            "data_vars": {
                "total": {
                    "data": [629.0, 629.0, 629.0],
                    "dims": ("theta"),
                },
                "underforecast": {
                    "data": [296.0, 296.0, 296.0],
                    "dims": ("theta"),
                },
                "overforecast": {
                    "data": [333.0, 333.0, 333.0],
                    "dims": ("theta"),
                },
            },
            "coords": {
                "theta": {
                    "dims": ("theta",),
                    "data": thetas,
                },
            },
        }
    )
    xr.testing.assert_identical(result, expected)


def test_murphy_score_no_decomposition(monkeypatch):
    """murphy_score returns only the total score if decomposition is False."""
    fcst = _test_array([1.0, 2.0, 3.0, 4.0])
    obs = _test_array([0.0, np.nan, 0.6, 137.4])
    thetas = [0.0, 2.0, 10.0]
    _ = patch_scoring_func(monkeypatch, "_quantile_elementary_score", thetas)

    result = murphy.murphy_score(
        fcst=fcst,
        obs=obs,
        thetas=thetas,
        functional=murphy.QUANTILE,
        alpha=0.5,
        huber_a=3,
        decomposition=False,
    )

    assert list(result.variables.keys()) == ["theta", "total"]


def _test_array(data):
    """Return a test array for a forecast or obs input."""
    assert len(data) <= 4
    return xr.DataArray.from_dict(
        {
            "dims": ("station_number"),
            "data": data,
            "coords": {
                "station_number": {
                    "dims": ("station_number",),
                    "data": [46012, 46126, 46128, 46129][0 : len(data)],
                },
            },
        }
    )


def _rel_test_array(data, theta):
    """Return a test array for *_elementary_score function inputs."""
    assert len(data) <= 4
    return xr.DataArray.from_dict(
        {
            "dims": ("station_number", "theta"),
            "data": data,
            "coords": {
                "station_number": {
                    "dims": ("station_number",),
                    "data": [46012, 46126, 46128, 46129][0 : len(data)],
                },
                "theta": {
                    "dims": ("theta",),
                    "data": theta,
                },
            },
        }
    )


def test__quantile_elementary_score():
    """_quantile_elementary_score returns the expected values."""
    fcst = _rel_test_array(data=np.array([[0, 1, 4]] * 2).T, theta=[0, 2])
    obs = _rel_test_array(data=np.array([[1, 3, 1]] * 2).T, theta=[0, 2])
    theta = _rel_test_array(data=[[0, 2]] * 3, theta=[0, 2])
    alpha = 0.1

    result = murphy._quantile_elementary_score(fcst, obs, theta, alpha)

    assert len(result) == 2
    np.testing.assert_equal(result[0], np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.9]]))
    np.testing.assert_equal(result[1], np.array([[0.1, np.nan], [np.nan, 0.1], [np.nan, np.nan]]))


def test__huber_elementary_score():
    """_huber_elementary_score returns the expected values."""
    fcst = _rel_test_array(data=np.array([[0, 1, 4]] * 2).T, theta=[0, 2])
    obs = _rel_test_array(data=np.array([[1, 3, 1]] * 2).T, theta=[0, 2])
    theta = _rel_test_array(data=[[0, 2]] * 3, theta=[0, 2])
    alpha = 0.1

    result = murphy._huber_elementary_score(fcst, obs, theta, alpha, huber_a=0.5)

    assert len(result) == 2
    np.testing.assert_equal(result[0], np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.45]]))
    np.testing.assert_equal(result[1], np.array([[0.05, np.nan], [np.nan, 0.05], [np.nan, np.nan]]))


def test__expectile_elementary_score():
    """_expectile_elementary_score returns the expected values."""
    fcst = _rel_test_array(data=np.array([[0, 1, 4]] * 2).T, theta=[0, 2])
    obs = _rel_test_array(data=np.array([[1, 3, 1]] * 2).T, theta=[0, 2])
    theta = _rel_test_array(data=[[0, 2]] * 3, theta=[0, 2])
    alpha = 0.1

    result = murphy._expectile_elementary_score(fcst, obs, theta, alpha)

    assert len(result) == 2
    np.testing.assert_equal(result[0], np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, 0.9]]))
    np.testing.assert_equal(result[1], np.array([[0.1, np.nan], [np.nan, 0.1], [np.nan, np.nan]]))


@pytest.mark.parametrize(
    ("new_kwargs", "expected_exception_msg"),
    (
        [
            {"alpha": 0},
            "alpha (=0) argument for Murphy scoring function should be strictly " "between 0 and 1.",
        ],
        [
            {"alpha": 1},
            "alpha (=1) argument for Murphy scoring function should be strictly " "between 0 and 1.",
        ],
        [
            {"functional": "?"},
            "Functional option '?' for Murphy scoring function is unknown, should be "
            "one of ['quantile', 'huber', 'expectile'].",
        ],
        [
            {"functional": "huber", "huber_a": 0},
            "huber_a (=0) argument should be > 0 when functional='huber'.",
        ],
        [
            {"functional": "huber", "huber_a": None},
            "huber_a (=None) argument should be > 0 when functional='huber'.",
        ],
    ),
)
def test_murphy_score_invalid_input(new_kwargs, expected_exception_msg):
    """murphy_score raises an exception for invalid inputs."""
    fcst = _test_array([1.0, 2.0, 3.0, 4.0])
    obs = _test_array([0.0, np.nan, 0.6, 137.4])
    thetas = [0.0, 2.0, 10.0]
    kwargs = {
        "fcst": fcst,
        "obs": obs,
        "thetas": thetas,
        "functional": murphy.QUANTILE,
        "alpha": 0.5,
        "huber_a": 3,
        "decomposition": True,
    }

    with pytest.raises(ValueError, match=re.escape(expected_exception_msg)):
        _ = murphy.murphy_score(**{**kwargs, **new_kwargs})


@pytest.mark.parametrize(
    ("functional", "left_limit_delta", "expected"),
    [
        [murphy.QUANTILE, 0.1, [0.0, 2.0, 4.0, 5.0, 10.0, 15.0]],
        [
            murphy.HUBER,
            0.1,
            [-0.1, 0.0, 1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 4.9, 5.0, 9.9, 10.0, 14.9, 15.0],
        ],
        [murphy.EXPECTILE, 0.1, [-0.1, 0.0, 2.0, 4.0, 4.9, 5.0, 9.9, 10.0, 14.9, 15.0]],
        [murphy.EXPECTILE, None, [0.0, 2.0, 4.0, 5.0, 10.0, 15.0]],
    ],
)
def test_murphy_thetas(functional, left_limit_delta, expected):
    """murphy_thetas returns the expected object."""

    result = murphy_thetas(
        forecasts=[FCST],
        obs=OBS,
        functional=functional,
        huber_a=0.5,
        left_limit_delta=left_limit_delta,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("functional"),
    [murphy.QUANTILE, murphy.HUBER, murphy.EXPECTILE],
)
@patch("scores.continuous.murphy_impl._quantile_thetas", autospec=True)
@patch("scores.continuous.murphy_impl._huber_thetas", autospec=True)
@patch("scores.continuous.murphy_impl._expectile_thetas", autospec=True)
def test_murphy_thetas_calls(mock__expectile_thetas, mock__huber_thetas, mock__quantile_thetas, functional):
    """murphy_thetas makes the expected function call."""
    result = murphy_thetas(
        forecasts=1,
        obs=2,
        functional=functional,
        huber_a=4,
        left_limit_delta=5,
    )

    expected_kwargs = {
        "forecasts": 1,
        "obs": 2,
        "huber_a": 4,
        "left_limit_delta": 5,
    }
    if functional == murphy.QUANTILE:
        assert result == mock__quantile_thetas.return_value
        mock__quantile_thetas.assert_called_once_with(**expected_kwargs)
        mock__huber_thetas.assert_not_called()
        mock__expectile_thetas.assert_not_called()
    elif functional == murphy.HUBER:
        assert result == mock__huber_thetas.return_value
        mock__quantile_thetas.assert_not_called()
        mock__huber_thetas.assert_called_once_with(**expected_kwargs)
        mock__expectile_thetas.assert_not_called()
    elif functional == murphy.EXPECTILE:
        assert result == mock__expectile_thetas.return_value
        mock__quantile_thetas.assert_not_called()
        mock__huber_thetas.assert_not_called()
        mock__expectile_thetas.assert_called_once_with(**expected_kwargs)


def test__quantile_thetas():
    """_quantile_thetas returns the expected values."""
    forecasts = [_test_array([1.0, 2.0, 3.0]), _test_array([0.0, 10.0, np.nan])]
    obs = _test_array([0.0, 0.6, 137.4])

    result = _quantile_thetas(forecasts, obs)

    assert result == [0.0, 0.6, 1.0, 2.0, 3.0, 10.0, 137.4]


@pytest.mark.parametrize(
    ("left_limit_delta", "expected"),
    [
        (0.1, [-10.0, -0.1, 0.0, 0.9, 1.0, 1.9, 2.0, 10.0, 127.4, 137.4, 147.4]),
        (0, [-10.0, 0.0, 1.0, 2.0, 10.0, 127.4, 137.4, 147.4]),
    ],
)
def test__huber_thetas(left_limit_delta, expected):
    """_huber_thetas returns the expected values."""
    forecasts = [_test_array([1.0, 2.0]), _test_array([0.0, np.nan])]
    obs = _test_array([0.0, 137.4])

    result = _huber_thetas(forecasts, obs, huber_a=10.0, left_limit_delta=left_limit_delta)

    assert result == expected


@pytest.mark.parametrize(
    ("left_limit_delta", "expected"),
    [
        (0.1, [-0.1, 0.0, 0.9, 1.0, 1.9, 2.0, 137.4]),
        (0, [0.0, 1.0, 2.0, 137.4]),
    ],
)
def test__expectile_thetas(left_limit_delta, expected):
    """_expectile_thetas returns the expected values."""
    forecasts = [_test_array([1.0, 2.0]), _test_array([0.0, np.nan])]
    obs = _test_array([0.0, 137.4])

    result = _expectile_thetas(forecasts, obs, left_limit_delta=left_limit_delta)

    assert result == expected


@pytest.mark.parametrize(
    ("new_kwargs", "expected_exception_msg"),
    (
        [
            {"functional": "?"},
            "Functional option '?' for Murphy scoring function is unknown, should be "
            "one of ['quantile', 'huber', 'expectile'].",
        ],
        [
            {"functional": "huber", "huber_a": 0},
            "huber_a (=0) argument should be > 0 when functional='huber'.",
        ],
        [
            {"functional": "expectile", "left_limit_delta": -0.1},
            "left_limit_delta (=-0.1) argument should be >= 0.",
        ],
    ),
)
def test_murphy_thetas_invalid_inputs(new_kwargs, expected_exception_msg):
    """murphy_thetas raises an exception for invalid inputs."""
    forecasts = [_test_array([1.0, 2.0]), _test_array([0.0, np.nan])]
    obs = _test_array([0.0, 137.4])
    kwargs = {
        "forecasts": forecasts,
        "obs": obs,
        "functional": murphy.QUANTILE,
        "huber_a": 10.0,
        "left_limit_delta": 1.0,
    }

    with pytest.raises(ValueError, match=re.escape(expected_exception_msg)):
        _ = murphy_thetas(**{**kwargs, **new_kwargs})
