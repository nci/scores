"""
Contains unit tests for scores.probability.continuous.quantile_interval_score and
scores.probability.continuous.interval_score
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here  # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore # pylint: disable=invalid-name  # pragma: no cover

import pytest
import xarray as xr

from scores.continuous import interval_score, quantile_interval_score
from scores.utils import DimensionError
from tests.continuous import quantile_interval_score_test_data as qistd


@pytest.mark.parametrize(
    ("lower_qtile", "upper_qtile", "lower_fcst", "upper_fcst", "error_message"),
    [
        (0, 1, qistd.FCST_LOWER, qistd.FCST_UPPER, qistd.ERROR_MESSAGE_QTILE.format(0, 1)),
        (-0.1, 0.3, qistd.FCST_LOWER, qistd.FCST_UPPER, qistd.ERROR_MESSAGE_QTILE.format(-0.1, 0.3)),
        (0.5, 0.4, qistd.FCST_LOWER, qistd.FCST_UPPER, qistd.ERROR_MESSAGE_QTILE.format(0.5, 0.4)),
        (0.1, 0.9, qistd.FCST_UPPER, qistd.FCST_LOWER, qistd.ERROR_MESSAGE_FCST_COND),
    ],
)
def test_qis_value_errors(lower_qtile, upper_qtile, lower_fcst, upper_fcst, error_message):
    """quantile_interval_score raises ValueError."""
    with pytest.raises(ValueError, match=error_message):
        quantile_interval_score(
            fcst_lower_qtile=lower_fcst,
            fcst_upper_qtile=upper_fcst,
            obs=qistd.OBS,
            lower_qtile_level=lower_qtile,
            upper_qtile_level=upper_qtile,
        )


@pytest.mark.parametrize(
    ("obs", "lower_fcst", "upper_fcst", "reduce_dims", "preserve_dims"),
    [
        # lower and upper quantile forecasts with mismatched dims
        (qistd.OBS, qistd.FCST_LOWER, qistd.FCST_UPPER_STATION, None, None),
        # fcst and obs with mismatched dims
        (qistd.OBS_STATION, qistd.FCST_LOWER, qistd.FCST_UPPER, None, None),
        # dims not in fcst or obs
        (qistd.OBS, qistd.FCST_LOWER, qistd.FCST_UPPER, ["bananas"], None),
        (qistd.OBS, qistd.FCST_LOWER, qistd.FCST_UPPER, None, ["orange"]),
    ],
)
def test_qis_exceptions(obs, lower_fcst, upper_fcst, reduce_dims, preserve_dims):
    """quantile_score raises DimensionError."""
    with pytest.raises(DimensionError):
        quantile_interval_score(
            obs=obs,
            fcst_lower_qtile=lower_fcst,
            fcst_upper_qtile=upper_fcst,
            lower_qtile_level=0.1,
            upper_qtile_level=0.9,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )


@pytest.mark.parametrize(
    (
        "lower_fcst",
        "upper_fcst",
        "obs",
        "lower_qtile",
        "upper_qtile",
        "preserve_dims",
        "reduce_dims",
        "weights",
        "expected",
    ),
    [
        (
            qistd.FCST_LOWER,
            qistd.FCST_UPPER,
            qistd.OBS,
            0.1,
            0.9,
            ["time"],
            None,
            None,
            qistd.EXPECTED_WITH_TIME,
        ),
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.1,
            0.9,
            ["time", "station"],
            None,
            None,
            qistd.EXPECTED_2D,
        ),
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.1,
            0.9,
            None,
            ["time"],
            None,
            qistd.EXPECTED_2D_WITHOUT_TIME,
        ),
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.2,
            0.9,
            ["time", "station"],
            None,
            None,
            qistd.EXPECTED_2D_NON_BALANCE,
        ),
        # To test with missing data
        (
            qistd.FCST_LOWER_2D_WITH_NAN,
            qistd.FCST_UPPER_2D_WITH_NAN,
            qistd.OBS_2D,
            0.1,
            0.9,
            ["time", "station"],
            None,
            None,
            qistd.EXPECTED_2D_WITH_NAN,
        ),
        # To test weight
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.1,
            0.9,
            ["time", "station"],
            None,
            qistd.WEIGHTS,
            qistd.EXPECTED_2D_WITH_WEIGHTS,
        ),
    ],
)
def test_qsf_calculations(
    lower_fcst, upper_fcst, obs, lower_qtile, upper_qtile, preserve_dims, reduce_dims, weights, expected
):
    """quantile_score returns the expected object."""
    result = quantile_interval_score(
        obs=obs,
        fcst_lower_qtile=lower_fcst,
        fcst_upper_qtile=upper_fcst,
        lower_qtile_level=lower_qtile,
        upper_qtile_level=upper_qtile,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(result, expected)


def test_quantile_interval_score_dask():
    """Tests quantile_interval_score works with dask"""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = quantile_interval_score(
        fcst_lower_qtile=qistd.FCST_LOWER_2D.chunk(),
        fcst_upper_qtile=qistd.FCST_UPPER_2D.chunk(),
        obs=qistd.OBS_2D.chunk(),
        lower_qtile_level=0.1,
        upper_qtile_level=0.9,
        preserve_dims=["time", "station"],
        reduce_dims=None,
        weights=None,
    )
    for var in result.data_vars:
        assert isinstance(result[var].data, dask.array.Array)
    result = result.compute()
    xr.testing.assert_allclose(result, qistd.EXPECTED_2D)


@pytest.mark.parametrize(
    ("interval_range", "lower_fcst", "upper_fcst", "error_message"),
    [
        (1, qistd.FCST_LOWER, qistd.FCST_UPPER, qistd.ERROR_MESSAGE_INTERVAL_RANGE),
        (-0.1, qistd.FCST_LOWER, qistd.FCST_UPPER, qistd.ERROR_MESSAGE_INTERVAL_RANGE),
        (0.5, qistd.FCST_UPPER, qistd.FCST_LOWER, qistd.ERROR_MESSAGE_FCST_COND),
    ],
)
def test_is_value_errors(interval_range, lower_fcst, upper_fcst, error_message):
    """interval_score raises ValueError."""
    with pytest.raises(ValueError, match=error_message):
        interval_score(
            fcst_lower_qtile=lower_fcst,
            fcst_upper_qtile=upper_fcst,
            obs=qistd.OBS,
            interval_range=interval_range,
        )


@pytest.mark.parametrize(
    ("lower_fcst", "upper_fcst", "obs", "interval_range", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.5,
            ["time", "station"],
            None,
            None,
            qistd.EXPECTED_2D_INTERVAL,
        ),
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.4,
            None,
            ["time"],
            None,
            qistd.EXPECTED_2D_WITHOUT_TIME_INTERVAL,
        ),
        # To test weight
        (
            qistd.FCST_LOWER_2D,
            qistd.FCST_UPPER_2D,
            qistd.OBS_2D,
            0.4,
            ["time", "station"],
            None,
            qistd.WEIGHTS,
            qistd.EXPECTED_2D_WITH_WEIGHTS_INTERVAL,
        ),
    ],
)
def test_is_calculations(lower_fcst, upper_fcst, obs, interval_range, preserve_dims, reduce_dims, weights, expected):
    """interval_score returns the expected object."""
    result = interval_score(
        obs=obs,
        fcst_lower_qtile=lower_fcst,
        fcst_upper_qtile=upper_fcst,
        interval_range=interval_range,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(result, expected)
