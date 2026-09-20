"""
Implementation of continuous metrics for Narwhals types
"""

import math

import narwhals as nw

from scores.df.typing import SeriesType


def _to_series(data: SeriesType) -> nw.Series:
    """
    Wraps a native data series in the Narwhals interface. Scalars are passed through
    unchanged so that a series can be compared against a single point value.
    """
    return nw.from_native(data, series_only=True, pass_through=True)


def _error(fcst: nw.Series, obs: nw.Series, is_angular: bool) -> nw.Series:
    """
    Calculates the error between `fcst` and `obs`.

    When `is_angular` is True the smaller of the two explementary angles is returned,
    matching :py:func:`scores.functions.angular_difference`.
    """
    if is_angular:
        difference = (fcst - obs).abs() % 360
        return difference.zip_with(difference <= 180, 360 - difference)

    return fcst - obs


def mse(
    fcst: SeriesType,
    obs: SeriesType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> float:
    """Calculates the mean squared error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_squared_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2


    Notes:
        This function will accept any data series that is supported by Narwhals,
        including pandas Series and Polars Series.

        Dimensional reduction is not supported for Narwhals and the user should
        convert their data to xarray to formulate the call to the base metric,
        `scores.continuous.mse`.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        float:
            An object containing a single floating point number representing the mean squared
            error for the supplied data. All dimensions will be reduced.

    Examples:
        >>> from scores.df.continuous import mse
        >>> import pandas as pd

        >>> fcst = pd.Series([1.5, 0.7, 1.4], name="forecast")
        >>> obs = pd.Series([1.2, 0.8, 1.5], name="observed")

        >>> mse(fcst, obs)
        0.03666666666666669

    """
    error = _error(_to_series(fcst), _to_series(obs), is_angular)

    return float((error * error).mean())


def rmse(
    fcst: SeriesType,
    obs: SeriesType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> float:
    """Calculates the root mean squared error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math ::
        \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2}


    Notes:
        This function will accept any data series that is supported by Narwhals,
        including pandas Series and Polars Series.

        Dimensional reduction is not supported for Narwhals and the user should
        convert their data to xarray to formulate the call to the base metric,
        `scores.continuous.rmse`.

        Missing values are skipped, following the conventions of the underlying
        library. Note that Polars treats null and NaN as distinct, and only skips
        null - see the `Narwhals documentation
        <https://narwhals-dev.github.io/narwhals/pandas_like_concepts/null_handling/>`_.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        float:
            An object containing a single floating point number representing the root mean
            squared error for the supplied data. All dimensions will be reduced.

    Examples:
        >>> from scores.df.continuous import rmse
        >>> import pandas as pd

        >>> fcst = pd.Series([1.5, 0.7, 1.4], name="forecast")
        >>> obs = pd.Series([1.2, 0.8, 1.5], name="observed")

        >>> rmse(fcst, obs)
        0.1914854215512677

    """
    return math.sqrt(mse(fcst, obs, is_angular=is_angular))


def mae(
    fcst: SeriesType,
    obs: SeriesType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> float:
    """Calculates the mean absolute error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n | \\text{forecast}_i - \\text{observed}_i |


    Notes:
        This function will accept any data series that is supported by Narwhals,
        including pandas Series and Polars Series.

        Dimensional reduction is not supported for Narwhals and the user should
        convert their data to xarray to formulate the call to the base metric,
        `scores.continuous.mae`.

        Missing values are skipped, following the conventions of the underlying
        library. Note that Polars treats null and NaN as distinct, and only skips
        null - see the `Narwhals documentation
        <https://narwhals-dev.github.io/narwhals/pandas_like_concepts/null_handling/>`_.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        float:
            An object containing a single floating point number representing the mean
            absolute error for the supplied data. All dimensions will be reduced.

    Examples:
        >>> from scores.df.continuous import mae
        >>> import pandas as pd

        >>> fcst = pd.Series([1.5, 0.7, 1.4], name="forecast")
        >>> obs = pd.Series([1.2, 0.8, 1.5], name="observed")

        >>> mae(fcst, obs)
        0.16666666666666674

    """
    error = _error(_to_series(fcst), _to_series(obs), is_angular)

    return float(error.abs().mean())
