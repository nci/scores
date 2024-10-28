"""
Import the functions from the implementations into the public API
"""

from scores import continuous as __continuous
from scores.pandas.typing import PandasType


def mse(
    fcst: PandasType,
    obs: PandasType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> PandasType:
    """Calculates the mean squared error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_squared_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2


    Notes:
        Dimensional reduction is not supported for pandas and the user should
        convert their data to xarray to formulate the call to the base metric,
        `scores.continuous.mse`.

    Args:
        fcst: Forecast or predicted variables in pandas.
        obs: Observed variables in pandas.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        pandas.Series:
            An object containing a single floating point number representing the mean squared
            error for the supplied data. All dimensions will be reduced.

    """
    return __continuous.mse(fcst, obs, is_angular=is_angular)  # type: ignore  # mypy is wrong, I think


def rmse(
    fcst: PandasType,
    obs: PandasType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> PandasType:
    """Calculate the Root Mean Squared Error from xarray or pandas objects.

    A detailed explanation is on https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math ::
        \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2}

    Notes:
        Dimensional reduction is not supported for pandas and users wishing this extra functionality
        should convert their data to xarray to formulate the call to `scores.continuous.rmse`.

    Args:
        fcst: Forecast or predicted variables in pandas.
        obs: Observed variables in pandas.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        pandas.Series:
            An object containing a single floating point number representing the root mean squared
            error for the supplied data. All dimensions will be reduced.

    """
    return __continuous.rmse(fcst, obs, is_angular=is_angular)  # type: ignore  # mypy is wrong, I think


def mae(
    fcst: PandasType,
    obs: PandasType,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
) -> PandasType:
    """Calculates the mean absolute error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n | \\text{forecast}_i - \\text{observed}_i |

    Notes:
        Dimensional reduction is not supported for pandas and the user should
        convert their data to xarray to formulate the call to `scores.continuous.mae`.

    Args:
        fcst: Forecast or predicted variables in pandas.
        obs: Observed variables in pandas.
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        pandas.Series: An object containing a single floating point number representing
        the mean absolute error for the supplied data. All dimensions will be reduced.

    """
    return __continuous.mae(fcst, obs, is_angular=is_angular)  # type: ignore  # mypy is wrong, I think
