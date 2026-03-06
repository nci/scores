import scores
from scores import continuous as __continuous


def mse(
    fcst,
    obs,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
):
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
    return __continuous.mse(fcst, obs, is_angular=is_angular)


def additive_bias(
    fcst,
    obs,
    *,
    reduce_dims=None,
    weights=None,
):
    """
    Calculates the additive bias which is also sometimes called the mean error.

    It is defined as

    .. math::
        \\text{Additive bias} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}


    See "Mean error" section at https://jwgfvr.github.io/forecastverification/index.html#meanerror
    for more information.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the additive bias. All other dimensions will be preserved.
        weights: An array of weights to apply to the score (e.g., weighting a grid by latitude).
            If None, no weights are applied. If provided, the weights must be broadcastable
            to the data dimensions and must not contain negative or NaN values. If
            appropriate, NaN values in weights  can be replaced by ``weights.fillna(0)``.
            The weighting approach follows :py:class:`xarray.computation.weighted.DataArrayWeighted`.
            See the ``scores`` weighting tutorial for more information on how to use weights.

    Returns:
        An xarray object with the additive bias of a forecast.

    References:
        -   https://jwgfvr.github.io/forecastverification/index.html#meanerror

    """

    error = fcst - obs
    score = scores.processing.aggregate(error, reduce_dims="all", weights=weights)

    return score
