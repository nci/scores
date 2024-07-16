"""
Implementation of some basic threshold weighted scoring functions.
See Taggart (2022) https://doi.org/10.1002/qj.4206
"""

import functools
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr

from scores.continuous.consistent_impl import (
    check_alpha,
    check_huber_param,
    consistent_expectile_score,
    consistent_huber_score,
    consistent_quantile_score,
)
from scores.typing import FlexibleDimensionTypes

HUBER_FUNCS = ["huber_loss", "scaled_huber_loss"]
ALPHA_FUNCS = ["quantile_score", "expectile_score"]
OTHER_FUNCS = ["squared_error", "absolute_error"]
SCORING_FUNCS = OTHER_FUNCS + ALPHA_FUNCS + HUBER_FUNCS
AuxFuncType = Callable[[xr.DataArray], xr.DataArray]
EndpointType = Union[int, float, xr.DataArray]


def threshold_weighted_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    scoring_func: Literal[
        "squared_error",
        "absolute_error",
        "quantile_score",
        "expectile_score",
        "huber_loss",
        "scaled_huber_loss",
    ],
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    alpha: Optional[float] = None,
    huber_param: Optional[float] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the scores calculated using the specified threshold weighted scoring function,
    averaged over the specified dimensions in `dims`.

    Taggart (2022) showed that many commonly used scoring functions, such as the squared
    error and absolute error scoring functions, can be decomposed as a sum of weighted
    scoring functions, where the weights sum to 1, and where the original scoring function and
    each member of its decomposition are consistent for the same statistical functional.
    Each member of the decomposition measures predictive performance with emphasis on the
    region specified by the corresponding weight.

    The function `threshold_weighted_score` supports threshold weighting for the following
    scoring functions:

        - squared error (see `mse`): :math:`S(x, y) = (x - y)^2`
        - absolute error: :math:`S(x, y) = |x - y|`
        - quantile score (see `quantile_score`): :math:`S(x, y) = (1 - \\alpha)(x - y)`
          if :math:`x \\geq y` and :math:`S(x, y) = \\alpha(y - x)` if :math:`x < y`
        - expectile score: :math:`S(x, y) = (1 - \\alpha)(x - y)^2`
          if :math:`x \\geq y` and :math:`S(x, y) = \\alpha(y - x)^2` if :math:`x < y`
        - Huber loss: :math:`S(x, y) = \\frac\\{1\\}\\{2\\}(x-y)^2` if :math:`|x - y| \\leq a` and
          :math:`S(x, y) = a|x-y| - a^2/2` if :math:`|x - y| > a`
        - Scaled Huber loss: Huber loss scaled by a factor of :math:`1/a`

    Where

        - :math:`S(x, y)` is the score,
        - :math:`x` is the forecast,
        - :math:`y` the observation,
        - :math:`\\alpha` is the quantile level in the case of the `quantile_score` and
            the expectile level in the case of the `expectile_square`,
        - :math:`a` is the Huber parameter.

    Two types of threshold weighting are supported: rectangular and trapezoidal.

        - To specify a rectangular weight, set `interval_where_positive=None` and set
          `interval_where_one` to be the interval where the weight is 1.
          For example, if  `interval_where_one=(0, 10)` then a weight of 1 is applied to decision thresholds
          satisfying 0 <= threshold < 10, and weight of 0 is applied otherwise.
          Interval endpoints can be `-numpy.inf` or `numpy.inf`.
        - To specify a trapezoidal weight, specify `interval_where_positive` and `interval_where_one`
          using desired endpoints. For example, if `interval_where_positive=(-2, 10)` and
          `interval_where_one=(2, 4)` then a weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding `interval_where_one` endpoint
          is infinite. End points of `interval_where_positive` and `interval_where_one` must differ
          except when the endpoints are infinite.

    Users who prefer more complex weighting systems should use one of the functions
    `scores.continuous.consistent_quantile_score`, `scores.continuous.consistent_expectile_score`,
    or `scores.continuous.huber_score`.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        scoring_func: name of the scoring function to apply threshold weighting to.
            Must be one of 'squared_error', 'absolute_error', 'quantile_score',
            'expectile_score', 'huber_loss' or 'scaled_huber_loss'.
        interval_where_one: endpoints of the interval where the weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            `interval_where_one` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
            If `None`, then it is assumed that the weights are positive only where they are 1.
        alpha: the quantile level or expectile level, if scoring_func is 'quantile_score' or
            'expectile_score'.
        huber_param: the Huber transition parameter if scoring_func is  'huber_loss' or
            'scaled_huber_loss'.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        array of threshold weighted scores, with the dimensions specified by `dims`.
        If `dims` is `None`, the returned DataArray will have only one entry, the overall mean score.

    Raises:
        ValueError: if `interval_where_one` is not length 2.
        ValueError: if `interval_where_positive` is not length 2 when it is not `None`.
        ValueError: if `scoring_func` is not a valid string.
        ValueError: if `alpha` is None or if `alpha` is not in the open interval (0,1)
            whenever `scoring_func` is 'quantile_score' or 'expectile_score'.
        ValueError: if `huber_param` is None or if `huber_param` in not positive
            whenever `scoring_func` is 'huber_loss' or 'scaled_huber_loss'.
        ValueError: if `interval_where_one` is not increasing.
        ValueError: if `interval_where_one` and `interval_where_positive` do not
            specify a valid trapezoidal weight.

    Notes:
        - 'quantile_score' uses the same underlying scoring function as `quantile_scoring_function`.
        - 'expectile_score' is asymmetric squared loss. When alpha = 0.5, it is half squared loss.
        - 'huber_loss' is the standard Huber loss. It is related to scaled_huber_loss via
          huber_loss = scaled_huber_loss * huber_parameter.

    References:
        Taggart, R. (2022). "Evaluation of point forecasts for extreme events
        using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
        https://doi.org/10.1002/qj.4206

    """
    _check_tws_args(scoring_func, alpha, huber_param, interval_where_one, interval_where_positive)

    g, phi, phi_prime = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)

    # calculate the scores
    if scoring_func == "absolute_error":
        result = 2 * consistent_quantile_score(
            fcst, obs, 0.5, g, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
        )

    if scoring_func == "quantile_score":
        result = consistent_quantile_score(
            fcst, obs, alpha, g, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
        )

    if scoring_func == "squared_error":
        result = consistent_expectile_score(
            fcst, obs, 0.5, phi, phi_prime, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
        )

    if scoring_func == "expectile_score":
        result = 0.5 * consistent_expectile_score(
            fcst, obs, alpha, phi, phi_prime, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
        )

    if scoring_func == "huber_loss":
        result = 0.5 * consistent_huber_score(
            fcst,
            obs,
            huber_param,
            phi,
            phi_prime,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
            weights=weights,
        )

    if scoring_func == "scaled_huber_loss":
        result = (
            0.5
            * consistent_huber_score(
                fcst,
                obs,
                huber_param,
                phi,
                phi_prime,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
            )
            / huber_param
        )

    return result


def _check_tws_args(scoring_func, alpha, huber_param, interval_where_one, interval_where_positive):
    """
    Some argument checks for `threshold_weighted_score`.
    Checks for valid interval endpoints are done in `_auxiliary_funcs`.
    """
    if len(interval_where_one) != 2:
        raise ValueError("`interval_where_one` must have length 2")

    if interval_where_positive is not None and len(interval_where_positive) != 2:
        raise ValueError("`interval_where_positive` must be length 2 when not `None`")

    if scoring_func not in SCORING_FUNCS:
        raise ValueError("`scoring_func` must be one of:" + ", ".join(SCORING_FUNCS))

    if scoring_func in ALPHA_FUNCS:
        if alpha is None:
            raise ValueError(f"When `scoring_func={scoring_func}, `alpha` must be supplied")
        check_alpha(alpha)

    if scoring_func in HUBER_FUNCS:
        if huber_param is None:
            raise ValueError(f"When `scoring_func={scoring_func}, `huber_param` must be supplied")
        check_huber_param(huber_param)


def _auxiliary_funcs(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]],
) -> Tuple[AuxFuncType, AuxFuncType, AuxFuncType]:
    """
    Returns the three auxiliary functions g, phi and phi_prime
    which are used to construct quantile, expectile or Huber mean scoring functions.
    See Equations (8), (10) and (11) from Taggart (2022) for the role of g, phi and phi_prime.
    """

    if interval_where_positive is None:  # rectangular weight
        a, b = interval_where_one

        if isinstance(a, (float, int)):
            a = xr.DataArray(a)
            b = xr.DataArray(b)

        if (a >= b).any():
            raise ValueError("left endpoint of `interval_where_one` must be strictly less than right endpoint")

        # safest to work with finite a and b
        a = a.where(a > -np.inf, float(min(fcst.min(), obs.min(), b.min())) - 1)
        b = b.where(b < np.inf, float(max(fcst.max(), obs.max(), a.max())) + 1)

        g = functools.partial(_g_j_rect, a, b)
        phi = functools.partial(_phi_j_rect, a, b)
        phi_prime = functools.partial(_phi_j_prime_rect, a, b)

    else:  # trapezoidal weight
        a, d = interval_where_positive
        b, c = interval_where_one

        if isinstance(a, (float, int)):
            a = xr.DataArray(a)
            d = xr.DataArray(d)

        if isinstance(b, (float, int)):
            b = xr.DataArray(b)
            c = xr.DataArray(c)

        if (b >= c).any():
            raise ValueError("left endpoint of `interval_where_one` must be strictly less than right endpoint")

        if (np.isinf(a) & (a != b)).any() or (np.isinf(d) & (c != d)).any():
            raise ValueError(
                "`interval_where_positive` endpoint can only be infinite when "
                "corresponding `interval_where_one` endpoint is infinite."
            )

        if not ((a < b) | ((a == b) & np.isinf(a))).all():
            raise ValueError(
                "left endpoint of `interval_where_positive` must be less than "
                "left endpoint of `interval_where_one`, unless both are `-numpy.inf`."
            )

        if not ((c < d) | ((c == d) & np.isinf(c))).all():
            raise ValueError(
                "right endpoint of `interval_where_positive` must be greater than "
                "right endpoint of `interval_where_one`, unless both are `numpy.inf`."
            )

        # safest to work with finite intervals
        b = b.where(b > -np.inf, min(fcst.min(), obs.min(), c.min()) - 1)
        a = a.where(a > -np.inf, b.min() - 1)
        c = c.where(c < np.inf, max(fcst.max(), obs.max(), b.max()) + 1)
        d = d.where(d < np.inf, c.max() + 1)

        g = functools.partial(_g_j_trap, a, b, c, d)
        phi = functools.partial(_phi_j_trap, a, b, c, d)
        phi_prime = functools.partial(_phi_j_prime_trap, a, b, c, d)

    return g, phi, phi_prime


def _g_j_rect(a: EndpointType, b: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a nondecreasing function g_j, where g_j is obtained by integrating
    a rectangular weight function. The weight is 1 on the interval [a, b) and 0 otherwise.
    The formula is based on the first row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where weight = 1. Can be `-np.inf`.
        b: right endpoint of the interval where weight = 1. Can be `np.inf`.
        x: points where g_j is to be evaluated.

    Returns:
        array of function values for the function g_j.

    Note:
        Requires a < b. This is tested in `_auxiliary_funcs`.
    """
    # results correspond to each case in the first row of Table B1, Taggart (2022).
    result1 = 0
    result2 = x - a
    result3 = b - a

    result = result2.where(x < b, result3).where(x >= a, result1)
    result = result.where(~np.isnan(x), np.nan)

    return result


def _phi_j_rect(a: EndpointType, b: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a convex function phi_j, where phi_j is obtained by integrating
    a rectangular weight function. The weight is 1 on the interval [a, b) and 0 otherwise.
    The formula is based on the second row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where weight = 1. Can be `-np.inf`.
        b: right endpoint of the interval where weight = 1. Can be `np.inf`.
        x: points where phi_j is to be evaluated.

    Returns:
        array of function values for the function phi_j.

    Note:
        Requires a < b. This is tested in `_auxiliary_funcs`.
    """
    # results correspond to each case in the second row of Table B1, Taggart (2022).
    result1 = 0
    result2 = 2 * (x - a) ** 2
    result3 = 4 * (b - a) * x + 2 * (a**2 - b**2)

    result = result2.where(x < b, result3).where(x >= a, result1)
    result = result.where(~np.isnan(x), np.nan)

    return result


def _phi_j_prime_rect(a: EndpointType, b: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    The subderivative of `_phi_j_rect(a, b, x)` w.r.t. x.
    """
    return 4 * _g_j_rect(a, b, x)


def _g_j_trap(a: EndpointType, b: EndpointType, c: EndpointType, d: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a nondecreasing function g_j, where g_j is obtained by integrating
    a trapezoidal weight function. The weight is 1 on the interval (b, c) and 0 outside
    the interval (a,d). The formula is based on the third row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where weight > 0.
        b: left endpoint of the interval where weight = 1.
        c: right endpoint of the interval where weight = 1.
        d: right endpoint of interval where weight > 0.
        x: points where g_j is to be evaluated.

    Returns:
        array of function values for the function g_j.

    Note:
        Requires a < b < c < d. This is tested in `_auxiliary_funcs`.
    """
    # results correspond to each case in the third row of Table B1, Taggart (2022).
    result0 = 0
    result1 = (x - a) ** 2 / (2 * (b - a))
    result2 = x - (b + a) / 2
    result3 = -((d - x) ** 2) / (2 * (d - c)) + (d + c - a - b) / 2
    result4 = (d + c - a - b) / 2
    result = (
        result1.where(x >= a, result0)
        .where(x < b, result2)
        .where(x < c, result3)
        .where(x < d, result4)
        .where(~np.isnan(x), np.nan)
    )
    return result


def _phi_j_trap(a: EndpointType, b: EndpointType, c: EndpointType, d: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a convex function phi_j, where phi_j is obtained by integrating
    a trapezoidal weight function. The weight is 1 on the interval (b, c) and 0 outside
    the interval (a,d). The formula is based on the fourth row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where weight > 0.
        b: left endpoint of the interval where weight = 1.
        c: right endpoint of the interval where weight = 1.
        d: right endpoint of interval where weight > 0.
        x: points where phi_j is to be evaluated.

    Returns:
        array of function values for the function phi_j.

    Note:
        Requires a < b < c < d. This is tested in `_auxiliary_funcs`.
    """
    # results correspond to each case in the fourth row of Table B1, Taggart (2022).
    result0 = 0
    result1 = 2 * (x - a) ** 3 / (3 * (b - a))
    result2 = 2 * x**2 - 2 * (a + b) * x + 2 * (b - a) ** 2 / 3 + 2 * a * b
    result3 = (
        2 * (d - x) ** 3 / (3 * (d - c))
        + 2 * (d + c - a - b) * x
        + 2 * ((b - a) ** 2 + 3 * a * b - (d - c) ** 2 - 3 * c * d) / 3
    )
    result4 = 2 * (d + c - a - b) * x + 2 * ((b - a) ** 2 + 3 * a * b - (d - c) ** 2 - 3 * c * d) / 3
    result = (
        result1.where(x >= a, result0)
        .where(x < b, result2)
        .where(x < c, result3)
        .where(x < d, result4)
        .where(~np.isnan(x), np.nan)
    )
    return result


def _phi_j_prime_trap(
    a: EndpointType, b: EndpointType, c: EndpointType, d: EndpointType, x: xr.DataArray
) -> xr.DataArray:
    """
    The subderivative of `_phi_j_trap(a, b, c, d, x)` w.r.t. x.
    """
    return 4 * _g_j_trap(a, b, c, d, x)


def threshold_weighted_squared_error(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the scores computed using a threshold weighted scoring
    function derived from the squared error function.

    This is a convenience function for `threshold_weighted_score`
    with the option `scoring_func="squared_error"`. The weight is
    1 on the specified interval and 0 elsewhere. For more flexible weighting schemes,
    see `scores.continuous.threshold_weighted_score` and `scores.continuous.consistent_expectile_score`.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_square_error. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_squared_error. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_squared_error
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        xarray data array of the threshold weighted squared error

    Raises:
        ValueError: if `interval_where_one` is not length 2.
        ValueError: if `interval_where_one` is not increasing.

    References:
        Taggart, R. (2022). "Evaluation of point forecasts for extreme events
        using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
        https://doi.org/10.1002/qj.4206
    """
    return threshold_weighted_score(
        fcst,
        obs,
        "squared_error",
        interval_where_one,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )


def threshold_weighted_absolute_error(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the scores computed using a threshold weighted scoring
    function derived from the absolute error function.

    This is a convenience function for `threshold_weighted_score`
    with the option `scoring_func="absolute_error"`. The weight is
    1 on the specified interval and 0 elsewhere. For more flexible weighting schemes,
    see `scores.continuous.threshold_weighted_score` and `scores.continuous.consistent_absolute_error`.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_absolute_error. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_absolute_error. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_absolute_error
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        xarray data array of the threshold weighted absolute error

    Raises:
        ValueError: if `interval_where_one` is not length 2.
        ValueError: if `interval_where_one` is not increasing.

    References:
        Taggart, R. (2022). "Evaluation of point forecasts for extreme events
        using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
        https://doi.org/10.1002/qj.4206
    """
    return threshold_weighted_score(
        fcst,
        obs,
        "absolute_error",
        interval_where_one,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )


def threshold_weighted_quantile_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    alpha: float,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the scores computed using a threshold weighted scoring
    function derived from the standard `quantile_score`.

    This is a convenience function for `threshold_weighted_score`
    with the option `scoring_func="quantile_score"`. The weight is
    1 on the specified interval and 0 elsewhere. For more flexible weighting schemes,
    see `scores.continuous.threshold_weighted_score` and `scores.continuous.consistent_quantile_score`.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        alpha: the quantile level.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_quantile_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_quantile_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_quantile_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        xarray data array of the threshold weighted quantile error

    Raises:
        ValueError: if `interval_where_one` is not length 2.
        ValueError: if `interval_where_one` is not increasing.
        ValueError: if `alpha` is not in the open interval (0,1).

    References:
        Taggart, R. (2022). "Evaluation of point forecasts for extreme events
        using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
        https://doi.org/10.1002/qj.4206
    """
    return threshold_weighted_score(
        fcst,
        obs,
        "quantile_score",
        interval_where_one,
        alpha=alpha,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
