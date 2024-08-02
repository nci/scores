"""
Implementation of some basic threshold weighted scoring functions.
See Taggart (2022) https://doi.org/10.1002/qj.4206
"""

import functools
from typing import Callable, Optional, Tuple, Union

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

AuxFuncType = Callable[[xr.DataArray], xr.DataArray]
EndpointType = Union[int, float, xr.DataArray]


def _check_tws_args(
    interval_where_one: Tuple[EndpointType, EndpointType],
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]],
):
    """
    Some argument checks for the threshold weighted scores.
    Checks for valid interval endpoints are done in :py:func`_auxiliary_funcs`.

    Args:
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
    """
    if len(interval_where_one) != 2:
        raise ValueError("`interval_where_one` must have length 2")

    if interval_where_positive is not None and len(interval_where_positive) != 2:
        raise ValueError("`interval_where_positive` must be length 2 when not `None`")


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

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.

    Raises:
        ValueError: if the left endpoint of ``interval_where_one`` is not less than
            the right endpoint.
        ValueError: if an ``interval_where_positive`` endpoint is infinite when the
            ``interval_where_one`` endpoint is infinite.
        ValueError: If the right endpoint of ``interval_where_positive`` is not greater
            than the right endpoint of ``interval_where_one`` and neither are infinite.
    """

    if interval_where_positive is None:  # rectangular threshold weight
        a, b = interval_where_one

        if isinstance(a, (float, int)):
            a = xr.DataArray(a)
            b = xr.DataArray(b)

        if (a >= b).any():
            raise ValueError("left endpoint of `interval_where_one` must be strictly less than right endpoint")

        # safest to work with finite a and b
        a = a.where(a > -np.inf, float(min(fcst.min(), obs.min(), b.min())) - 1)  # type: ignore
        b = b.where(b < np.inf, float(max(fcst.max(), obs.max(), a.max())) + 1)  # type: ignore

        g = functools.partial(_g_j_rect, a, b)
        phi = functools.partial(_phi_j_rect, a, b)
        phi_prime = functools.partial(_phi_j_prime_rect, a, b)

    else:  # trapezoidal threshold weight
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
        b = b.where(b > -np.inf, min(fcst.min(), obs.min(), c.min()) - 1)  # type: ignore
        a = a.where(a > -np.inf, b.min() - 1)  # type: ignore
        c = c.where(c < np.inf, max(fcst.max(), obs.max(), b.max()) + 1)  # type: ignore
        d = d.where(d < np.inf, c.max() + 1)  # type: ignore

        g = functools.partial(_g_j_trap, a, b, c, d)
        phi = functools.partial(_phi_j_trap, a, b, c, d)
        phi_prime = functools.partial(_phi_j_prime_trap, a, b, c, d)

    return g, phi, phi_prime


def _g_j_rect(a: EndpointType, b: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a nondecreasing function g_j, where g_j is obtained by integrating
    a rectangular threshold weight function. The threshold weight is 1 on the interval [a, b) and 0 otherwise.
    The formula is based on the first row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where the threshold weight = 1. Can be ``-np.inf``.
        b: right endpoint of the interval where the threshold weight = 1. Can be ``np.inf``.
        x: points where g_j is to be evaluated.

    Returns:
        array of function values for the function g_j.

    Note:
        Requires a < b. This is tested in :py:func:`_auxiliary_funcs`.
    """
    # results correspond to each case in the first row of Table B1, Taggart (2022).
    result1 = 0
    result2 = x - a
    result3 = b - a

    result = result2.where(x < b, result3)
    result = result.where(x >= a, result1)
    result = result.where(~np.isnan(x), np.nan)

    return result


def _phi_j_rect(a: EndpointType, b: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a convex function phi_j, where phi_j is obtained by integrating
    a rectangular threshold weight function. The threshold weight is 1 on the interval
    [a, b) and 0 otherwise.
    The formula is based on the second row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where the threshold weight = 1. Can be ``-np.inf``.
        b: right endpoint of the interval where the threshold weight = 1. Can be ``np.inf``.
        x: points where phi_j is to be evaluated.

    Returns:
        array of function values for the function phi_j.

    Notes:
        - Requires a < b. This is tested in :py:func:`_auxiliary_funcs`.
        - phi is this case is 2x^2 to be consistent with Taggart (2022) and is used to
            produce a twMSE. This means that a scaling factor is introduced for the
            threshold weighted expectile and Huber Loss scores.
    """
    # results correspond to each case in the second row of Table B1, Taggart (2022).
    result1 = 0
    result2 = 2 * (x - a) ** 2
    result3 = 4 * (b - a) * x + 2 * (a**2 - b**2)

    result = result2.where(x < b, result3)
    result = result.where(x >= a, result1)
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
    a trapezoidal threshold weight function. The threshold weight is 1 on the interval (b, c) and 0 outside
    the interval (a,d). The formula is based on the third row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where the threshold weight > 0.
        b: left endpoint of the interval where the threshold weight = 1.
        c: right endpoint of the interval where the threshold weight = 1.
        d: right endpoint of interval where the threshold weight > 0.
        x: points where g_j is to be evaluated.

    Returns:
        array of function values for the function g_j.

    Note:
        Requires a < b < c < d. This is tested in :py:func:`_auxiliary_funcs`.
    """
    # results correspond to each case in the third row of Table B1, Taggart (2022).
    result0 = 0
    result1 = (x - a) ** 2 / (2 * (b - a))
    result2 = x - (b + a) / 2
    result3 = -((d - x) ** 2) / (2 * (d - c)) + (d + c - a - b) / 2
    result4 = (d + c - a - b) / 2
    result = result1.where(x >= a, result0)
    result = result.where(x < b, result2)
    result = result.where(x < c, result3)
    result = result.where(x < d, result4)
    result = result.where(~np.isnan(x), np.nan)
    return result


def _phi_j_trap(a: EndpointType, b: EndpointType, c: EndpointType, d: EndpointType, x: xr.DataArray) -> xr.DataArray:
    """
    Returns values of a convex function phi_j, where phi_j is obtained by integrating
    a trapezoidal threshold weight function. The threshold weight is 1 on the interval (b, c) and 0 outside
    the interval (a,d). The formula is based on the fourth row of Table B1 from Taggart (2022).

    Args:
        a: left endpoint of interval where the threshold weight > 0.
        b: left endpoint of the interval where the threshold weight = 1.
        c: right endpoint of the interval where the threshold weight = 1.
        d: right endpoint of interval where the threshold weight > 0.
        x: points where phi_j is to be evaluated.

    Returns:
        array of function values for the function phi_j.

    Note:
        - Requires a < b < c < d. This is tested in :py:func`_auxiliary_funcs`.
        - phi is this case is 2x^2 to be consistent with Taggart (2022) and is used to
            produce a twMSE. This means that a scaling factor is introduced for the
            threshold weighted expectile and Huber Loss scores.
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

    result = result1.where(x >= a, result0)
    result = result.where(x < b, result2)
    result = result.where(x < c, result3)
    result = result.where(x < d, result4)
    result = result.where(~np.isnan(x), np.nan)

    return result


def _phi_j_prime_trap(
    a: EndpointType, b: EndpointType, c: EndpointType, d: EndpointType, x: xr.DataArray
) -> xr.DataArray:
    """
    Calculates the subderivative of :py:func`_phi_j_trap(a, b, c, d, x)` w.r.t. x.

    Args:
        a: left endpoint of interval where the threshold weight > 0.
        b: left endpoint of the interval where the threshold weight = 1.
        c: right endpoint of the interval where the threshold weight = 1.
        d: right endpoint of interval where the threshold weight > 0.
        x: points where phi_j is to be evaluated.

    Returns:
        The subderivative of :py:func`_phi_j_trap(a, b, c, d, x)` w.r.t. x.
    """
    return 4 * _g_j_trap(a, b, c, d, x)


def tw_squared_error(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the the threshold weighted squared error.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_expectile_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
          ``interval_where_one`` to be the interval where the threshold weight is 1.
          For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
          is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
          applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
          using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
          ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
          is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
          except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_square_error. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_squared_error. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_squared_error
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom). Note that these weights are different to threshold weighting
            which is done by decision threshold.

    Returns:
        xarray data array of the threshold weighted squared error

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` and ``interval_where_positive`` do not
            specify a valid trapezoidal weight.

    Reference:
        Taggart, R. (2022). Evaluation of point forecasts for extreme events using
        consistent scoring functions. Quarterly Journal of the Royal Meteorological
        Society, 148(742), 306-320. https://doi.org/10.1002/qj.4206
    """
    _check_tws_args(interval_where_one=interval_where_one, interval_where_positive=interval_where_positive)
    _, phi, phi_prime = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)

    return consistent_expectile_score(
        fcst, obs, 0.5, phi, phi_prime, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )


def tw_absolute_error(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the threshold weighted absolute error.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_quantile_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
          ``interval_where_one`` to be the interval where the threshold weight is 1.
          For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
          is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
          applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
          using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
          ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
          is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
          except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_absolute_error. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_absolute_error. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_absolute_error
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom). Note that these weights are different to threshold weighting
            which is done by decision threshold.

    Returns:
        xarray data array of the threshold weighted absolute error

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` and ``interval_where_positive`` do not
            specify a valid trapezoidal weight.

    References:
        Taggart, R. (2022). Evaluation of point forecasts for extreme events using
        consistent scoring functions. Quarterly Journal of the Royal Meteorological
        Society, 148(742), 306-320. https://doi.org/10.1002/qj.4206
    """
    _check_tws_args(interval_where_one=interval_where_one, interval_where_positive=interval_where_positive)
    g, _, _ = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)

    # Note that the absolute error is twice the quantile score when alpha=0.5
    return 2 * consistent_quantile_score(
        fcst, obs, 0.5, g, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )


def tw_quantile_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    alpha: float,
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the threshold weighted quantile score.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_quantile_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
          ``interval_where_one`` to be the interval where the threshold weight is 1.
          For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
          is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
          applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
          using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
          ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
          is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
          except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        alpha: the quantile level. Must be strictly between 0 and 1.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_quantile_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_quantile_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_quantile_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom). Note that these weights are different to threshold weighting
            which is done by decision threshold.

    Returns:
        xarray data array of the threshold weighted quantile error

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``alpha`` is not in the open interval (0,1).
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` and ``interval_where_positive`` do not
            specify a valid trapezoidal weight.

    References:
        Taggart, R. (2022). Evaluation of point forecasts for extreme events using
        consistent scoring functions. Quarterly Journal of the Royal Meteorological
        Society, 148(742), 306-320. https://doi.org/10.1002/qj.4206
    """

    check_alpha(alpha)
    _check_tws_args(interval_where_one=interval_where_one, interval_where_positive=interval_where_positive)
    g, _, _ = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)

    return consistent_quantile_score(
        fcst, obs, alpha, g, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )


def tw_expectile_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    alpha: float,
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the threshold weighted expectile score.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_expectile_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
          ``interval_where_one`` to be the interval where the threshold weight is 1.
          For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
          is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
          applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
          using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
          ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
          is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
          except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        alpha: expectile level. Must be strictly between 0 and 1.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_expectile_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_expectile_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_expectile_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom). Note that these weights are different to threshold weighting
            which is done by decision threshold.

    Returns:
        xarray data array of the threshold weighted expectile error

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``alpha`` is not in the open interval (0,1).
        ValueError: if ``interval_where_one`` and ``interval_where_positive`` do not
            specify a valid trapezoidal weight.

    References:
        Taggart, R. (2022). Evaluation of point forecasts for extreme events using
        consistent scoring functions. Quarterly Journal of the Royal Meteorological
        Society, 148(742), 306-320. https://doi.org/10.1002/qj.4206
    """

    check_alpha(alpha)
    _check_tws_args(interval_where_one=interval_where_one, interval_where_positive=interval_where_positive)
    _, phi, phi_prime = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)
    # We multiply the output by a factor of two here due to the scaling of phi and phi_prime
    # Since phi(s)=2s^2 was used in `_auxiliary_funcs` to be consistent with Taggart (2022)
    return 0.5 * consistent_expectile_score(
        fcst, obs, alpha, phi, phi_prime, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights
    )


def tw_huber_loss(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    huber_param: float,
    interval_where_one: Tuple[EndpointType, EndpointType],
    *,
    interval_where_positive: Optional[Tuple[EndpointType, EndpointType]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the threshold weighted Huber loss.

    For more flexible threshold weighting schemes,
    see :py:func:`scores.continuous.consistent_huber_score`.

    Two types of threshold weighting are supported: rectangular and trapezoidal.
        - To specify a rectangular threshold weight, set ``interval_where_positive=None`` and set
          ``interval_where_one`` to be the interval where the threshold weight is 1.
          For example, if  ``interval_where_one=(0, 10)`` then a threshold weight of 1
          is applied to decision thresholds satisfying 0 <= threshold < 10, and a threshold weight of 0 is
          applied otherwise. Interval endpoints can be ``-numpy.inf`` or ``numpy.inf``.
        - To specify a trapezoidal threshold weight, specify ``interval_where_positive`` and ``interval_where_one``
          using desired endpoints. For example, if ``interval_where_positive=(-2, 10)`` and
          ``interval_where_one=(2, 4)`` then a threshold weight of 1 is applied to decision thresholds
          satisfying 2 <= threshold < 4. The threshold weight increases linearly from 0 to 1 on the interval
          [-2, 2) and decreases linearly from 1 to 0 on the interval [4, 10], and is 0 otherwise.
          Interval endpoints can only be infinite if the corresponding ``interval_where_one`` endpoint
          is infinite. End points of ``interval_where_positive`` and ``interval_where_one`` must differ
          except when the endpoints are infinite.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        huber_param: the Huber transition parameter.
        interval_where_one: endpoints of the interval where the threshold weights are 1.
            Must be increasing. Infinite endpoints are permissible. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        interval_where_positive: endpoints of the interval where the threshold weights are positive.
            Must be increasing. Infinite endpoints are only permissible when the corresponding
            ``interval_where_one`` endpoint is infinite. By supplying a tuple of
            arrays, endpoints can vary with dimension.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the threshold_weighted_expectile_score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the threshold_weighted_expectile_score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the threshold_weighted_expectile_score
            at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom). Note that these weights are different to threshold weighting
            which is done by decision threshold.

    Returns:
        xarray data array of the threshold weighted expectile error

    Raises:
        ValueError: if ``interval_where_one`` is not length 2.
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``alpha`` is not in the open interval (0,1).
        ValueError: if ``huber_param`` is not positive
        ValueError: if ``interval_where_one`` is not increasing.
        ValueError: if ``interval_where_one`` and ``interval_where_positive`` do not
            specify a valid trapezoidal threshold weight.

    References:
        Taggart, R. (2022). Evaluation of point forecasts for extreme events using
        consistent scoring functions. Quarterly Journal of the Royal Meteorological
        Society, 148(742), 306-320. https://doi.org/10.1002/qj.4206
    """

    check_huber_param(huber_param)
    _check_tws_args(interval_where_one=interval_where_one, interval_where_positive=interval_where_positive)
    _, phi, phi_prime = _auxiliary_funcs(fcst, obs, interval_where_one, interval_where_positive)

    # We multiply the output by a factor of two here due to the scaling of phi and phi_prime
    # Since phi(s)=2s^2 was used in `_auxiliary_funcs` to be consistent with Taggart (2022)
    return 0.5 * consistent_huber_score(
        fcst,
        obs,
        huber_param,
        phi,
        phi_prime,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
