"""
Implementation of scoring functions that are consistent for
single-valued forecasts targeting quantiles, expectiles or Huber functionals.
"""

from typing import Callable, Optional

import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes
from scores.utils import gather_dimensions


def consistent_expectile_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    alpha: float,
    phi: Callable[[xr.DataArray], xr.DataArray],
    phi_prime: Callable[[xr.DataArray], xr.DataArray],
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Returns the score using a scoring function that is consistent for the
    alpha-expectile functional, based on a supplied convex function phi.
    See Geniting (2011), or Equation (10) from Taggart (2021).

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        alpha: expectile level. Must be strictly between 0 and 1.
        phi: a convex function on the real numbers, accepting a single array like argument.
        phi_prime: a subderivative of `phi`, accepting a single array like argument.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the consistent expectile score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the consistent quantile score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the consistent quantile
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        array of (mean) scores that is consistent for alpha-expectile functional,
        with the dimensions specified by `dims`. If `dims` is `None`, the returned DataArray will have
        only one entry, the overall mean score.

    Raises:
        ValueError: if `alpha` is not strictly between 0 and 1.

    Note:
        .. math::

            S(x, y) =
            \\begin{cases}
            (1 - \\alpha)(\\phi(y) - \\phi(x) - \\phi'(x)(y-x)), & y < x \\\\
            \\alpha(\\phi(y) - \\phi(x) - \\phi'(x)(y-x)), & x \\leq y
            \\end{cases}

        where

            - :math:`x` is the forecast
            - :math:`y` is the observation
            - :math:`\\alpha` is the expectile level
            - :math:`\\phi` is a convex function of a single variable
            - :math:`\\phi'` is the subderivative of :math:`\\phi`
            - :math:`S(x,y)` is the score.

        Note that if :math:`\\phi` is differentiable then `\\phi'` is its derivative.

    References:
        -   Gneiting, T. (2011). "Making and evaluating point forecasts",
            J. Amer. Statist. Assoc.,
            https://doi.org/10.1198/jasa.2011.r10138
        -   Taggart, R. (2021). "Evaluation of point forecasts for extreme events
            using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
            https://doi.org/10.1002/qj.4206

    """
    check_alpha(alpha)

    if preserve_dims or reduce_dims:
        reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

    score_overfcst = (1 - alpha) * (phi(obs) - phi(fcst) - phi_prime(fcst) * (obs - fcst))
    score_underfcst = alpha * (phi(obs) - phi(fcst) - phi_prime(fcst) * (obs - fcst))
    result = score_overfcst.where(obs < fcst, score_underfcst)
    result = apply_weights(result, weights=weights)
    result = result.mean(dim=reduce_dims)

    return result


def consistent_huber_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    huber_param: float,
    phi: Callable[[xr.DataArray], xr.DataArray],
    phi_prime: Callable[[xr.DataArray], xr.DataArray],
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Score that is consistent for the Huber mean functional with tuning parameter `tuning_param`,
    based on convex function phi. See Taggart (2022), or Equation (11) from Taggart (2021).
    See Taggart (2021), end of Section 3.4, for the standard formula.

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        huber_param: Huber mean tuning parameter. This corresponds to the transition point between
            linear and quadratic loss for Huber loss. Must be positive.
        phi: a convex function on the real numbers, accepting a single array like argument.
        phi_prime: a subderivative of `phi`, accepting a single array like argument.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the consistent Huber score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the consistent Huber score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the consistent Huber
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        array of (mean) scores that is consistent for Huber mean functional,
        with the dimensions specified by `dims`. If `dims` is `None`, the returned DataArray will have
        only one entry, the overall mean score.

    Raises:
       ValueError: if `huber_param <= 0`.

    References:
        -   Taggart, R. (2021). "Evaluation of point forecasts for extreme events
            using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
            https://doi.org/10.1002/qj.4206
        -   Taggart, R. (2022). "Point forecasting and forecast evaluation with
            generalized Huber loss", Electron. J. Statist. 16(1): 201-231.
            DOI: 10.1214/21-EJS1957
    """
    check_huber_param(huber_param)
    if preserve_dims or reduce_dims:
        reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

    kappa = (fcst - obs).clip(min=-huber_param, max=huber_param)
    result = 0.5 * (phi(obs) - phi(kappa + obs) + kappa * phi_prime(fcst))
    result = apply_weights(result, weights=weights)
    result = result.mean(dim=reduce_dims)

    return result


def consistent_quantile_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    alpha: float,
    g: Callable[[xr.DataArray], xr.DataArray],
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Score that is consistent for the alpha-quantile functional, based on nondecreasing function g.
    See Gneiting (2011), or Equation (8) from Taggart (2022).

    Args:
        fcst: array of forecast values.
        obs: array of corresponding observation values.
        alpha: quantile level. Must be strictly between 0 and 1.
        g: nondecreasing function on the real numbers, accepting a single array like argument.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the consistent quantile score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the consistent quantile score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the consistent quantile
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        array of (mean) scores that are consistent for alpha-quantile functional,
        with the dimensions specified by `dims`. If `dims` is `None`, the returned DataArray will have
        only one entry, the overall mean score.

    Raises:
        ValueError: if `alpha` is not strictly between 0 and 1.

    Note:
        .. math::

            S(x, y) =
            \\begin{cases}
            (1 - \\alpha)(g(x) - g(y)), & y < x \\\\
            \\alpha(g(y) - g(x)), & x \\leq y
            \\end{cases}

        where

            - :math:`x` is the forecast
            - :math:`y` is the observation
            - :math:`\\alpha` is the quantile level
            - :math:`g` is a nondecreasing function of a single variable
            - :math:`S(x,y)` is the score.

        Note that if :math:`\\phi` is differentiable then `\\phi'` is its derivative.

    References:
        -   Gneiting, T. (2011). "Making and evaluating point forecasts",
            J. Amer. Statist. Assoc.,
            https://doi.org/10.1198/jasa.2011.r10138
        -   Taggart, R. (2021). "Evaluation of point forecasts for extreme events
            using consistent scoring functions", Q. J. Royal Meteorol. Soc.,
            https://doi.org/10.1002/qj.4206
    """
    check_alpha(alpha)
    if preserve_dims or reduce_dims:
        reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

    score_overfcst = (1 - alpha) * (g(fcst) - g(obs))
    score_underfcst = -alpha * (g(fcst) - g(obs))
    result = score_overfcst.where(obs < fcst, score_underfcst)
    result = apply_weights(result, weights=weights)
    result = result.mean(dim=reduce_dims)

    return result


def check_alpha(alpha: float) -> None:
    """Raises if quantile or expectile level `alpha` not in the open interval (0,1)."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("`alpha` must be strictly between 0 and 1")


def check_huber_param(huber_param: float) -> None:
    """Raises if `huber_param` is not positive."""
    if huber_param <= 0:
        raise ValueError("`huber_param` must be positive")