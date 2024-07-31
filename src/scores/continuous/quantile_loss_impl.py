"""
Implementation of quantile loss (score)
"""
from typing import Optional

import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_dims, gather_dimensions


def quantile_score(
    fcst: XarrayLike,
    obs: XarrayLike,
    alpha: float,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates a score that targets alpha-quantiles.
    Use with alpha = 0.5 for forecasts of the median.
    Use with alpha = 0.9 for forecasts of the 90th percentile.

    Args:
        fcst: array of forecasts
        obs: array of observations
        alpha: A value between 0 and 1 (exclusive)
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the quantile score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            quantile score. All other dimensions will be reduced. As a special case, 'all'
            will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the quantile
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of `reduce_dims`
            and `preserve_dims` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        A DataArray with values being the mean generalised piecewise linear (GPL)
        scoring function, with the dimensions specified in `dims`.
        If `dims` is `None`, the returned DataArray will have only one element,
        the overall mean GPL score.

    Raises:
        ValueError: if `alpha` is not between 0 and 1.

    Notes:

        .. math::

            gpl(x) = \\begin{cases}\\alpha * (-x) & x \\leq 0\\\\
           (1-\\alpha) x & x > 0\\end{cases}

        where:
            - :math:`\\alpha` is the targeted quantile.
            - :math:`x` is the difference, fcst - obs

    References:
        T. Gneiting, "Making and evaluating point forecasts",
        J. Amer. Stat. Assoc., Vol. 106 No. 494 (June 2011), pp. 754--755,
        Theorem 9

    """
    specified_dims = reduce_dims or preserve_dims
    # check requested dims are a subset of fcst dimensions
    if specified_dims is not None:
        check_dims(xr_data=fcst, expected_dims=specified_dims, mode="superset")
    # check obs dimensions are a subset of fcst dimensions
    check_dims(xr_data=obs, expected_dims=fcst.dims, mode="subset")  # type: ignore

    # check that alpha is between 0 and 1 as required
    if (alpha <= 0) or (alpha >= 1):
        raise ValueError("alpha is not between 0 and 1")

    # Do this operation once to save compute time
    diff = fcst - obs

    # calculate the score applicable when fcst <= obs
    score_fcst_lte_obs = alpha * (-diff)

    # calculate the score applicable when fcst > obs
    score_fcst_ge_obs = (1 - alpha) * diff

    result = xr.where(diff > 0, score_fcst_ge_obs, score_fcst_lte_obs)

    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)  # type: ignore[assignment]
    results = apply_weights(result, weights=weights)
    score = results.mean(dim=reduce_dims)

    return score  # type: ignore
