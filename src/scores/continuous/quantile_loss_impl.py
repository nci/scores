"""
Implementation of quantile loss (score)
"""

from typing import Optional

import xarray as xr

from scores.processing import aggregate
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
        weights: An array of weights to apply to the score (e.g., weighting a grid by latitude).
            If None, no weights are applied. If provided, the weights must be broadcastable
            to the data dimensions and must not contain negative or NaN values. If
            appropriate, users can choose to replace NaN values in weights by calling ``weights.fillna(0)``.
            The weighting approach follows :py:class:`xarray.computation.weighted.DataArrayWeighted`.
            See the scores weighting tutorial for more information on how to use weights.

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
        - T. Gneiting, "Making and evaluating point forecasts",
          J. Amer. Stat. Assoc., Vol. 106 No. 494 (June 2011), pp. 754--755,
          Theorem 9

    Examples:
        >>> import xarray as xr
        >>> from scores.continuous import quantile_score
        >>> times = ["2024-01-01", "2024-01-02"]
        >>> lats = [-35, -30, -25]
        >>> lons = [140, 150]
        >>> obs = xr.DataArray(
        ...       [[[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
        ...       [[1.5, 1.6], [2.5, 2.6], [3.5, 3.6]]],
        ...       coords={"time": times, "lat": lats, "lon": lons},
        ...       dims=["time", "lat", "lon"]
        ...       )
        >>> fcst = xr.DataArray(
        ...       [[[1.3, 1.0], [1.7, 2.7], [3.6, 3.0]],
        ...       [[1.0, 1.1], [2.2, 2.7], [3.3, 3.9]]],
        ...       coords={"time": times, "lat": lats, "lon": lons},
        ...       dims=["time", "lat", "lon"]
        ...       )
        >>> quantile_score(fcst, obs, alpha=0.5)
        <xarray.DataArray ()> Size: 8B
        array(0.1625)
        >>> quantile_score(fcst, obs, alpha=0.5, preserve_dims=["time"])
        <xarray.DataArray (time: 2)> Size: 16B
        array([0.16666667, 0.15833333])
        Coordinates:
        * time     (time) <U10 80B '2024-01-01' '2024-01-02'

    """
    specified_dims = reduce_dims or preserve_dims
    # check requested dims are a subset of fcst dimensions
    if specified_dims is not None and specified_dims != "all":
        check_dims(xr_data=fcst, expected_dims=specified_dims, mode="superset")
    # check obs dimensions are a subset of fcst dimensions
    check_dims(xr_data=obs, expected_dims=fcst.dims, mode="subset")
    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

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

    score = aggregate(result, weights=weights, reduce_dims=reduce_dims)

    return score
