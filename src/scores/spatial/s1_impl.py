"""
This module contains the S1 score, a measure of the skill of the gradient (spatial pattern)
of a forecast field
"""

from typing import Optional

import numpy as np

import scores.utils
from scores.processing import aggregate, broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike


def s1(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    x_dim: str,
    y_dim: str,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the WMO S1 score, a measure of the skill of the gradient (spatial pattern)
    of a forecast field relative to the corresponding observed or analysed field.

    The S1 score is computed on a two-dimensional verification grid (e.g. latitude/longitude,
    or any other pair of spatial dimensions), and is defined as

    .. math::
        S1 = 100 \\cdot \\frac{\\sum_{i=1}^n w_i (eg)_i}{\\sum_{i=1}^n w_i (GL)_i}

    where the gradients are approximated by differences computed on the verification grid:

    .. math::
        eg = \\left| \\frac{\\partial}{\\partial x}(x_f - x_v) \\right| +
             \\left| \\frac{\\partial}{\\partial y}(x_f - x_v) \\right|

    .. math::
        GL = \\max\\left(\\left|\\frac{\\partial x_f}{\\partial x}\\right|,
             \\left|\\frac{\\partial x_v}{\\partial x}\\right|\\right) +
             \\max\\left(\\left|\\frac{\\partial x_f}{\\partial y}\\right|,
             \\left|\\frac{\\partial x_v}{\\partial y}\\right|\\right)

    where:
        - :math:`x_f` = the forecast value of the parameter in question
        - :math:`x_v` = the corresponding verifying (observed/analysed) value
        - :math:`n` = the number of grid points or observations in the verification area
        - :math:`w_i` = the weight applied at grid point or observation location :math:`i`

    A lower S1 score indicates a better match between the spatial gradients of the forecast
    and the verifying field. A perfect forecast has an S1 score of 0.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed or analysed variables.
        x_dim: The name of one of the two spatial dimensions of the verification grid
            (e.g. ``"lon"``). Gradients are approximated using adjacent-point differences
            along this dimension.
        y_dim: The name of the other spatial dimension of the verification grid
            (e.g. ``"lat"``). Gradients are approximated using adjacent-point differences
            along this dimension.
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            S1 score. All other dimensions will be preserved. As the S1 score is computed
            over the verification grid, ``x_dim`` and ``y_dim`` must always be included
            among the dimensions being reduced.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the S1 score. All other dimensions will be reduced. As the S1 score is computed
            over the verification grid, ``x_dim`` and ``y_dim`` must not be preserved.
        weights: An array of weights to apply to each grid point. If provided,
            the weights must be broadcastable to the data dimensions and must not
            contain negative or NaN values.

            If None, all points are weighted equally. Since S1 is a ratio, any
            uniform weighting cancels out and gives the same result as
            :math:`w_i = 1/n`.

            For gridded fields on a latitude/longitude grid, area-weighting may be
            desirable because grid cells represent different physical areas at
            different latitudes. In this case, weights of
            :math:`w_i = \\cos(\\theta_i)` can be supplied.

    Returns:
        An xarray object with the S1 score.

    Raises:
        ValueError: If ``x_dim`` or ``y_dim`` is not a dimension of both ``fcst`` and ``obs``.
        ValueError: If ``x_dim`` and ``y_dim`` are the same dimension.
        ValueError: If ``x_dim`` or ``y_dim`` would not be reduced, given the values of
            ``reduce_dims`` and/or ``preserve_dims``.

    References:
        -   Teweles, S., & Wobus, H. B. (1954). Verification of prognostic charts.
            Bulletin of the American Meteorological Society, 35(10), 455-463.

    Examples:
        >>> import xarray as xr
        >>> from scores.spatial import s1

        >>> lats = [-35, -30, -25]
        >>> lons = [140, 150, 160]

        >>> obs = xr.DataArray(
        ...     [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        ...     coords={"lat": lats, "lon": lons},
        ...     dims=["lat", "lon"],
        ... )

        >>> fcst = xr.DataArray(
        ...     [[1.3, 1.8, 3.4], [1.9, 3.3, 3.8], [3.1, 4.4, 4.6]],
        ...     coords={"lat": lats, "lon": lons},
        ...     dims=["lat", "lon"],
        ... )

        >>> s1(fcst, obs, x_dim="lon", y_dim="lat")
        <xarray.DataArray ()> Size: 8B
        array(36.17021277)
    """
    if x_dim not in fcst.dims or x_dim not in obs.dims:
        raise ValueError(f"`x_dim` ('{x_dim}') must be a dimension of both `fcst` and `obs`")
    if y_dim not in fcst.dims or y_dim not in obs.dims:
        raise ValueError(f"`y_dim` ('{y_dim}') must be a dimension of both `fcst` and `obs`")
    if x_dim == y_dim:
        raise ValueError("`x_dim` and `y_dim` must be different dimensions")

    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    if x_dim not in reduce_dims or y_dim not in reduce_dims:
        raise ValueError(
            "The S1 score is computed over the verification grid, so both `x_dim` and `y_dim` "
            "must be reduced. Check the `reduce_dims` and `preserve_dims` arguments."
        )

    fcst, obs = broadcast_and_match_nan(fcst, obs)

    error = fcst - obs

    # The x- and y-direction gradients are aggregated (summed) separately before being
    # added together, rather than being added as full arrays first. The x-direction
    # diff has shape (..., n_y, n_x - 1) and the y-direction diff has shape
    # (..., n_y - 1, n_x): adding these arrays directly would trigger xarray's default
    # inner join on their mismatched coordinates, silently dropping the x-gradient row
    # and y-gradient column at one edge of the grid. Summing each term independently
    # avoids ever aligning two differently-shaped arrays, while preserving the
    # intended calculation because the x- and y-direction gradient contributions
    # are summed independently before being combined.
    error_gradient_x = abs(error.diff(dim=x_dim))
    error_gradient_y = abs(error.diff(dim=y_dim))

    fcst_gradient_x = abs(fcst.diff(dim=x_dim))
    obs_gradient_x = abs(obs.diff(dim=x_dim))
    fcst_gradient_y = abs(fcst.diff(dim=y_dim))
    obs_gradient_y = abs(obs.diff(dim=y_dim))

    numerator = aggregate(
        error_gradient_x,
        reduce_dims=reduce_dims,
        weights=weights,
        method="sum",
    ) + aggregate(
        error_gradient_y,
        reduce_dims=reduce_dims,
        weights=weights,
        method="sum",
    )

    denominator = aggregate(
        np.maximum(fcst_gradient_x, obs_gradient_x),
        reduce_dims=reduce_dims,
        weights=weights,
        method="sum",
    ) + aggregate(
        np.maximum(fcst_gradient_y, obs_gradient_y),
        reduce_dims=reduce_dims,
        weights=weights,
        method="sum",
    )

    return 100 * numerator / denominator
