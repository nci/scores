"""
Implementation of the calculation of data for a Quantile-Quantile (Q-Q) plot.
"""

from typing import Dict, Iterable

import numpy as np
import xarray as xr

from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions


def quantile_quantile(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    quantiles: Iterable[float],
    interpolation_method: str = "linear",
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
) -> Dict[str, XarrayLike]:
    """
    Calculate the quantiles of empirical forecast and observation data for a
    quantile-quantile (Q-Q) plot.

    Forecast and observation data are broadcast and NaN values are matched
    before calculating the quantiles. This function calculates the quantiles
    for the user to plot using their preferred plotting library.

    Currently this function is implemented for working with raw forecast and
    observation data rather than a theoretical probability distribution.

    Args:
        fcst: Forecast data.
        obs: Observation data.
        quantiles: Quantiles to compute, e.g., [0.1, 0.5, 0.9].
        interpolation_method: Interpolation method for estimating the quantile value,
            default is 'linear'.
            Valid options include 'inverted_cdf', 'averaged_inverted_cdf',
            'closest_observation', 'interpolated_inverted_cdf', 'hazen',
            'weibull', 'linear', 'median_unbiased', 'normal_unbiased',
            'lower', 'higher', 'midpoint', and 'nearest'. See :py:func:`numpy.quantile`
            for more information on interpolation method options.
        reduce_dims: Dimensions to reduce over when calculating quantiles.
        preserve_dims: Dimensions to preserve when calculating quantiles.

    Returns:
        A dictionary with keys 'fcst_q' and 'obs_q', containing the quantiles
        of the forecast and observation data, respectively.

    Raises:
        ValueError: If the interpolation method is invalid.
        ValueError: if quantiles are not between 0 and 1.

    References:
        Déqué, M. (2012). Deterministic forecasts of continuous variables. In I. T.
        Jolliffe & D. B. Stephenson (Eds.), Forecast verification: A practitioner’s
        guide in atmospheric science (2nd ed., pp. 77–94). Wiley. https://doi.org/10.1002/9781119960003

    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.continuous import quantile_quantile
        >>> fcst = xr.DataArray(np.random.rand(100), dims='time')
        >>> obs = xr.DataArray(np.random.rand(100), dims='time')
        >>> result = quantile_quantile(fcst, obs, quantiles=[0.1, 0.5, 0.9])
    """
    if interpolation_method not in [
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]:
        raise ValueError(
            f"Invalid interpolation method: {interpolation_method}. "
            "Choose from 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', "
            "'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', "
            "'normal_unbiased', 'lower', 'higher', 'midpoint', or 'nearest'."
        )
    if not (0 <= np.array(quantiles)).all() or not (np.array(quantiles) <= 1).all():
        raise ValueError("Quantiles must be in the range [0, 1]")

    reduce_dims: FlexibleDimensionTypes = gather_dimensions(
        fcst_dims=fcst.dims,
        obs_dims=obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    fcst, obs = broadcast_and_match_nan(fcst, obs)

    fcst_quantiles = fcst.quantile(
        quantiles=quantiles,
        dim=reduce_dims,
        method=interpolation_method,
    )
    obs_quantiles = fcst.quantile(
        quantiles=quantiles,
        dim=reduce_dims,
        method=interpolation_method,
    )
    return {"fcst_q": fcst_quantiles, "obs_q": obs_quantiles}
