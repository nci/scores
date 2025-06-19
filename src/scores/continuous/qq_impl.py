"""
Implementation of the calculation of data for a Quantile-Quantile (Q-Q) plot.
"""

from typing import Dict, Iterable

import numpy as np
import xarray as xr

from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike, all_same_xarraylike
from scores.utils import gather_dimensions

NP_INTERP_METHODS = [
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
]


def generate_qq_plot_data(
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
        ValueError: If dimensions named 'data_source' are present in the input data.
        ValueError: If fcst and obs are Datasets with different data variables.
        ValueError: If a user tries to preserve all dimensions.
        TypeError: If fcst and obs are not both xarray DataArrays or Datasets.

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
        >>> result = generate_qq_plot_data(fcst, obs, quantiles=[0.1, 0.5, 0.9])
    """
    if interpolation_method not in NP_INTERP_METHODS:
        raise ValueError(
            f"Invalid interpolation method: {interpolation_method}. "
            "Choose from 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', "
            "'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', "
            "'normal_unbiased', 'lower', 'higher', 'midpoint', or 'nearest'."
        )
    if not (0 <= np.array(quantiles)).all() or not (np.array(quantiles) <= 1).all():
        raise ValueError("Quantiles must be in the range [0, 1]")

    # Check that there isn't a dimension called 'data_source' in either fcst or obs
    if "data_source" in fcst.dims or "data_source" in obs.dims:
        raise ValueError("Dimensions named 'data_source' are not allowed in the input data.")
    # Check if fcst and obs are both the same type (xarray.DataArray or xarray.Dataset)
    if not all_same_xarraylike([fcst, obs]):
        raise TypeError("Both fcst and obs must be either xarray DataArrays or xarray Datasets.")
    # Check if datasets, that they have the same data vars
    if isinstance(fcst, xr.Dataset):
        if set(fcst.data_vars) != set(obs.data_vars) and isinstance(fcst, xr.Dataset):
            raise ValueError("Both xr.Datasets must contain the same variables.")

    reduce_dims: FlexibleDimensionTypes = gather_dimensions(
        fcst_dims=fcst.dims,
        obs_dims=obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    if len(reduce_dims) == 0:
        raise ValueError("You cannot preserve all dimensions with generate_qq_plot_data.")

    fcst, obs = broadcast_and_match_nan(fcst, obs)

    fcst_quantiles = fcst.quantile(
        q=quantiles,
        dim=reduce_dims,
        method=interpolation_method,
    )
    obs_quantiles = obs.quantile(
        q=quantiles,
        dim=reduce_dims,
        method=interpolation_method,
    )

    result = xr.concat([fcst_quantiles, obs_quantiles], dim=xr.DataArray(["fcst", "obs"], dims=["data_source"]))
    return result
