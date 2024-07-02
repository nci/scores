"""
This module contains functions to calculate the correlation.
"""

from typing import Optional

import xarray as xr

import scores.utils
from scores.typing import FlexibleDimensionTypes


def pearsonr(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> xr.DataArray:
    """
    Calculates the Pearson's correlation coefficient between two xarray DataArrays

    .. math::
        \\rho = \\frac{\\sum_{i=1}^{n}{(x_i - \\bar{x})(y_i - \\bar{y})}}
        {\\sqrt{\\sum_{i=1}^{n}{(x_i-\\bar{x})^2}\\sum_{i=1}^{n}{(y_i - \\bar{y})^2}}}

    where:
        - :math:`\\rho` = Pearson's correlation coefficient
        - :math:`x_i` = the values of x in a sample (i.e. forecast values)
        - :math:`\\bar{x}` = the mean value of the forecast sample
        - :math:`y_i` = the values of y in a sample (i.e. observed values)
        - :math:`\\bar{y}` = the mean value of the observed sample value

    Args:
        fcst: Forecast or predicted variables
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Pearson's correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Pearson's correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
    Returns:
        xr.DataArray: An xarray object with Pearson's correlation coefficient values

    Note:
        This function isn't set up to take weights.

    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )

    return xr.corr(fcst, obs, reduce_dims)
