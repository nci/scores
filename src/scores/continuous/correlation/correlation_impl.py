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


def spearmanr(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> xr.DataArray:
    """
    Calculates the Spearman's rank correlation coefficient between two xarray DataArrays \
    Spearman's correlation is identical to Pearson's correlation when the relationship \
    is linear. They diverge when the relationship is not linear as Spearman's correlation \
    assesses monotic relationships whereas Pearson's correlation assesses strictly linear \
    functions.


    .. math::
        \\rho = 1 - \\frac{6\\sum_{i=1}^{n}{d_i^2}}{n(n^2-1)}

    where:
        - :math:`\\rho` = Spearman's rank correlation coefficient
        - :math:`d_i` = the difference between the ranks of x and y in a sample
        - :math:`n` = the number of samples

    Args:
        fcst: Forecast or predicted variables
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Spearman's rank correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Spearman's rank correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
    Returns:
        xr.DataArray: An xarray object with Spearman's rank correlation coefficient values

    Note:
        This function isn't set up to take weights.

    See also:
    :py:func:`scores.continuous.correlation.pearsonr`

    Reference:
        Spearman, C. (1904). The Proof and Measurement of Association between Two Things. The American Journal of Psychology, 15(1), 72–101.
        https://doi.org/10.2307/1412159
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )

    # If reduce_dims contains multiple dimensions, handle ranking per dimension
    fcst_ranks = fcst
    obs_ranks = obs
    for dim in reduce_dims:
        fcst_ranks = fcst_ranks.rank(dim=dim)
        obs_ranks = obs_ranks.rank(dim=dim)

    return xr.corr(fcst_ranks, obs_ranks, reduce_dims)
