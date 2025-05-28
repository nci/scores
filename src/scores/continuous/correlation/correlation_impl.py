"""
This module contains functions to calculate the correlation.
"""

from typing import Optional

import xarray as xr

import scores.utils
from scores.typing import FlexibleDimensionTypes, XarrayLike


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

    Raises:
        ValueError: If `preserve_dims` is set to 'all', a ValueError will be raised.

    Note:
        This function isn't set up to take weights.

    Reference:
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    See Also:
        :py:func:`scores.continuous.correlation.spearmanr`

    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.continuous.correlation.correlation_impl import pearsonr
        >>> # Create example forecast and observation DataArrays
        >>> fcst = xr.DataArray(np.random.rand(10, 5), dims=("time", "location"))
        >>> obs = xr.DataArray(np.random.rand(10, 5), dims=("time", "location"))
        >>> # Calculate Pearson's correlation coefficient
        >>> result = pearsonr(fcst, obs, reduce_dims="time")
        >>> print(result)
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    if preserve_dims == "all":
        raise ValueError(
            "The 'preserve_dims' argument cannot be set to 'all' for the Pearson's correlation coefficient."
        )
    return xr.corr(fcst, obs, reduce_dims)


def spearmanr(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> XarrayLike:
    """
    Calculates the Spearman's rank correlation coefficient between two xarray objects.

    .. math::
        r_s = \\rho\\big(R(x), R(y)\\big)

    where:
        - :math:`\\rho` is the Pearson correlation coefficient.
        - :math:`R` is the ranking operator.

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Spearman's rank correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Spearman's rank correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved.

    Returns:
        An xarray object with Spearman's rank correlation coefficient values.

    Raises:
        ValueError: If `preserve_dims` is set to 'all', a ValueError will be raised.
        TypeError: If the input types are not xarray DataArrays or Datasets.
        ValueError: If the input Datasets do not have the same data variables.

    Note:
        This function isn't set up to take weights.

    Reference:
        Spearman, C. (1904). The Proof and Measurement of Association between Two Things. The American Journal of Psychology, 15(1), 72–101.
        https://doi.org/10.2307/1412159

    See also:
        :py:func:`scores.continuous.correlation.pearsonr`

    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.continuous.correlation.correlation_impl import spearmanr
        >>> # Create example forecast and observation DataArrays
        >>> fcst = xr.DataArray(np.random.rand(10, 5), dims=("time", "location"))
        >>> obs = xr.DataArray(np.random.rand(10, 5), dims=("time", "location"))
        >>> # Calculate Spearman's rank correlation coefficient
        >>> result = spearmanr(fcst, obs, reduce_dims="time")
        >>> print(result)
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    if preserve_dims == "all":
        raise ValueError(
            "The 'preserve_dims' argument cannot be set to 'all' for the Spearman's correlation coefficient."
        )

    def _spearman_calc(fcst, obs, reduce_dims):
        """
        Core calculation for Spearman's rank correlation coefficient.
        """
        # Flatten/stack the arrays along the dimensions to be reduced
        tmp_dim = "".join(list(fcst.dims) + list(obs.dims))
        obs_stacked = obs.stack({tmp_dim: reduce_dims})
        fcst_stacked = fcst.stack({tmp_dim: reduce_dims})
        # Rank
        fcst_ranks = fcst_stacked.rank(dim=tmp_dim)
        obs_ranks = obs_stacked.rank(dim=tmp_dim)

        return pearsonr(fcst_ranks, obs_ranks, reduce_dims=tmp_dim)

    if isinstance(fcst, xr.DataArray) and isinstance(obs, xr.DataArray):
        return _spearman_calc(fcst, obs, reduce_dims)

    if isinstance(fcst, xr.Dataset) and isinstance(obs, xr.Dataset):
        # Ensure both datasets have the same variables
        if set(fcst.data_vars) != set(obs.data_vars):
            raise ValueError("Both datasets must contain the same variables.")

        # Apply spearmanr to each variable
        results = {var: _spearman_calc(fcst[var], obs[var], reduce_dims) for var in fcst.data_vars}

        return xr.Dataset(results)

    raise TypeError("Both fcst and obs must be either xarray DataArrays or xarray Datasets.")
