"""
This module contains methods related to the Brier score
"""

from typing import Optional

import xarray as xr

from scores.continuous import mse
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_binary


def brier_score(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: bool = True,
) -> XarrayLike:
    """
    Calculates the Brier score for forecast and observed data. For an explanation of the
    Brier score, see https://en.wikipedia.org/wiki/Brier_score.

    .. math::
        \\text{brier score} = \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2

    If you want to speed up performance with Dask and are confident of your input data,
    or if you want observations to take values between 0 and 1, set ``check_args`` to ``False``.

    Args:
        fcst: Forecast or predicted variables in xarray.
        obs: Observed variables in xarray.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Brier score. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Brier score. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        check_args: will perform some tests on the data if set to True
    Raises:
        ValueError: if ``fcst`` contains non-nan values outside of the range [0, 1]
        ValueError: if ``obs`` contains non-nan values not in the set {0, 1}
    """
    if check_args:
        error_msg = ValueError("`fcst` contains values outside of the range [0, 1]")
        if isinstance(fcst, xr.Dataset):
            # The .values is required as item() is not yet a valid method on Dask
            if fcst.to_array().max().values.item() > 1 or fcst.to_array().min().values.item() < 0:
                raise error_msg
        else:
            if fcst.max().values.item() > 1 or fcst.min().values.item() < 0:
                raise error_msg
        check_binary(obs, "obs")

    return mse(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights)
