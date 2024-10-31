"""
This module contains methods related to the Brier score
"""

from typing import Optional
from collections.abc import Sequence

import xarray as xr

from scores.continuous import mse
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_binary, gather_dimensions
from scores.functions import apply_weights


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
    Calculates the Brier score for forecast and observed data.

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

    References:
        - Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
          Monthly Weather Review, 78(1), 1-3. https://doi.org/fp62r6
        - https://en.wikipedia.org/wiki/Brier_score
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


def ensemble_brier_score(
    fcst: XarrayLike,
    obs: XarrayLike,
    ensemble_member_dim: str,
    thresholds: Sequence[float],
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    fair_correction: bool = True,
) -> XarrayLike:
    """
    Calculates the Brier score for an ensemble forecast for the provided thresholds. By default,
    the fair Brier score is calculated, which is a modified version of the Brier score that
    applies a correction based on the number of ensemble members.

    Args:
        fcst: Forecast or predicted variables in xarray.
        obs: Observed variables in xarray.
        ensemble_member_dim: The dimension name of the ensemble member.
        thresholds: The thresholds to use for the Fair Brier score that define what is
            considered an event.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Brier score. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Fair Brier score. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom).
        fair_correction: Whether or not to apply the fair Brier score correction. Default is True.

    Returns:
        The Brier score for the ensemble forecast.

    """
    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims

    dims_for_mean = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=ensemble_member_dim,
    )
    thresholds_xr = xr.DataArray(thresholds, dims=["threshold"], coords={"threshold": thresholds})
    member_event_count = (fcst >= thresholds_xr).sum(dim=ensemble_member_dim)
    total_member_count = len(fcst[ensemble_member_dim])

    fair_correction = (member_event_count * (total_member_count - member_event_count)) / (
        total_member_count**2 * (total_member_count - 1)
    )
    result = (member_event_count / total_member_count - obs) ** 2
    if fair_correction:
        result -= fair_correction

    # apply weights and take means across specified dims
    result = apply_weights(result, weights=weights).mean(dim=dims_for_mean)  # type: ignore

    return result  # type: ignore
