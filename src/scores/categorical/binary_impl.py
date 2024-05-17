"""
This module contains methods for binary categories
"""

from typing import Optional

import numpy as np
import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_binary, gather_dimensions


def probability_of_detection(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: Optional[bool] = True,
) -> XarrayLike:
    """
    Calculates the Probability of Detection (POD), also known as the Hit Rate.
    This is the proportion of observed events (obs = 1) that were correctly
    forecast as an event (fcst = 1).

    Args:
        fcst: An array containing binary values in the set {0, 1, np.nan}
        obs: An array containing binary values in the set {0, 1, np.nan}
        reduce_dims: Optionally specify which dimensions to sum when
            calculating the POD. All other dimensions will be not summed. As a
            special case, 'all' will allow all dimensions to be summed. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to sum across all dims.
        preserve_dims: Optionally specify which dimensions to not sum
            when calculating the POD. All other dimensions will be summed.
            As a special case, 'all' will allow all dimensions to be
            not summed. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the POD score at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely. Only one of `reduce_dims` and `preserve_dims` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        check_args: Checks if `fcst and `obs` data only contains values in the set
            {0, 1, np.nan}. You may want to skip this check if you are sure about your
            input data and want to improve the performance when working with dask.

    Returns:
        A DataArray of the Probability of Detection.

    Raises:
        ValueError: if there are values in `fcst` and `obs` that are not in the
            set {0, 1, np.nan} and `check_args` is true.

    """
    # fcst & obs must be 0s and 1s
    if check_args:
        check_binary(fcst, "fcst")
        check_binary(obs, "obs")
    dims_to_sum = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

    misses = (obs == 1) & (fcst == 0)
    hits = (obs == 1) & (fcst == 1)

    # preserve NaNs
    misses = misses.where((~np.isnan(fcst)) & (~np.isnan(obs)))
    hits = hits.where((~np.isnan(fcst)) & (~np.isnan(obs)))

    misses = apply_weights(misses, weights=weights)
    hits = apply_weights(hits, weights=weights)

    misses = misses.sum(dim=dims_to_sum)
    hits = hits.sum(dim=dims_to_sum)

    pod = hits / (hits + misses)
    return pod


def probability_of_false_detection(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: Optional[bool] = True,
) -> XarrayLike:
    """
    Calculates the Probability of False Detection (POFD), also known as False
    Alarm Rate (not to be confused with the False Alarm Ratio). The POFD is
    the proportion of observed non-events (obs = 0) that were incorrectly
    forecast as event (i.e. fcst = 1).

    Args:
        fcst: An array containing binary values in the set {0, 1, np.nan}
        obs: An array containing binary values in the set {0, 1, np.nan}
        reduce_dims: Optionally specify which dimensions to sum when
            calculating the POFD. All other dimensions will be not summed. As a
            special case, 'all' will allow all dimensions to be summed. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to sum across all dims.
        preserve_dims: Optionally specify which dimensions to not sum
            when calculating the POFD. All other dimensions will be summed.
            As a special case, 'all' will allow all dimensions to be
            not summed. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the POD score at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely. Only one of `reduce_dims` and `preserve_dims` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        check_args: Checks if `fcst and `obs` data only contains values in the set
            {0, 1, np.nan}. You may want to skip this check if you are sure about your
            input data and want to improve the performance when working with dask.

    Returns:
        A DataArray of the Probability of False Detection.

    Raises:
        ValueError: if there are values in `fcst` and `obs` that are not in the
            set {0, 1, np.nan} and `check_args` is true.
    """
    # fcst & obs must be 0s and 1s
    if check_args:
        check_binary(fcst, "fcst")
        check_binary(obs, "obs")
    dims_to_sum = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)

    false_alarms = (obs == 0) & (fcst == 1)
    correct_negatives = (obs == 0) & (fcst == 0)

    # preserve NaNs
    false_alarms = false_alarms.where((~np.isnan(fcst)) & (~np.isnan(obs)))
    correct_negatives = correct_negatives.where((~np.isnan(fcst)) & (~np.isnan(obs)))

    false_alarms = apply_weights(false_alarms, weights=weights)
    correct_negatives = apply_weights(correct_negatives, weights=weights)

    false_alarms = false_alarms.sum(dim=dims_to_sum)
    correct_negatives = correct_negatives.sum(dim=dims_to_sum)

    pofd = false_alarms / (false_alarms + correct_negatives)
    return pofd
