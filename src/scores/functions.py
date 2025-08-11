"""
Contains functions which transform data or perform calculations
"""

import warnings
from typing import Optional, overload

import numpy as np
import xarray as xr

from scores.typing import FlexibleDimensionTypes, XarrayLike


def apply_weights(values, *, weights: Optional[XarrayLike] = None):
    """
    Returns:
        A new array with the elements of values multiplied by the specified weights.

    Args:
        - weights: The weightings to be used at every location in the values array. If weights contains additional
        dimensions, these will be taken to mean that multiple weightings are wanted simultaneoulsy, and these
        dimensions will be added to the new array.
        - values: The unweighted values to be used as the basis for weighting calculation


    Note - this weighting function is different to the .weighted method contained in xarray in that xarray's
    method does not allow NaNs to be present in the weights or data.
    """

    if weights is not None:
        result = values * weights
        return result

    return values


def apply_weighted_agg(
    values: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
    method: str = "mean",
) -> XarrayLike:
    """
    Computes a weighted or unweighted aggregation of the input data across specified dimensions.
    The input data is typically the "score" at each point.

    This function applies a mean reduction over the dimensions given by ``reduce_dims`` on
    the input ``values``, optionally using weights to compute a weighted mean. Weighting
    is performed using xarray's `.weighted()` method.

    If `reduce_dims` is None, no reduction is performed and the original `values` are
    returned unchanged.

    If `weights` is None, an unweighted mean is computed. If weights are provided, negative
    weights are not allowed and will raise a `ValueError`.

    If weights are provided but `reduce_dims` is None (i.e., no reduction), a `UserWarning`
    is emitted since the weights will be ignored.

    Args:
        values: Input data to be reduced. Typically an `xr.DataArray` or `xr.Dataset`.
        reduce_dims: Dimensions over which to apply the mean. Can be a string, list of
            strings, or None. If None, no reduction is performed. Defaults to None.
        weights: Weights to apply for weighted averaging.
            Must be broadcastable to `values` and contain no negative values. If None,
            an unweighted mean is calculated. Defaults to None.
        method: Aggregation method to use. Either "mean" or "sum". Defaults to "mean".

    Returns:
        An xarray object (same type as the input) with (un)weighted mean of ``values``

    Raises:
        ValueError: If `weights` contains any negative values.

    Warnings:
        UserWarning: If weights are provided but no reduction is performed (`reduce_dims` is None),
        a warning is issued since weights are ignored.

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> da = xr.DataArray(np.arange(6).reshape(2, 3), dims=['x', 'y'])
        >>> weights = xr.DataArray([1, 2], dims=['x'])
        >>> apply_weighted_mean(da, reduce_dims=['x'], weights=weights)
        <xarray.DataArray (y: 3)>
        array([2., 3., 4.])
        Dimensions without coordinates: y

    """
    if method not in ["mean", "sum"]:
        raise ValueError(f"Method must be either 'mean' or 'sum', got '{method}'")

    if weights is not None:
        if (weights < 0).any():
            raise ValueError("Weights must not contain negative values.")
    if reduce_dims is None and weights is not None:
        warnings.warn(
            """
            Weights were provided but all the score across all dimensions is being preserved. 
            Weights will be ignored.
            """,
            UserWarning,
        )
    if reduce_dims is not None:
        if weights is not None:
            weights = weights.fillna(0)
            values = values.weighted(weights)

        if method == "mean":
            values = values.mean(reduce_dims)
        else:
            values = values.sum(reduce_dims)
    return values


def create_latitude_weights(latitudes):
    """
    A common way of weighting errors is to make them proportional to the amount of area
    which is contained in a particular region. This is approximated by the cosine
    of the latitude on an LLXY grid. Nuances not accounted for include the variation in
    latitude across the region, or the irregularity of the surface of the earth.

    Returns:
        An xarray containing the weight values to be used for area approximation

    Args:
        An xarray (or castable type) containing latitudes between +90 and -90 degrees

    Note - floating point behaviour can vary between systems, precisions and other factors
    """
    weights = np.cos(np.deg2rad(latitudes))
    return weights


# Dataset input types lead to a Dataset return type
@overload
def angular_difference(source_a: xr.Dataset, source_b: xr.Dataset) -> xr.Dataset: ...


# DataArray input types lead to a DataArray return type
@overload
def angular_difference(source_a: xr.DataArray, source_b: xr.DataArray) -> xr.DataArray:  # type: ignore
    ...


def angular_difference(source_a: XarrayLike, source_b: XarrayLike) -> XarrayLike:
    """
    Determines, in degrees, the smaller of the two explementary angles between
    two sources of directional data (e.g. wind direction).

    Args:
        source_a: direction data in degrees, first source
        source_b: direction data in degrees, second source

    Returns:
        An array containing angles within the range [0, 180].
    """
    difference = np.abs(source_a - source_b) % 360
    difference = difference.where(difference <= 180, 360 - difference)  # type: ignore
    return difference  # type: ignore
