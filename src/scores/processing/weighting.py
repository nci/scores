"""
Functions related to weighted aggregations
"""

import warnings
from typing import Optional

from scores.typing import FlexibleDimensionTypes, XarrayLike


def weighted_agg(
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

    Weights must not contain NaN values. Missing values can be filled by ``weights.fillna(0)``
    if you would like to assign a weight of zero to those points (e.g., masking).

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
        ValueError: if `weights` contains any NaN values
        ValueError: if `method` is not 'mean' or 'sum'

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
        if weights.isnull().any():
            raise ValueError(
                """
                Weights must not contain NaN values. If appropriate consider 
                filling missing data with `weights.fillna(0)`
                """
            )
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
            values = values.weighted(weights)

        match method:
            case "mean":
                values = values.mean(reduce_dims)
            case "sum":
                values = values.sum(reduce_dims)
    return values
