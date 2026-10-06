"""
Functions related to aggregating data
"""

import warnings
from typing import Optional

import array_api_compat
import xarray as xr

from scores.processing.matching import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_weights


def aggregate(
    values: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None,
    weights: Optional[XarrayLike] = None,
    method: str = "mean",
) -> XarrayLike:
    """
    Computes a weighted or unweighted aggregation of the input data across specified dimensions.
    The input data is typically the "score" at each point.

    This function applies a mean reduction or a sum over the dimensions given by ``reduce_dims`` on
    the input ``values``, optionally using weights to compute a weighted mean or sum.
    The ``method`` arg specifies if you want to produce a weighted mean or weighted sum.

    If `reduce_dims` is None, no aggregation is performed and the original ``values`` are
    returned unchanged.

    If ``weights`` is None, an unweighted mean or sum is computed. If weights are provided, negative
    weights are not allowed and will raise a ``ValueError``.

    If weights are provided but ``reduce_dims`` is None (i.e., no reduction), a ``UserWarning``
    is emitted since the weights will be ignored.

    Weights must not contain NaN values. Missing values can be filled by ``weights.fillna(0)``
    if you would like to assign a weight of zero to those points (e.g., masking).

    Args:
        values: Input data to be aggregated.
        reduce_dims: Dimensions over which to apply the mean. Can be a string, list of
            strings, or None. If None, no reduction is performed.
        weights: Weights to apply for weighted averaging.
            Must be broadcastable to ``values`` and contain no negative values. If None,
            an unweighted mean is calculated. Defaults to None.
        method: Aggregation method to use. Either "mean" or "sum". Defaults to "mean".

    Returns:
        An xarray object (same type as the input) with (un)weighted mean or sum of ``values``

    Raises:
        ValueError: If ``weights`` contains any negative values.
        ValueError: if ``weights`` contains any NaN values
        ValueError: if ``method`` is not 'mean' or 'sum'
        ValueError: if ``weights`` is an xr.Dataset when ``values`` is an xr.DataArray

    Warnings:
        UserWarning: If weights are provided but no reduction is performed (``reduce_dims`` is None),
        a warning is issued since weights are ignored.

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.processing import aggregate

        >>> # Create sample data
        >>> da = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [1, 2], "y": ["A", "B", "C"]},
        ... )
        >>> da
        <xarray.DataArray (x: 2, y: 3)> Size: 48B
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) int64 16B 1 2
          * y        (y) <U1 12B 'A' 'B' 'C'

        >>> # Unweighted mean over x dimension
        >>> aggregate(da, reduce_dims="x")
        <xarray.DataArray (y: 3)> Size: 24B
        array([1.5, 2.5, 3.5])
        Coordinates:
          * y        (y) <U1 12B 'A' 'B' 'C'

        >>> # Weighted mean over x dimension
        >>> weights = xr.DataArray([1, 2], dims=["x"])
        >>> aggregate(da, reduce_dims="x", weights=weights)
        <xarray.DataArray (y: 3)> Size: 24B
        array([2., 3., 4.])
        Coordinates:
          * y        (y) <U1 12B 'A' 'B' 'C'

        >>> # Weighted sum over y dimension
        >>> weights_y = xr.DataArray([0.5, 1.0, 1.5], dims=["y"])
        >>> aggregate(da, reduce_dims="y", weights=weights_y, method="sum")
        <xarray.DataArray (x: 2)> Size: 16B
        array([ 4., 13.])
        Coordinates:
          * x        (x) int64 16B 1 2
    """

    if isinstance(values, XarrayLike):
        _check_aggregate_inputs(values, reduce_dims, weights, method)

    if reduce_dims is None:
        return values

    match method:
        case "mean":
            if weights is not None:
                return _weighted_mean(values, weights, reduce_dims)
            return values.mean(reduce_dims)
        case "sum":
            if weights is not None:
                return _weighted_sum(values, weights, reduce_dims)
            return values.sum(reduce_dims)
        case _:  # pragma: no cover - invalid method is checked in `_check_aggregate_inputs`
            raise ValueError(f"Unsupported method {method}. Expected 'mean' or 'sum'.")


def _weighted_mean(values, weights, reduce_dims=None):
    if isinstance(weights, xr.Dataset):
        w_results = {}
        for name, da in values.data_vars.items():
            w = weights[name]
            da_aligned, w_aligned = broadcast_and_match_nan(da, w)

            # `check_weights` in `_check_aggregate_inputs` ensures that `weights`
            # has at least one positive value and will raise an error.
            # However, if a value in w_aligned.sum(dim=reduce_dims) is zero,
            # a NaN will be produced for that point.
            w_results[name] = (da_aligned * w_aligned).sum(dim=reduce_dims) / w_aligned.sum(dim=reduce_dims)

        return xr.Dataset(w_results)

    elif isinstance(values, xr.DataArray) or isinstance(values, xr.Dataset):
        values = values.weighted(weights)
        return values.mean(reduce_dims)

    # Else attempt to handle with array compatibility
    else:
        xp = array_api_compat.array_namespace(values, weights)
        weighted_error = xp.mul(values, weights)
        weighted_sum_of_error = xp.nansum(weighted_error)
        sum_of_weights = xp.nansum(weights)
        weighted_mean = weighted_sum_of_error / sum_of_weights
        return weighted_mean


def _weighted_sum(
    values: XarrayLike,
    weights: XarrayLike,
    reduce_dims: FlexibleDimensionTypes,
) -> XarrayLike:
    """
    Calculated the weighted sum of `values` using `weights` over specified dimensions.
    """
    if isinstance(weights, xr.Dataset):
        w_results = {}
        for name, da in values.data_vars.items():
            w = weights[name]
            da_aligned, w_aligned = broadcast_and_match_nan(da, w)
            summed = (da_aligned * w_aligned).sum(dim=reduce_dims)
            # If weights sum to zero for a point that has been aggregated over reduce_dims,
            # we want the result to be NaN, not zero.
            summed = summed.where(w_aligned.sum(dim=reduce_dims) != 0)
            w_results[name] = summed

        return xr.Dataset(w_results)
    values = values.weighted(weights)
    summed_values = values.sum(reduce_dims)
    # Handle NaNs in `values`
    summed_values = summed_values.where(~xr.ufuncs.isnan(values.mean(reduce_dims)))
    return summed_values


def _check_aggregate_inputs(
    values: XarrayLike, reduce_dims: FlexibleDimensionTypes | None, weights: XarrayLike | None, method: str
):
    """
    This function checks the inputs to the aggregate function.

    It checks that:
    - `method` is either 'mean' or 'sum'
    - `weights` does not contain negative values
    - `weights` does not contain NaN values
    - `weights` were provided, `reduce_dims` is not None
    - `weights` is not an xr.Dataset when `values` is an xr.DataArray
    - if `values is an xr.Dataset`, and `weights` is an xr.Dataset, it must have the same variables

    Args:
        values: The input data to be reduced in :py:func:`aggregate`.
        reduce_dims: The dimensions over which to apply the mean in :py:func:`aggregate`.
        weights: The weights to apply for weighted averaging in :py:func:`aggregate`.
        method: The aggregation method to use, either "mean" or "sum" in :py:func:`aggregate`.

    """
    if method not in ["mean", "sum"]:
        raise ValueError(f"Method must be either 'mean' or 'sum', got '{method}'")

    if weights is not None:
        check_weights(weights)

    if reduce_dims is None and weights is not None:
        warnings.warn(
            """
            Weights were provided but the point-wise score across all dimensions is being preserved.
            Weights will be ignored.
            """,
            UserWarning,
        )
    if reduce_dims is not None:
        if weights is not None:
            # xarray doesn't allow .weighted to take xr.Dataset as weights, so we need to do it ourselves
            if isinstance(weights, xr.Dataset):
                if isinstance(values, xr.DataArray):
                    raise ValueError("`weights` cannot be an xr.Dataset when `values` is an xr.DataArray")
                for name in values.data_vars:
                    if name not in weights:
                        raise KeyError(f"No weights provided for variable '{name}'")
