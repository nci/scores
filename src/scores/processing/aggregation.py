"""
Functions related to aggregating data
"""

import warnings
from typing import Optional

import xarray as xr

from scores.processing.matching import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike


def agg(
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
    the input ``values``, optionally using weights to compute a weighted mean. Weighting
    is performed using xarray's `.weighted()` method. The `method` arg specifies if you
    want to produce a weighted mean or weighted sum.

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
            strings, or None. If None, no reduction is performed.ß
        weights: Weights to apply for weighted averaging.
            Must be broadcastable to `values` and contain no negative values. If None,
            an unweighted mean is calculated. Defaults to None.
        method: Aggregation method to use. Either "mean" or "sum". Defaults to "mean".

    Returns:
        An xarray object (same type as the input) with (un)weighted mean or sum of ``values``

    Raises:
        ValueError: If `weights` contains any negative values.
        ValueError: if `weights` contains any NaN values
        ValueError: if `method` is not 'mean' or 'sum'
        ValueError: if `weights` is an xr.Dataset when `values` is an xr.DataArray
        NotImplementedError: if `method` is sum and weights is an xr.Dataset

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
        if isinstance(weights, xr.DataArray):
            if (weights < 0).any().item():
                raise ValueError("Weights must not contain negative values.")
            if weights.isnull().any():
                raise ValueError(
                    """
                        Weights must not contain NaN values. If appropriate consider 
                        filling missing data with `weights.fillna(0)`
                        """
                )
        else:
            if xr.concat([(weights[var] < 0).any() for var in weights.data_vars], dim="vars").any():
                raise ValueError("Weights must not contain negative values.")
            if xr.concat([(weights[var].isnull()).any() for var in weights.data_vars], dim="vars").any():
                raise ValueError(
                    """
                    Weights must not contain NaN values. If appropriate consider 
                    filling missing data with `weights.fillna(0)`
                    """
                )

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
                if method == "sum":
                    raise NotImplementedError(
                        "using the method 'sum' with weights that are xr.Datasets is not currently supported"
                    )
                if isinstance(values, xr.DataArray):
                    raise ValueError("`weights` cannot be an xr.Dataset when `values` is an xr.DataArray")
                w_results = {}
                for name, da in values.data_vars.items():
                    if name not in weights:
                        raise KeyError(f"No weights provided for variable '{name}'")

                    w = weights[name]
                    da_aligned, w_aligned = broadcast_and_match_nan(da, w)
                    w_results[name] = (da_aligned * w_aligned).sum(dim=reduce_dims) / w_aligned.sum(dim=reduce_dims)

                return xr.Dataset(w_results)

            values = values.weighted(weights)

        match method:
            case "mean":
                values = values.mean(reduce_dims)
            case "sum":  # pragma: no cover (we explicitly check the case where the method doesn't match earlier)
                values = values.sum(reduce_dims)
    return values
