"""
This module contains methods which make assertions at runtime about the state of various data
structures and values
"""

import numpy as np
import xarray as xr


def coords_increasing(da: xr.DataArray, dim: str):
    """Checks if coordinates in a given DataArray are increasing.

    Note: No in-built raise if `dim` is not a dimension of `da`.

    Args:
        da (xr.DataArray): Input data
        dim (str): Dimension to check if increasing
    Returns:
        (bool):  Returns True if coordinates along `dim` dimension of
        `da` are increasing, False otherwise.
    """
    result = (da[dim].diff(dim) > 0).all()
    return result


def cdf_values_within_bounds(cdf: xr.DataArray) -> bool:
    """Checks that 0 <= cdf <= 1. Ignores NaNs.

    Args:
        cdf (xr.DataArray): array of CDF values

    Returns:
        (bool): `True` if `cdf` values are all between 0 and 1 whenever values are not NaN,
            or if all values are NaN; and `False` otherwise.
    """
    flag = cdf.count() == 0 or ((cdf.min() >= 0) & (cdf.max() <= 1))
    return flag  # type: ignore  # mypy thinks flag could be a DataArray


def check_nan_decreasing_inputs(cdf, threshold_dim, tolerance):
    """Checks inputs to `nan_decreasing_cdfs` and `_decreasing_cdfs`."""

    if threshold_dim not in cdf.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `cdf`")

    if tolerance < 0:
        raise ValueError("`tolerance` must be nonnegative.")

    if not coords_increasing(cdf, threshold_dim):
        raise ValueError(f"Coordinates along '{threshold_dim}' dimension should be increasing.")

    all_nan_or_no_nan = np.isnan(cdf).all(threshold_dim) | (~np.isnan(cdf)).all(threshold_dim)
    if not all_nan_or_no_nan.all():
        raise ValueError("CDFs should have no NaNs or be all NaN along `threshold_dim`")
