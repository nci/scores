"""
This module contains a variety of functions which modify data in various ways to process data structures
to support probabilistic verification.
"""

from collections.abc import Iterable
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from scores.probability.checks import (
    cdf_values_within_bounds,
    check_nan_decreasing_inputs,
)
from scores.typing import XarrayLike


def round_values(array: xr.DataArray, rounding_precision: float, *, final_round_decpl: int = 7) -> xr.DataArray:
    """Round data array to specified precision.

    Rounding is done differently to `xarray.DataArray.round` or `numpy.round` where
    the number of decimal places is specified in those cases. Instead, here the rounding
    precision is specified as a float. The value is rounded to the nearest value that is
    divisible by `rounding_precision`.

    For example, 3.73 rounded to precision 0.2 is 3.8, and 37.3 rounded to precision 20
    is 40.

    Assumes that rounding_precision >=0, with 0 indicating no rounding to be performed.
    If rounding_precision > 0, a final round to `final_round_decpl` decimal places is performed
    to remove artefacts of python rounding process.

    Args:
        array (xr.DataArray): array of data to be rounded
        rounding_precision (float): rounding precision
        final_round_decpl (int): final round to specified number of decimal
            places when `rounding_precision` > 0.

    Returns:
        xr.DataArray: DataArray with rounded values.

    Raises:
        ValueError: If `rounding_precision` < 0.

    Examples:
        >>> from scores.processing.cdf import round_values
        >>> import xarray as xr

        >>> data = [3.73, 4.10000001, 37.3]
        >>> coords = {"sample": [1, 2, 3]}

        >>> da = xr.DataArray(data, coords=coords, dims="sample")

        >>> round_values(da, rounding_precision=0.2)
        <xarray.DataArray (sample: 3)> Size: 24B
        array([ 3.8,  4.2, 37.2])
        Coordinates:
          * sample   (sample) int64 24B 1 2 3

        >>> round_values(da, rounding_precision=20)
        <xarray.DataArray (sample: 3)> Size: 24B
        array([ 0.,  0., 40.])
        Coordinates:
          * sample   (sample) int64 24B 1 2 3
    """
    if rounding_precision < 0:
        raise ValueError(f"rounding_precision '{rounding_precision}' is negative")

    if rounding_precision > 0:
        array = (array / rounding_precision).round() * rounding_precision
        array = array.round(decimals=final_round_decpl)

    return array


def propagate_nan(cdf: XarrayLike, threshold_dim: str) -> XarrayLike:
    """Propagates the NaN values from a "cdf" variable along the `threshold_dim`.

    Args:
        cdf (xr.DataArray): CDF values, so that P(X <= threshold) = cdf_value for
            each threshold in the `threshold_dim` dimension.
        threshold_dim (str): name of the threshold dimension in `cdf`.

    Returns:
        xr.DataArray: `cdf` variable with NaNs propagated.

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `cdf`.

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.processing.cdf import propagate_nan

        >>> data = [[0.1, 0.4, np.nan, 0.9], [0.2, 0.5, 0.7, 1.0]]
        >>> coords = {"station": [1, 2], "threshold": [10, 20, 30, 40]}

        >>> cdf = xr.DataArray(data, coords=coords, dims=["station", "threshold"])

        >>> propagate_nan(cdf, "threshold")
        <xarray.DataArray (station: 2, threshold: 4)> Size: 64B
        array([[nan, nan, nan, nan],
               [0.2, 0.5, 0.7, 1. ]])
        Coordinates:
          * station    (station) int64 16B 1 2
          * threshold  (threshold) int64 32B 10 20 30 40
    """
    if threshold_dim not in cdf.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `cdf`")

    where_nan = xr.DataArray(np.isnan(cdf)).any(dim=threshold_dim)
    result = cdf.where(~where_nan, np.nan)
    return result


def observed_cdf(
    obs: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    threshold_values: Optional[Iterable[float]] = None,
    include_obs_in_thresholds: bool = True,
    precision: float = 0,
) -> xr.DataArray:
    """Returns a data array of observations converted into CDF format.

    Such that:
        returned_value = 0 if threshold < observation
        returned_value = 1 if threshold >= observation

    Args:
        obs (xr.DataArray): observations
        threshold_dim (str): name of dimension in returned array that contains the threshold values.
        threshold_values (Optional[Iterable[float]]): values to include among thresholds.
        include_obs_in_thresholds (bool): if `True`, include (rounded) observed values among thresholds.
        precision (float): precision applied to observed values prior to constructing the CDF and
            thresholds. Select 0 for highest precision (i.e. no rounding).

    Returns:
        xr.DataArray: Observed CDFs and thresholds in the `threshold_dim` dimension.

    Raises:
        ValueError: if `precision < 0`.
        ValueError: if all observations are NaN and no non-NaN `threshold_values`
            are not supplied.

    Examples:
        >>> import xarray as xr
        >>> from scores.processing.cdf import observed_cdf

        >>> coords = {"station": ["A", "B", "C"]}
        >>> obs = xr.DataArray([2.09, 5.73, 8.11], coords=coords, dims="station")

        >>> # Thresholds are observations
        >>> observed_cdf(obs, "threshold")
        <xarray.DataArray (station: 3, threshold: 3)> Size: 72B
        array([[1., 1., 1.],
               [0., 1., 1.],
               [0., 0., 1.]])
        Coordinates:
          ...

        >>> # Thresholds are specified but also include observations
        >>> observed_cdf(obs, "threshold", threshold_values=[0, 5, 10])
        <xarray.DataArray (station: 3, threshold: 6)> Size: 144B
        array([[0., 1., 1., 1., 1., 1.],
               [0., 0., 0., 1., 1., 1.],
               [0., 0., 0., 0., 1., 1.]])
        Coordinates:
          ...

        >>> # Thresholds are specified but also include
        >>> # observations after rounding to precision
        >>> observed_cdf(obs, "threshold", threshold_values=[0, 5, 10], precision=1)
        <xarray.DataArray (station: 3, threshold: 6)> Size: 144B
        array([[0., 1., 1., 1., 1., 1.],
               [0., 0., 0., 1., 1., 1.],
               [0., 0., 0., 0., 1., 1.]])
        Coordinates:
          ...

        >>> # Only evaluate at specified thresholds
        >>> observed_cdf(obs, "threshold", threshold_values=[0, 5, 10],
        ...              include_obs_in_thresholds = False)
        <xarray.DataArray (station: 3, threshold: 3)> Size: 72B
        array([[0., 1., 1.],
               [0., 0., 1.],
               [0., 0., 1.]])
        Coordinates:
          ...

    """
    if precision < 0:
        raise ValueError("`precision` must be nonnegative.")

    threshold_values_as_array = np.array(threshold_values)

    if np.isnan(obs).all() and (threshold_values is None or np.isnan(threshold_values_as_array).all()):
        raise ValueError("must include non-NaN observations in thresholds or supply threshold values")

    if precision > 0:
        obs = round_values(obs, precision)

    thresholds = threshold_values_as_array if threshold_values is not None else []

    if include_obs_in_thresholds:
        thresholds = np.concatenate((obs.values.flatten(), thresholds))

    # remove any NaN
    thresholds = [x for x in thresholds if not np.isnan(x)]

    # pandas.unique retains the original ordering whereas set() may not
    # pandas.unique no longer accepts a simple array as input
    thresholds = pd.Series(thresholds)
    thresholds = np.sort(pd.unique(thresholds))

    da_thresholds = xr.DataArray(
        thresholds,
        dims=[threshold_dim],
        coords={threshold_dim: thresholds},
    )

    dab_obs, dab_thresholds = xr.broadcast(obs, da_thresholds)

    cdf = dab_thresholds >= dab_obs

    # convert to 0 and 1
    cdf = cdf.astype(float)

    cdf = cdf.where(~np.isnan(dab_obs))

    return cdf


def integrate_square_piecewise_linear(function_values: xr.DataArray, threshold_dim: str) -> xr.DataArray:
    """Calculates integral values and collapses `threshold_dim`.

    Calculates :math:`\\int{(F(t)^2)}`, where:
        - If t is a threshold value in `threshold_dim` then F(t) is in `function_values`,
        - F is piecewise linear between each of the t values in `threshold_dim`.

    This function assumes that:
        - `threshold_dim` is a dimension of `function_values`
        - coordinates of `threshold_dim` are increasing.

    Args:
        function_values (xr.DataArray): array of function values F(t).
        threshold_dim (xr.DataArray): dimension along which to integrate.

    Returns:

        xr.DataArray: Integral values and `threshold_dim` collapsed:

        - Returns value of the integral with `threshold_dim` collapsed and other dimensions preserved.
        - Returns NaN if there are less than two non-NaN function_values.

    Examples:
        >>> import xarray as xr
        >>> from scores.processing.cdf import integrate_square_piecewise_linear

        >>> # Linear function F(t) = t on [0, 2]
        >>> t_vals = [0, 0.5, 1, 1.2, 2]
        >>> f_vals = [0, 0.75, 1, 1.1, 2]

        >>> da = xr.DataArray(f_vals, coords={"threshold": t_vals}, dims="threshold")

        >>> integrate_square_piecewise_linear(da, "threshold")
        <xarray.DataArray ()> Size: 8B
        array(2.67583333)

        >>> # Multiple stations
        >>> data = [[0, 0.75, 1, 1.1, 2], [0, 0.5, 1, 1.2, 2]]
        >>> coords = {"station": [1, 2], "threshold": [0, 0.5, 1, 1.2, 2]}
        >>> da = xr.DataArray(data, coords=coords, dims=["station", "threshold"])

        >>> integrate_square_piecewise_linear(da, "threshold")
        <xarray.DataArray (station: 2)> Size: 16B
        array([2.67583333, 2.66666667])
        Coordinates:
          * station  (station) int64 16B 1 2
    """

    # notation: Since F is piecewise linear we have
    # F(t) = mt + b, whenever x[i-1] <= t <= x[i].

    # difference in x
    diff_xs = function_values[threshold_dim] - function_values[threshold_dim].shift(**{threshold_dim: 1})

    # difference in function values
    diff_ys = function_values - function_values.shift(**{threshold_dim: 1})

    # gradients m
    m_values = diff_ys / diff_xs

    # y intercepts b
    b_values = function_values.shift(**{threshold_dim: 1})

    # integral for x[i-1] <= t <= x[i]
    piece_integral = (m_values**2) * (diff_xs**3) / 3 + m_values * b_values * (diff_xs**2) + (b_values**2) * diff_xs

    # Need at least one non-NaN piece_integral to return float.
    # Note: need at least two non-NaN function values to get a non-NaN piece_integral.
    return piece_integral.sum(threshold_dim, min_count=1)


def add_thresholds(
    cdf: xr.DataArray,
    threshold_dim: str,
    new_thresholds: Iterable[float],
    fill_method: Literal["linear", "step", "forward", "backward", "none"],
    *,  # Force keywords arguments to be keyword-only
    min_nonnan: int = 2,
) -> xr.DataArray:
    """Takes a CDF data array with dimension `threshold_dim` and adds values from `new_thresholds`.

    The CDF is then filled to replace any NaN values.
    The array `cdf` requires at least 2 non-NaN values along `threshold_dim`.

    Args:
        cdf (xr.DataArray): array of CDF values.
        threshold_dim (str): name of the threshold dimension in `cdf`.
        new_thresholds (Iterable[float]): new thresholds to add to `cdf`.
        fill_method (Literal["linear", "step", "forward", "backward", "none"]): one of "linear",
            "step", "forward" or "backward", as described in `fill_cdf`. If no filling, set to "none".
        min_nonnan (int): passed onto `fill_cdf` for performing filling.

    Returns:
        xr.DataArray: Additional thresholds, and values at those thresholds
        determined by the specified fill method.

    Examples:
        >>> import xarray as xr
        >>> from scores.processing.cdf import add_thresholds

        >>> data = [0.2, 0.8]
        >>> coords = {"threshold": [10, 30]}
        >>> cdf = xr.DataArray(data, coords=coords, dims="threshold")

        >>> add_thresholds(cdf, "threshold", [5, 15, 35], "linear")
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0.05, 0.2 , 0.35, 0.8 , 0.95])
        Coordinates:
          * threshold  (threshold) int64 40B 5 10 15 30 35

        >>> add_thresholds(cdf, "threshold", [5, 15, 35], "step")
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0. , 0.2, 0.2, 0.8, 0.8])
        Coordinates:
          * threshold  (threshold) int64 40B 5 10 15 30 35
    """

    thresholds = np.concatenate((cdf[threshold_dim].values, new_thresholds))
    thresholds = np.sort(pd.unique(thresholds))
    thresholds = thresholds[~np.isnan(thresholds)]

    da_thresholds = xr.DataArray(data=thresholds, dims=[threshold_dim], coords={threshold_dim: thresholds})

    da_cdf = xr.broadcast(cdf, da_thresholds)[0]

    if fill_method != "none":
        da_cdf = fill_cdf(da_cdf, threshold_dim, fill_method, min_nonnan)

    return da_cdf


def fill_cdf(
    cdf: xr.DataArray,
    threshold_dim: str,
    method: Literal["linear", "step", "forward", "backward"],
    min_nonnan: int,
) -> xr.DataArray:
    """
    Fills NaNs in a CDF of a real-valued random variable along `threshold_dim` with appropriate values between 0 and 1.

    Args:
        cdf (xr.DataArray): CDF values, where P(Y <= threshold) = cdf_value for each threshold in `threshold_dim`.
        threshold_dim (str): the threshold dimension in the CDF, along which filling is performed.
        method (Literal["linear", "step", "forward", "backward"]): one of:

            - "linear": use linear interpolation, and if needed also extrapolate linearly. Clip to 0 and 1. \
              Needs at least two non-NaN values for interpolation, so returns NaNs where this condition fails.
            - "step": use forward filling then set remaining leading NaNs to 0. \
              Produces a step function CDF (i.e. piecewise constant).
            - "forward": use forward filling then fill any remaining leading NaNs with backward filling.
            - "backward": use backward filling then fill any remaining trailing NaNs with forward filling.
        min_nonnan (int): the minimum number of non-NaN entries required along `threshold_dim` for filling to
            be performed. All CDF values are set to `np.nan` where this condition fails.
            `min_nonnan` must be at least 2 for the "linear" method, and at least 1 for the other methods.

    Returns:
        xr.DataArray: Containing the same values as `cdf` but with NaNs filled.

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `cdf`.
        ValueError: If `min_nonnan` < 1 when `method="step"` or if `min_nonnan` < 2 when `method="linear"`.
        ValueError: If `method` is not "linear", "step", "forward" or "backward".
        ValueError: If any non-NaN value of `cdf` lies outside the unit interval [0,1].

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>> from scores.processing.cdf import fill_cdf

        >>> data = [np.nan, 0.2, np.nan, 0.8, np.nan]
        >>> coords = {"threshold": [10, 20, 30, 40, 50]}
        >>> cdf = xr.DataArray(data, coords=coords, dims="threshold")

        >>> fill_cdf(cdf, "threshold", "linear", min_nonnan = 2)
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0. , 0.2, 0.5, 0.8, 1. ])
        Coordinates:
          * threshold  (threshold) int64 40B 10 20 30 40 50

        >>> fill_cdf(cdf, "threshold", "step", min_nonnan = 1)
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0. , 0.2, 0.2, 0.8, 0.8])
        Coordinates:
          * threshold  (threshold) int64 40B 10 20 30 40 50

        >>> fill_cdf(cdf, "threshold", "forward", min_nonnan = 1)
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0.2, 0.2, 0.2, 0.8, 0.8])
        Coordinates:
          * threshold  (threshold) int64 40B 10 20 30 40 50

        >>> fill_cdf(cdf, "threshold", "backward", min_nonnan = 1)
        <xarray.DataArray (threshold: 5)> Size: 40B
        array([0.2, 0.2, 0.8, 0.8, 0.8])
        Coordinates:
          * threshold  (threshold) int64 40B 10 20 30 40 50
    """

    if method not in ["linear", "step", "forward", "backward"]:
        raise ValueError("`method` must be 'linear', 'step', 'forward' or 'backward'")

    if not cdf_values_within_bounds(cdf):
        raise ValueError("Input CDF has some values less than 0 or greater than 1.")

    if threshold_dim not in cdf.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `cdf`")

    if min_nonnan < 1 and method != "linear":
        raise ValueError(f"`min_nonnan` must be at least 1 when `method='{method}'`")

    if min_nonnan < 2 and method == "linear":
        raise ValueError("`min_nonnan` must be at least 2 when `method='linear'`")

    # set cdf values to be NaN where min_nonnan requirement fails
    nonnan_count = cdf.count(threshold_dim)
    cdf = cdf.where(nonnan_count >= min_nonnan)

    # NaN filling
    if method == "linear":
        cdf = cdf.interpolate_na(threshold_dim, method="linear", fill_value="extrapolate").clip(min=0, max=1)

    if method == "step":
        cdf = cdf.ffill(threshold_dim).fillna(0)
        # NaN cdfs will now be all zero, so bring back Nans
        cdf = cdf.where(nonnan_count >= min_nonnan)

    if method == "forward":
        cdf = cdf.ffill(threshold_dim).bfill(threshold_dim)

    if method == "backward":
        cdf = cdf.bfill(threshold_dim).ffill(threshold_dim)

    return cdf


def decreasing_cdfs(cdf: xr.DataArray, threshold_dim: str, tolerance: float) -> xr.DataArray:
    """A CDF of a real-valued random variable should be nondecreasing along threshold_dim.

    This is sometimes violated due to rounding issues or bad forecast process.
    `decreasing_cdfs` checks CDF values decrease beyond specified tolerance; that is,
    whenever the sum of the incremental decreases exceeds tolerarance.

    For example, if the CDF values are `[0, 0.4, 0.3, 0.9, 0.88, 1]`
    then the sum of incremental decreases is -0.12. Given a specified positive `tolerance`,
    the CDF values decrease beyond tolerance if the sum of incremental decreases < -`tolerance`.

    Intended use is for CDFs with increasing coordinates along `threshold_dim` dimension, and where
    either each CDF is always NaN or always non-NaN.

    Args:
        cdf (xr.DataArray): data array of CDF values
        threshold_dim (str): threshold dimension, such that P(Y < threshold) = cdf_value.
        tolerance (float): nonnegative tolerance value.

    Returns:
        xr.DataArray: Containing `threshold_dim` collapsed and values True if and only if
        the CDF is decreasing outside tolerance. If the CDF consists only of NaNs then
        the value is False.

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `cdf`.
        ValueError: If `tolerance` is negative.
        ValueError: If coordinates are not increasing along `threshold_dim`.
        ValueError: If some, but not all, CDF values in `cdf` along `threshold_dim` are NaN.

    Examples:
        >>> import xarray as xr
        >>> from scores.processing.cdf import decreasing_cdfs

        >>> # Well-behaved CDF
        >>> coords = {"threshold": [10, 20, 30, 40]}
        >>> good_cdf = xr.DataArray([0, 0.3, 0.7, 1.0],
        ...                         coords=coords, dims="threshold")
        >>> decreasing_cdfs(good_cdf, "threshold", 0.01)
        <xarray.DataArray ()> Size: 1B
        array(False)

        >>> # CDF with small violation within tolerance
        >>> coords = {"threshold": [10, 20, 30, 40, 50]}
        >>> minor_cdf = xr.DataArray([0, 0.4, 0.39, 0.9, 1],
        ...                          coords=coords, dims="threshold")
        >>> decreasing_cdfs(minor_cdf, "threshold", 0.05)
        <xarray.DataArray ()> Size: 1B
        array(False)

        >>> # CDF with violation exceeding tolerance
        >>> coords = {"threshold": [10, 20, 30, 40, 50, 60]}
        >>> bad_cdf = xr.DataArray([0, 0.4, 0.3, 0.9, 0.88, 1],
        ...                        coords=coords, dims="threshold")
        >>> decreasing_cdfs(bad_cdf, "threshold", 0.05)
        <xarray.DataArray ()> Size: 1B
        array(True)

    """
    check_nan_decreasing_inputs(cdf, threshold_dim, tolerance)

    # difference between consecutive terms along threshold_dim
    diff = cdf - cdf.shift(**{threshold_dim: 1})

    result = diff.clip(max=0).sum(dim=threshold_dim) < -tolerance

    return result


def cdf_envelope(
    cdf: xr.DataArray,
    threshold_dim: str,
) -> xr.DataArray:
    """Forecast cumulative distribution functions (CDFs) for real-valued random variables.

    CDFs that are reconstructed from known points on the distribution should be nondecreasing
    with respect to the threshold dimension. However, sometimes this may fail due to rounding
    or poor forecast process. This function returns the "envelope" of the original CDF, which
    consists of two bounding CDFs, both of which are nondecreasing.

    The following example shows values from an original CDF that has a decreasing subsequence
    (and so is not a true CDF). The resulting "upper" and "lower" CDFs minimally adjust
    "original" so that "lower" <= "original" <= "upper".

    - "original": [0, .5, .2, .8, 1]
    - "upper": [0, .5, .5, .8, 1]
    - "lower": [0, .2, .2, .8, 1]

    This function does not perform checks that `0 <= cdf <= 1`.

    Args:
        cdf (xr.DataArray): forecast CDF with thresholds in the thresholds_dim.
        threshold_dim (str): dimension in fcst_cdf that contains the threshold ordinates.

    Returns:
        xr.DataArray: An xarray DataArray consisting of three CDF arrays indexed along the `"cdf_type"` dimension
        with the following indices:

            - "original": same data as `cdf`.
            - "upper": minimally adjusted "original" CDF that is nondecreasing and \
              satisfies "upper" >= "original".
            - "lower": minimally adjusted "original" CDF that is nondecreasing and \
              satisfies "lower" <= "original".

        NaN values in `cdf` are maintained in "original", "upper" and "lower".

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `cdf`.

    Examples:
        >>> import xarray as xr
        >>> from scores.processing.cdf import cdf_envelope

        >>> # CDF with violations
        >>> coords = {"threshold": [10, 20, 30, 40, 50]}
        >>> cdf = xr.DataArray([0, 0.5, 0.2, 0.8, 1.0],
        ...                    coords=coords, dims="threshold")
        >>> result = cdf_envelope(cdf, "threshold")

        >>> result.sel(cdf_type="original").values
        array([0. , 0.5, 0.2, 0.8, 1. ])

        >>> result.sel(cdf_type="upper").values
        array([0. , 0.5, 0.5, 0.8, 1. ])

        >>> result.sel(cdf_type="lower").values
        array([0. , 0.2, 0.2, 0.8, 1. ])

        >>> # Multiple stations
        >>> data = [[0, 0.6, 0.4, 1.0], [0, 0.3, 0.7, 1.0]]
        >>> coords = {"station": ["A", "B"], "threshold": [10, 20, 30, 40]}
        >>> cdf = xr.DataArray(data,
        ...       coords=coords, dims=["station", "threshold"])

        >>> envelope = cdf_envelope(cdf, "threshold")
        >>> envelope.sel(station="A", cdf_type="upper").values
        array([0. , 0.6, 0.6, 1. ])
    """
    if threshold_dim not in cdf.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `cdf`")

    # logic below assumes that cdf[threshold_dim].values are ascending
    cdf = cdf.sortby(threshold_dim)

    result = xr.full_like(cdf.expand_dims({"cdf_type": ["original", "upper", "lower"]}), np.nan)

    dim_idx = cdf.dims.index(threshold_dim)

    # use fmax so as not to propogate nans
    cdf_upper = np.fmax.accumulate(cdf.values, axis=dim_idx)
    cdf_lower = np.flip(
        1 - np.fmax.accumulate(1 - np.flip(cdf.values, axis=dim_idx), axis=dim_idx),
        axis=dim_idx,
    )

    result.loc["original"] = cdf.copy()
    result.loc["upper"] = np.where(~np.isnan(cdf), cdf_upper, np.nan)
    result.loc["lower"] = np.where(~np.isnan(cdf), cdf_lower, np.nan)

    return result
