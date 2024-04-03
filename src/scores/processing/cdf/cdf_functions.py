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
    coords_increasing,
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
        thresholds = np.concatenate((obs.values.flatten(), thresholds))  # type: ignore

    # remove any NaN
    thresholds = [x for x in thresholds if not np.isnan(x)]  # type: ignore

    # pandas.unique retains the original ordering whereas set() may not
    # pandas.unique no longer accepts a simple array as input
    thresholds = pd.Series(thresholds)
    thresholds = np.sort(pd.unique(thresholds))  # type: ignore

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

    Calculates integral(F(t)^2), where
        - If t in a threshold value in `threshold_dim` then F(t) is in `function_values`,
        - F is piecewise linear between each of the t values in `threshold_dim`.
    Returns value of the integral with `threshold_dim` collapsed and other dimensions preserved.
    Returns NaN if there are less than two non-NaN function_values.

    This function assumes that
        - `threshold_dim` is a dimension of `function_values`
        - coordinates of `threshold_dim` are increasing.

    Args:
        function_values (xr.DataArray): array of function values F(t).
        threshold_dim (xr.DataArray): dimension along which to integrate.

    Returns:
        xr.DataArray: Integral values and `threshold_dim` collapsed.
    """

    # notation: Since F is piecewise linear we have
    # F(t) = mt + b, whenever x[i-1] <= t <= x[i].

    # difference in x
    diff_xs = function_values[threshold_dim] - function_values[threshold_dim].shift(**{threshold_dim: 1})  # type: ignore

    # difference in function values
    diff_ys = function_values - function_values.shift(**{threshold_dim: 1})  # type: ignore

    # gradients m
    m_values = diff_ys / diff_xs

    # y intercepts b
    b_values = function_values.shift(**{threshold_dim: 1})  # type: ignore

    # integral for x[i-1] <= t <= x[i]
    piece_integral = (
        (m_values**2) * (diff_xs**3) / 3 + m_values * b_values * (diff_xs**2) + (b_values**2) * diff_xs
    )

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
    """

    thresholds = np.concatenate((cdf[threshold_dim].values, new_thresholds))  # type: ignore
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
    """Fills NaNs in a CDF of a real-valued random variable along `threshold_dim` with appropriate values between 0 and 1.

    Args:
        cdf (xr.DataArray): CDF values, where P(Y <= threshold) = cdf_value for each threshold in `threshold_dim`.
        threshold_dim (str): the threshold dimension in the CDF, along which filling is performed.
        method (Literal["linear", "step", "forward", "backward"]): one of
            - "linear": use linear interpolation, and if needed also extrapolate linearly. Clip to 0 and 1.
              Needs at least two non-NaN values for interpolation, so returns NaNs where this condition fails.
            - "step": use forward filling then set remaining leading NaNs to 0.
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
    For example, if the CDF values are
        [0, 0.4, 0.3, 0.9, 0.88, 1]
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
    """
    check_nan_decreasing_inputs(cdf, threshold_dim, tolerance)

    # difference between consecutive terms along threshold_dim
    diff = cdf - cdf.shift(**{threshold_dim: 1})  # type: ignore

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
        "original": [0, .5, .2, .8, 1]
        "upper": [0, .5, .5, .8, 1]
        "lower": [0, .2, .2, .8, 1]

    This function does not perform checks that `0 <= cdf <= 1`.

    Args:
        cdf (xr.DataArray): forecast CDF with thresholds in the thresholds_dim.
        threshold_dim (str): dimension in fcst_cdf that contains the threshold ordinates.

    Returns:
        An xarray DataArray consisting of three CDF arrays indexed along the `"cdf_type"` dimension
        with the following indices:
            - "original": same data as `cdf`.
            - "upper": minimally adjusted "original" CDF that is nondecreasing and
              satisfies "upper" >= "original".
            - "lower": minimally adjusted "original" CDF that is nondecreasing and
              satisfies "lower" <= "original".
        NaN values in `cdf` are maintained in "original", "upper" and "lower".

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `cdf`.
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


def check_cdf(
    cdf: xr.DataArray,
    threshold_dim: str,
    dims: Optional[Iterable[str]],
    varname_cdf: str = "cdf",
    varname_threshold_dim: str = "threshold_dim",
    varname_dims: str = "dims",
):
    """
    Performs various checks on a CDF values of a real-valued random variable
    in an array `cdf` and its dimensions. Does not check that `cdf` is nondecreasing
    along the `threshold_dim` dimension;
    use the function `nan_decreasing_cdfs` instead.

    Args:
        cdf: array of CDF values to check.
        threshold_dim: name of the threshold dimension in `cdf`.
        dims: if not `None`, a subset of dimensions of `cdf`, not including
            `threshold_dim`.
        varname_cdf: the variable name for `cdf` used in the error message.
        varname_threshold_dim: the variable name for `threshold_dim` used in the error message.
        varname_dims: the variable name for `dims` used in the error message.

    Raises:
        ValueError: if `threshold_dim` is not a dimension of `cdf`.
        ValueError: if `threshold_dim` is in `dims`.
        ValueError: if `dims` is not a subset of `cdf.dims`.
        ValueError: if `cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if `cdf[threshold_dim]` coordinates are not increasing.
    """
    if threshold_dim not in cdf.dims:
        raise ValueError(f"`{varname_threshold_dim}` is not a dimension of `{varname_cdf}`")

    if dims is not None:
        if threshold_dim in dims:
            raise ValueError(f"`{varname_threshold_dim}` is in `{varname_dims}`")
        if not set(dims).issubset(cdf.dims):
            raise ValueError(f"`{varname_dims}` is not a subset of dimensions of `{varname_cdf}`")

    if not cdf_values_within_bounds(cdf):
        raise ValueError(f"values of `{varname_cdf}` must be in the closed interval [0,1]")

    if not coords_increasing(cdf, threshold_dim):
        raise ValueError(f"coordinates along `{varname_threshold_dim}` of `{varname_cdf}` must be increasing")


def check_cdf_support(
    cdf: xr.DataArray,
    threshold_dim: str = "threshold",
    lower_supp: float = 0,
    upper_supp: float = np.inf,
    varname_cdf: str = "cdf",
    varname_threshold_dim: str = "threshold_dim",
):
    """
    Checks that the CDF of a real-valued random variable is supported on
    the closed interval [lower_supp, upper_supp], i.e., that

        - cdf(x) = 0 when x < `lower_support`
        - cdf(x) = 1 when x > `upper_support`

    for x in `cdf[threshold_dim].values`.

    NaNs are ignored. Tests for "equality" uses `np.allclose`.

    Args:
        cdf: array of CDF values, with threshold dimension `threshold_dim`.
        threshold_dim: name of the threshold dimension in `cdf`.
        lower_supp: left endpoint of the interval of support. Can be `-np.inf`.
        upper_supp: right endpoint of the interval of support. Can be `np.inf`.
        varname_cdf: the variable name for `cdf` used in the error message.
        varname_threshold_dim: the variable name for `cdf` used in the error message.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: if `lower_support` > `upper_support`.
        ValueError: if cdf(x) != 0 when x < `lower_support`.
        ValueError: if cdf(x) != 1 when x > `upper_support`.
    """
    if upper_supp < lower_supp:
        raise ValueError("`upper_supp < lower_supp`")

    upper = cdf.where(cdf[threshold_dim] > upper_supp).values
    lower = cdf.where(cdf[threshold_dim] < lower_supp).values

    if not np.allclose(upper[~np.isnan(upper)], 1):
        raise ValueError(f"`{varname_cdf}` is not 1 when `{varname_cdf}[{varname_threshold_dim}] > {upper_supp}`")

    if not np.allclose(lower[~np.isnan(lower)], 0):
        raise ValueError(f"`{varname_cdf}` is not 0 when `{varname_cdf}[{varname_threshold_dim}] < {lower_supp}`")


def expectedvalue_from_cdf(
    cdf: xr.DataArray,
    threshold_dim: str = "threshold",
    nonnegative_support: bool = True,
) -> xr.DataArray:
    """
    Returns the expected value E(Y) of a real-valued random variable Y with
    distribution given by `cdf`.

    The current implementation assumes that Y >= 0, i.e., the CDF has nonnnegative support.
    This assumption may be relaxed in a future implementation.

    The calculation is based on the formula

        E(Y) = integral(1 - cdf(x), x >=0),

    where it is assumed that the CDF is  piecewise linear between CDF values given by the
    array `cdf` when x >= 0.

    Args:
        cdf: array of CDF values, with threshold dimension `threshold_dim`.
        threshold_dim: name of the threshold dimension in `cdf`.
        nonnegative_support: `True` if the support of the probability measure
            described by `cdf` is contained in the half line [0, inf).

    Returns:
        xr.DataArray containing the expected value of each CDF in `cdf`, with
        `threshold_dim` collapsed.

    Raises:
        ValueError: if `nonnegative_support = False`.
        ValueError: if `threshold_dim` is not a dimension of `cdf`.
        ValueError: if `cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if `cdf[threshold_dim]` coordinates are not increasing.
        ValueError: if `cdf` is not 0 when `cdf[threshold_dim] < 0` and
            `nonnegative_support = True`.
    """
    if nonnegative_support:
        check_cdf_support(cdf, threshold_dim, 0, np.inf)
    else:
        raise ValueError("This function currently only handles the case when the cdf has nonnegative support")

    check_cdf(cdf, threshold_dim, None, "cdf", "threshold_dim", "dims")

    # only select CDF values where 0 <= threshold_dim.
    integrand = 1 - cdf.sel({threshold_dim: slice(0, np.inf)})

    return integrand.integrate(threshold_dim)


def variance_from_cdf(cdf: xr.DataArray, threshold_dim: str = "threshold") -> xr.DataArray:
    """
    Returns the variance Var(Y) of a random variable Y with distribution given by `cdf`,
    under the assumption that Y >= 0. The calculation is based on the formula

        Var(Y) = 2 * integral(x * (1 - cdf(x)), x >= 0) + E(Y)^2,

    where it is assumed that the CDF is  piecewise linear between CDF values given by the
    array `cdf` when x >= 0.

    Args:
        cdf: array of CDF values, with threshold dimension `threshold_dim`.
        threshold_dim: name of the threshold dimension in `cdf`.

    Returns:
        xr.DataArray containing the variance Var(Y) associated with each CDF in `cdf`,
        with `threshold_dim` collapsed.

    Raises:
        ValueError: if `threshold_dim` is not a dimension of `cdf`.
        ValueError: if `cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if `cdf[threshold_dim]` coordinates are not increasing.
        ValueError: if `cdf` is not 0 when `cdf[threshold_dim] < 0`.
    """
    check_cdf(cdf, threshold_dim, None, "cdf", "threshold_dim", "dims")
    check_cdf_support(cdf, threshold_dim, 0, np.inf)

    expected_value = expectedvalue_from_cdf(cdf, threshold_dim)

    # only use values of CDF where threshold >= 0.
    cdf = cdf.sel({threshold_dim: slice(0, np.inf)})
    integral = _var_from_cdf(cdf, threshold_dim)
    result = 2 * integral - expected_value**2

    return result


def _var_from_cdf(function_values, threshold_dim):
    """
    Calculates the integral

        integral(t * (1 - F(t)))

    where it is assumed that the function F is piecewise linear between values in
    given by the array `function_values`.

    If there is a NaN along `threshold_dim`, then NaN is returned.
    """
    # notation: F_i(t) = m_i * t + b_i whenever x[i-1] <= t <= x[i]
    # and x[i] is a value in `function_values[threshold_dim]`.

    x_values = function_values[threshold_dim]
    x_shifted = function_values[threshold_dim].shift(**{threshold_dim: 1})
    # difference in x values
    diff_xs = x_values - x_shifted
    # difference in function values y_i = F(x[i])
    diff_ys = function_values - function_values.shift(**{threshold_dim: 1})
    # gradients m
    m_values = diff_ys / diff_xs
    # intercepts b_i
    b_values = function_values - m_values * x_values
    # integral(t * (1 - F(t))) on the interval (x[i-1], x[i]), for each i, using calculus:
    integral_i = (1 - b_values) * (x_values**2 - x_shifted**2) / 2 - m_values * (x_values**3 - x_shifted**3) / 3

    integral = integral_i.sum(threshold_dim)
    # return NaN if NaN in function_values
    integral = integral.where(~np.isnan(function_values).any(threshold_dim))

    return integral
