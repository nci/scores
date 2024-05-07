"""
Functions for calculating a modified Diebold-Mariano test statistic
"""
import warnings
from typing import Literal

import numpy as np
import scipy as sp
import xarray as xr
from scipy.optimize import least_squares

from scores.stats.statistical_tests.acovf import acovf
from scores.utils import dims_complement


def diebold_mariano(  # pylint: disable=R0914
    da_timeseries: xr.DataArray,
    ts_dim: str,
    h_coord: str,
    *,  # Force keywords arguments to be keyword-only
    method: Literal["HG", "HLN"] = "HG",
    confidence_level: float = 0.95,
    statistic_distribution: Literal["normal", "t"] = "normal",
) -> xr.Dataset:
    """
    Given an array of (multiple) timeseries, with each timeseries consisting of score
    differences for h-step ahead forecasts, calculates a modified Diebold-Mariano test
    statistic for each timeseries. Several other statistics are also returned such as
    the confidence that the population mean of score differences is greater than zero
    and confidence intervals for that mean.

    Two methods for calculating the test statistic have been implemented: the "HG"
    method Hering and Genton (2011) and the "HLN" method of Harvey, Leybourne and
    Newbold (1997). The default "HG" method has an advantage of only generating positive
    estimates for the spectral density contribution to the test statistic. For further
    details see `scores.stats.confidence_intervals.impl._dm_test_statistic`.

    Prior to any calculations, NaNs are removed from each timeseries. If there are NaNs
    in `da_timeseries` then a warning will occur. This is because NaNs may impact the
    autocovariance calculation.

    To determine the value of h for each timeseries of score differences of h-step ahead
    forecasts, one may ask 'How many observations of the phenomenon will be made between
    making the forecast and having the observation that will validate the forecast?'
    For example, suppose that the phenomenon is afternoon precipitation accumulation in
    New Zealand (00z to 06z each day). Then a Day+1 forecast issued at 03z on Day+0 will
    be a 2-ahead forecast, since Day+0 and Day+1 accumulations will be observed before
    the forecast can be validated. On the other hand, a Day+1 forecast issued at 09z on
    Day+0 will be a 1-step ahead forecast. The value of h for each timeseries in the
    array needs to be specified in one of the sets of coordinates.
    See the example below.

    Confidence intervals and "confidence_gt_0" statistics are calculated using the
    test statistic, which is assumed to have either the standard normal distribution
    or Student's t distribution with n - 1 degrees of freedom (where n is the length of
    the timeseries). The distribution used is specified by `statistic_distribution`. See
    Harvey, Leybourne and Newbold (1997) for why the t distribution may be preferred,
    especially for shorter timeseries.

    If `da_timeseries` is a chunked array, data will be brought into memory during
    this calculation due to the autocovariance implementation.

    Args:
        da_timeseries: a 2 dimensional array containing the timeseries.
        ts_dim: name of the dimension which identifies each timeseries in the array.
        h_coord: name of the coordinate specifying, for each timeseries, that the
            timeseries is an h-step ahead forecast. `h_coord` coordinates must be
            indexed by the dimension `ts_dim`.
        method: method for calculating the test statistic, one of "HG" or "HLN".
        confidence_level: the confidence level, between 0 and 1 exclusive, at which to
            calculate confidence intervals.
        statistic_distribution: the distribution of the test-statistic under the null
            hypothesis of equipredictive skill. Used to calculate the "confidence_gt_0"
            statistic and confidence intervals. One of "normal" or "t" (for Student's t
            distribution).

    Returns:
        Dataset, indexed by `ts_dim`, with six variables:
        - "mean": the mean value for each timeseries, ignoring NaNs
        - "dm_test_stat": the modified Diebold-Mariano test statistic for each
          timeseries
        - "timeseries_len": the length of each timeseries, with NaNs removed.
        - "confidence_gt_0": the confidence that the mean value of the population is
          greater than zero, based on the specified `statistic_distribution`.
          Precisely, it is the value of the cumululative distribution function
          evaluated at `dm_test_stat`.
        - "ci_upper": the upper end point of a confidence interval about the mean at
          specified `confidence_level`.
        - "ci_lower": the lower end point of a confidence interval about the mean at
          specified `confidence_level`.

    Raises:
        ValueError: if `method` is not one of "HG" or "HLN".
        ValueError: if `statistic_distribution` is not one of "normal" or "t".
        ValueError: if `0 < confidence_level < 1` fails.
        ValueError: if `len(da_timeseries.dims) != 2`.
        ValueError: if `ts_dim` is not a dimension of `da_timeseries`.
        ValueError: if `h_coord` is not a coordinate of `da_timeseries`.
        ValueError: if `ts_dim` is not the only dimension of
            `da_timeseries[h_coord]`.
        ValueError: if `h_coord` values aren't positive integers.
        ValueError: if `h_coord` values aren't less than the lengths of the
            timeseries after NaNs are removed.
        RuntimeWarnning: if there is a NaN in diffs.

    References:
        - Diebold and Mariano, 'Comparing predictive accuracy', Journal of Business and
          Economic Statistics 13 (1995), 253-265.
        - Hering and Genton, 'Comparing spatial predictions',
          Technometrics 53 no. 4 (2011), 414-425.
        - Harvey, Leybourne and Newbold, 'Testing the equality of prediction mean
          squared errors', International Journal of Forecasting 13 (1997), 281-291.

    Example:

        This array gives three timeseries of score differences.
        Coordinates in the "lead_day" dimension uniquely identify each timeseries.
        Here `ts_dim="lead_day"`.
        Coordinates in the "valid_date" dimension give the forecast validity timestamp
        of each item in the timeseries.
        The "h" coordinates specify that the timeseries are for 2, 3 and 4-step
        ahead forecasts respectively. Here `h_coord="h"`.

        >>> da_timeseries = xr.DataArray(
        ...     data=[[1, 2, 3.0, 4, np.nan], [2.0, 1, -3, -1, 0], [1.0, 1, 1, 1, 1]],
        ...     dims=["lead_day", "valid_date"],
        ...     coords={
        ...         "lead_day": [1, 2, 3],
        ...         "valid_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
        ...         "h": ("lead_day", [2, 3, 4]),
        ...     },
        ... )

        >>> dm_test_stats(da_timeseries, "lead_day", "h")
    """
    if method not in ["HLN", "HG"]:
        raise ValueError("`method` must be one of 'HLN' or 'HG'.")

    if statistic_distribution not in ["normal", "t"]:
        raise ValueError("`statistic_distribution` must be one of 'normal' or 't'.")

    if not 0 < confidence_level < 1:
        raise ValueError("`confidence_level` must be strictly between 0 and 1.")

    if len(da_timeseries.dims) != 2:
        raise ValueError("`da_timeseries` must have exactly two dimensions.")

    if ts_dim not in da_timeseries.dims:
        raise ValueError(f"`ts_dim` '{ts_dim}' must be a dimension of `da_timeseries`.")

    if h_coord not in da_timeseries.coords:
        raise ValueError("`h_coord` must be among the coordinates of `da_timeseries`.")

    # the following will also catch NaNs in da_timeseries[h_coord]
    # It allows values like 7.0 to pass, which is OK for the application.
    if any(da_timeseries[h_coord].values % 1 != 0):
        raise ValueError("Every value in `da_timeseries[h_coord]` must be an integer.")

    if (da_timeseries[h_coord] <= 0).any():
        raise ValueError("Every value in `da_timeseries[h_coord]` must be positive.")

    other_dim = dims_complement(da_timeseries, dims=[ts_dim])[0]
    da_timeseries_len = da_timeseries.count(other_dim)

    if (da_timeseries_len <= da_timeseries[h_coord]).any():
        msg = "Each `h_coord` value must be less than the length of the corresponding timeseries"
        raise ValueError(msg + " after NaNs are removed")

    ts_dim_len = len(da_timeseries[ts_dim])
    test_stats = np.empty([ts_dim_len])
    ts_mean = da_timeseries.mean(other_dim).values

    for i in range(ts_dim_len):
        timeseries = da_timeseries.isel({ts_dim: i})
        h = int(timeseries[h_coord])
        test_stats[i] = _dm_test_statistic(timeseries.values, h, method=method)

    if statistic_distribution == "normal":
        pvals = sp.stats.norm.cdf(test_stats)
        ci_quantile = sp.stats.norm.ppf(1 - (1 - confidence_level) / 2)
    else:
        pvals = sp.stats.t.cdf(test_stats, da_timeseries_len.values - 1)
        ci_quantile = sp.stats.t.ppf(1 - (1 - confidence_level) / 2, da_timeseries_len.values - 1)

    result = xr.Dataset(
        data_vars={
            "mean": ([ts_dim], ts_mean),
            "dm_test_stat": ([ts_dim], test_stats),
            "timeseries_len": ([ts_dim], da_timeseries_len.values),
            "confidence_gt_0": ([ts_dim], pvals),
            "ci_upper": ([ts_dim], ts_mean * (1 + ci_quantile / test_stats)),
            "ci_lower": ([ts_dim], ts_mean * (1 - ci_quantile / test_stats)),
        },
        coords={ts_dim: da_timeseries[ts_dim].values},
    )

    return result


def _dm_test_statistic(diffs: np.ndarray, h: int, *, method: Literal["HG", "HLN"] = "HG") -> float:
    """
    Given a timeseries of score differences for h-step ahead forecasts, as a 1D numpy
    array, returns a modified Diebold-Mariano test statistic. NaNs are removed prior
    to computing the statistic.

    Two methods for computing the test statistic can be used: either the "HLN" method of
    Harvey, Leybourne and Newbold (1997), or "HG" method of Hering and Genton (2011).

    Both methods use a different technique for estimating the spectral density of
    `diffs` at frequency 0, compared with Diebold and Mariano (1995). The HLN method
    uses an improved and less biased estimate (see V_hat (see Equation (5) in Harvey)).
    However, this estimate can sometimes be nonpositive, in which case NaN is returned.

    The HG method estimates the spectral density component using an exponential model
    for the autocovariances of `diffs`, so that positivity is guaranteed.
    Hering and Genton (2011) fit model parameters using empirical autocovariances
    computed up to half of the maximum lag. In this implementation, empirical
    autocovariances are computed up to half of the maximum lag or a lag of `h`,
    whichever is larger. Model parameters are computed using
    `scipy.optimize.least_squares`. It is assumed that the two model parameters (sigma
    and theta, in the notation of Hering and Genton (2011)) are positive.

    In both methods, if the `diff` sequence consists only of 0 values then NaN is
    returned.

    Args:
        diffs: timeseries of score difference as a 1D numpy array, assumed not all NaN.
        h: integer indicating that forecasts are h-step ahead, assumed to be positive and
            less than the length of the timeseries with NaNs removed.
        method: the method for computing the test statistic, either "HG" or "HLN".

    Returns:
        Modified Diebold-Mariano test statistic for sequence of score differences, with
        NaNs removed.

    Raises:
        ValueError: if `method` is not one of "HLN" or "HG".
        ValueError: if `0 < h < len(diffs)` fails after NaNs removed.
        RuntimeWarnning: if there is a NaN in diffs.

    References:
        - Diebold and Mariano, 'Comparing predictive accuracy', Journal of Business and
          Economic Statistics 13 (1995), 253-265.
        - Hering and Genton, 'Comparing spatial predictions',
          Technometrics 53 no. 4 (2011), 414-425.
        - Harvey, Leybourne and Newbold, 'Testing the equality of prediction mean
          squared errors', International Journal of Forecasting 13 (1997), 281-291.
    """
    if method not in ["HLN", "HG"]:
        raise ValueError("`method` must be one of 'HLN' or 'HG'.")

    if np.isnan(np.sum(diffs)):
        warnings.warn(
            RuntimeWarning(
                "A least one NaN value was detected in `da_timeseries`. This may impact the "
                "calculation of autocovariances."
            )
        )

    diffs = diffs[~np.isnan(diffs)]

    if not 0 < h < len(diffs):
        raise ValueError("The condition `0 < h < len(diffs)`, after NaNs removed, failed.")

    nonzero_diffs = diffs[diffs != 0]

    if len(nonzero_diffs) == 0:
        test_stat = np.nan
    elif method == "HLN":
        test_stat = _hln_method_stat(diffs, h)
    else:  # method == 'HG'
        test_stat = _hg_method_stat(diffs, h)

    return test_stat  # type: ignore


def _hg_func(pars: list, lag: np.ndarray, acv: np.ndarray) -> np.ndarray:
    """
    Function whose values are to be minimised as part of the HG method for estimating
    the spectral density at 0.

    Args:
        pars: list of two model parameters (sigma and theta in the notation of Hering
            and Genton 2011)
        lag: 1D numpy array of the form [0, 1, 2, ..., n - 1], where n is the length
            of `acv`
        acv: 1D numpy array of empirical autocovariances with lags corresponding
            to `lag`

    Returns:
        Difference between modelled and empirical autocoveriances.

    References:
        Hering and Genton, 'Comparing spatial predictions',
        Technometrics 53 no. 4 (2011), 414-425.
    """
    return (pars[0] ** 2) * np.exp(-3 * lag / pars[1]) - acv  # type: ignore


def _hg_method_stat(diffs: np.ndarray, h: int) -> float:
    """
    Calculates the modified Diebold-Mariano test statistic using the "HG" method.
    Assumes that h < len(diffs).

    Args:
        diffs: a single (1D array) timeseries of score differences with NaNs removed.
        h: integer indicating that forecasts are h-step ahead, assumed to be positive
            and less than the length of the timeseries with NaNs removed.

    Returns:
        Diebold-Mariano test statistic using the HG method.
    """

    n = len(diffs)

    # use an exponential model for autocovariances of `diffs`
    max_lag = int(max(np.floor((n - 1) / 2), h))
    sample_autocvs = acovf(diffs)[0:max_lag]
    sample_lags = np.arange(max_lag)
    model_params = least_squares(_hg_func, [1, 1], args=(sample_lags, sample_autocvs), bounds=(0, np.inf)).x

    # use the model autocovariances to estimate spectral density at 0
    all_lags = np.arange(n)
    model_autocovs = (model_params[0] ** 2) * np.exp(-3 * all_lags / model_params[1])
    density_estimate = model_autocovs[0] + 2 * np.sum(model_autocovs[1:])
    test_stat = np.mean(diffs) / np.sqrt(density_estimate / n)

    return test_stat  # type: ignore


def _hln_method_stat(diffs: np.ndarray, h: int) -> float:
    """
    Given a timeseries of score differences for h-step ahead forecasts, as a 1D numpy
    array without NaNs, returns the modified Diebold-Mariano test statistic of
    Harvey et al (1997).

    If the value V_hat (see Equation (5) in Harvey) is nonpositive then NaN is returned.

    Args:
        diffs: timeseries of score difference as a 1D numpy array without any NaNs.
        h: integer indicating that forecasts are h-step ahead, assumed to be positive
            and less than the length of the timeseries with NaNs removed.

    Returns:
        Diebold-Mariano test statistic using the HLN method.
    """
    n = len(diffs)
    diffs_bar = np.mean(diffs)

    # Harvey (1997) Equation (3)
    test_stat = diffs_bar / _dm_v_hat(diffs, diffs_bar, n, h) ** 0.5

    # Harvey (1997) Equation (9)
    correction_factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
    test_stat = (correction_factor**0.5) * test_stat

    return test_stat  # type: ignore


def _dm_gamma_hat_k(diffs: np.ndarray, diffs_bar: float, n: int, k: int) -> float:
    """
    Computes the quantity (n - k) * gamma_hat_star_k of Equation (5) in
    Harvey et al (1997).

    Args:
        diffs: a single timeseries of score differences with NaNs removed.
        diffs_bar: mean of diffs.
        n: length of diffs.
        k: integer between 1 and n-1 (inclusive), where n = len(diffs)

    Returns:
        The quantity (n - k) * gamma_hat_star_k.
    """
    prod = (diffs[k:n] - diffs_bar) * (diffs[0 : n - k] - diffs_bar)

    return np.sum(prod)  # type: ignore


def _dm_v_hat(diffs: np.ndarray, diffs_bar: float, n: int, h: int) -> float:
    """
    Computes the the quantity V_hat(d_bar) of Equation (5) in Harvey et al (1997).

    Args:
        diffs: a single timeseries of score differences with NaNs removed.
        diffs_bar: mean of diffs.
        n: length of diffs.
        h: integer between 1 and n - 1 (inclusive), where n = len(diffs)

    Returns:
        The quantity V_hat(d_bar). If the result is not positive, NaN is returned.
    """
    summands = np.empty(h - 1)
    for k in range(h - 1):
        summands[k] = _dm_gamma_hat_k(diffs, diffs_bar, n, k + 1)

    result = (_dm_gamma_hat_k(diffs, diffs_bar, n, 0) + 2 * np.sum(summands)) / n**2

    if result <= 0:
        result = np.nan  # type: ignore

    return result  # type: ignore
