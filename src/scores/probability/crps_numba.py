"""
Fast and memory-efficient vectorised exact CRPS calculation.

The xarray wrapper function crps_cdf_exact_fast is based on the code for crps_ensemble from xskillscore
https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/probabilistic.py
Copyright xskillscore developers, released under the Apache-2.0 License (as at 11 Dec 2025).

The vectorisation of crps_at_point follows the example of _crps_ensemble_gufunc from properscoring
https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py
Copyright 2015 The Climate Corporation, released under the Apache-2.0 License

"""

import numpy as np
import xarray as xr
from numba import float64, guvectorize, int64, jit

# To avoid numerical instability caused by dividing by a very small number,
# we use a different calculation of the trapezoid volume if the difference
# in y values is less than EPSILON
EPSILON = 1e-8


@jit
def integral_below(x0, x1, y0, y1):  # pragma: no cover
    """Volume between line y=0 and straight line joining (x0, y0), (x1, y1)"""
    if x1 - x0 < EPSILON:
        return 0
    if abs(y1 - y0) < EPSILON:
        return (x1 - x0) * y0**2
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return (1 / (3 * slope)) * ((slope * x1 + intercept) ** 3 - (slope * x0 + intercept) ** 3)


@jit
def integral_above(x0, x1, y0, y1):  # pragma: no cover
    """Volume between line y=1 and straight line joining (x0, y0), (x1, y1)"""
    if x1 - x0 < EPSILON:
        return 0
    if abs(y1 - y0) < EPSILON:
        return (x1 - x0) * (1 - y0) ** 2
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return (-1 / (3 * slope)) * ((1 - slope * x1 - intercept) ** 3 - (1 - slope * x0 - intercept) ** 3)


@guvectorize(
    [(float64, float64[:], float64[:], int64[:], float64[:], float64[:])], "(),(n),(n),(n)->(),()"
)  # pragma: no cover
def crps_at_point(
    obs: float,
    fc: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    res_over: np.ndarray,
    res_under: np.ndarray,
) -> None:
    """CRPS at a single point for a thresholded probabilistic forecast.

    Args:
        obs: observed value
        fc: 1-d array of forecast probabilities
        thresholds: 1-d non-decreasing array of thresholds, same length as forecasts
        weights: 1-d array of weights. Must have same size as fc and have values in {0, 1}
        res_over, res_under: ndarrays same length as forecast, for over-forecast
            and under-forecast errors
    """
    if np.isnan(obs) or np.any(np.isnan(fc)) > 0:
        res_over[0] = np.nan
        res_under[0] = np.nan
        return
    obs_ind = np.searchsorted(thresholds, obs)  # thresholds[obs_ind - 1] <= obs < thresholds[obs_ind]
    over = 0
    under = 0
    if (obs_ind == 0) and (weights[0] == 1):
        over += integral_above(obs, thresholds[0], fc[0], fc[0])
    for i in range(1, obs_ind):
        if weights[i - 1] == 1:
            under += integral_below(thresholds[i - 1], thresholds[i], fc[i - 1], fc[i])
    if (obs_ind > 0) and (obs_ind < len(fc)) and (weights[obs_ind - 1] == 1):
        prob_at_obs = np.interp(
            obs,
            [thresholds[obs_ind - 1], thresholds[obs_ind]],
            [fc[obs_ind - 1], fc[obs_ind]],
        )
        under += integral_below(thresholds[obs_ind - 1], obs, fc[obs_ind - 1], prob_at_obs)
        over += integral_above(obs, thresholds[obs_ind], prob_at_obs, fc[obs_ind])
    for i in range(obs_ind + 1, len(fc)):
        if weights[i - 1] == 1:
            over += integral_above(thresholds[i - 1], thresholds[i], fc[i - 1], fc[i])
    if (obs_ind == len(fc)) and (weights[-1] == 1):
        under += integral_below(thresholds[-1], obs, fc[-1], fc[-1])
    res_over[0] = over
    res_under[0] = under


def crps_threshold(
    observations: np.ndarray, forecasts: np.ndarray, thresholds: np.ndarray, weights: np.ndarray
) -> (np.ndarray, np.ndarray):
    """Pointwise calculation of CRPS for a thresholded probabilistic forecast.

    Args:
        observations: n-dimensional array
        forecasts: n-dimensional array of probability forecasts with same shape as observations
            and additional threshold dimension as last dimension. Forecasts should describe
            a cdf, i.e. they must be in increasing order.
        thresholds: 1-dimensional monotone non-decreasing array with length forecasts.shape[-1]
        weights: n-d array of weights, same size as forecasts


    Returns:
        n-dimensional array with same dimensions as observations
    """
    # Convert NaNs in weights to 0 so that we can convert weight to int.
    # The calling function crps_cdf_exact_fast sets the output to NaN where weights are NaN.
    weights = np.where(np.isnan(weights), 0, weights)
    weights = weights.astype(np.int64)
    return crps_at_point(observations, forecasts, thresholds, weights)


def crps_cdf_exact_fast(
    cdf_fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold_weight: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    include_components=False,
) -> xr.Dataset:
    """
    Calculates exact value of CRPS assuming that:
        - the forecast CDF is continuous piecewise linear, with join points given by
          values in `cdf_fcst`,
        - the observation CDF contains observations in the same units as the CDF threshold dimension
        - the threshold weight function is right continuous with values in {0,1} given
          by `threshold_weight`.

    If these assumptions do not hold, it might be best to use `crps_approximate`, with a
    sufficiently high resolution along `threshold_dim`.

    This function assumes that values along the `threshold_dim` dimension are increasing.

    If numba is installed, the fast vectorised calculation will be used. Otherwise, the basic
    version will be used.

    Returns:
        (xr.Dataset): Dataset with `threshold_dim` collapsed containing DataArrays with
        CRPS and its decomposition, labelled "total", "underforecast_penalty" and
        "overforecast_penalty". NaN is returned if there is a NaN in the corresponding
        `cdf_fcst`, `cdf_obs` or `threshold_weight`.
    """
    # identify where input arrays have no NaN, collapsing `threshold_dim`
    # Mypy doesn't realise the isnan and any come from xarray not numpy
    inputs_without_nan = (
        ~np.isnan(cdf_fcst).any(threshold_dim)  # type: ignore
        & ~np.isnan(obs)  # type: ignore
        & ~np.isnan(threshold_weight).any(threshold_dim)  # type: ignore
    )
    over, under = xr.apply_ufunc(
        crps_threshold,
        obs,
        cdf_fcst,
        cdf_fcst[threshold_dim],
        threshold_weight,
        input_core_dims=[[], [threshold_dim], [threshold_dim], [threshold_dim]],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[float, float],
        keep_attrs=True,
    )
    over, under = over.where(inputs_without_nan), under.where(inputs_without_nan)
    total = over + under
    if "units" in cdf_fcst[threshold_dim].attrs:
        over.attrs["units"] = cdf_fcst[threshold_dim].attrs["units"]
        under.attrs["units"] = cdf_fcst[threshold_dim].attrs["units"]
        total.attrs["units"] = cdf_fcst[threshold_dim].attrs["units"]
    result = total.to_dataset(name="total")
    if include_components:
        result = xr.merge(
            [
                total.rename("total"),
                under.rename("underforecast_penalty"),
                over.rename("overforecast_penalty"),
            ]
        )
    return result
