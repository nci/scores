"""
This module contains methods which may be used for scoring multicategorical forecasts
"""
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from scores.utils import check_dims, dims_complement, gather_dimensions
from scores.functions import apply_weights


def firm(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    risk_parameter: float,
    categorical_thresholds: Sequence[float],
    threshold_weights: Sequence[Union[float, xr.DataArray]],
    discount_distance: Optional[float] = 0,
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[xr.DataArray]=None
) -> xr.Dataset:
    """
    Calculates the FIxed Risk Multicategorical (FIRM) score including the
    underforecast and overforecast penalties.

    `categorical_thresholds` and `threshold_weights` must be the same length.

    Args:
        fcst: An array of real-valued forecasts that we want to treat categorically.
        obs: An array of real-valued observations that we want to treat categorically.
        risk_parameter: Risk parameter (alpha) for the FIRM score. The value must
            satisfy 0 < `risk_parameter` < 1.
        categorical_thresholds: Category thresholds (thetas) to delineate the
            categories.
        threshold_weights: Weights that specify the relative importance of forecasting on
            the correct side of each category threshold. Either a positive
            float can be supplied for each categorical threshold or an
            xr.DataArray (with no negative values) can be provided for each
            categorical threshold as long as its dims are a subset of `obs` dims.
            NaN values are allowed in the xr.DataArray. For each NaN value at a
            given coordinate, the FIRM score will be NaN at that coordinate,
            before dims are collapsed.
        discount_distance: An optional discounting distance parameter which
            satisfies `discount_distance` >= 0 such that the cost of misses and
            false alarms are discounted whenever the observation is within
            distance `discount_distance` of the forecast category. A value of 0
            will not apply any discounting.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the FIRM score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating FIRM. All other dimensions will be reduced.
            As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the FIRM score at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely. Only one of `reduce_dims` and `preserve_dims` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        An xarray Dataset with data vars:

        * firm_score: A score for a single category for each coord based on
          the FIRM framework.
        * overforecast_penalty: Penalty for False Alarms.
        * underforecast_penalty: Penalty for Misses.

    Raises:
        ValueError: if `len(categorical_thresholds) < 1`.
        ValueError: if `categorical_thresholds` and `threshold_weights` lengths
            are not equal.
        ValueError: if `risk_parameter` <= 0 or >= 1.
        ValueError: if any values in `threshold_weights` are <= 0.
        ValueError: if `discount_distance` is not None and < 0.
        scores.utils.DimensionError: if `threshold_weights` is a list of xr.DataArrays
            and if the dimensions of these xr.DataArrays is not a subset of the `obs` dims.

    Note:
        Setting `discount distance` to None or 0, will mean that no
        discounting is applied. This means that errors will be penalised
        strictly categorically.

        Setting `discount distance` to np.inf means that the cost of a miss
        is always proportional to the distance of the observation from the
        threshold, and similarly for false alarms.

    References:
        Taggart, R., Loveday, N. and Griffiths, D., 2022. A scoring framework for tiered
        warnings and multicategorical forecasts based on fixed risk measures. Quarterly
        Journal of the Royal Meteorological Society, 148(744), pp.1389-1406.
    """
    _check_firm_inputs(
        obs,
        risk_parameter,
        categorical_thresholds,
        threshold_weights,
        discount_distance,
    )
    total_score = []
    for categorical_threshold, weight in zip(categorical_thresholds, threshold_weights):
        score = weight * _single_category_score(
            fcst, obs, risk_parameter, categorical_threshold, discount_distance
        )
        total_score.append(score)
    summed_score = sum(total_score)
    reduce_dims = gather_dimensions(
        fcst.dims, obs.dims, reduce_dims, preserve_dims
    )
    summed_score = apply_weights(summed_score, weights)
    score = summed_score.mean(dim=reduce_dims)

    return score


def _check_firm_inputs(
    obs,
    risk_parameter,
    categorical_thresholds,
    threshold_weights,
    discount_distance,
):
    """
    Checks that the FIRM inputs are suitable
    """
    if len(categorical_thresholds) < 1:
        raise ValueError("`categorical_thresholds` must have at least one threshold")

    if not len(categorical_thresholds) == len(threshold_weights):
        raise ValueError(
            "The length of `categorical_thresholds` and `weights` must be equal"
        )
    if risk_parameter <= 0 or risk_parameter >= 1:
        raise ValueError("0 < `risk_parameter` < 1 must be satisfied")

    for count, weight in enumerate(threshold_weights):
        if isinstance(weight, xr.DataArray):
            check_dims(weight, obs.dims, "subset")
            if np.any(weight <= 0):
                raise ValueError(
                    f"""
                    No values <= 0 are allowed in `weights`. At least one
                    negative value was found in index {count} of `weights`
                    """
                )
        elif weight <= 0:
            raise ValueError("All values in `weights` must be > 0")
    if discount_distance is not None:
        if discount_distance < 0:
            raise ValueError("`discount_distance` must be >= 0")


def _single_category_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    risk_parameter: float,
    categorical_threshold: float,
    discount_distance: float,
) -> xr.Dataset:
    """
    Calculates the score for a single category for the `firm` metric at each
    coord. Under-forecast and over-forecast penalties are also calculated

    Args:
        fcst: An array of real-valued forecasts.
        obs: An array of real-valued observations.
        risk_parameter: Risk parameter (alpha) for the FIRM score.
            Must satisfy 0 < risk parameter < 1. Note that `firm` checks this
            rather than this function.
        categorical_threshold: Category threshold (theta) to delineate the
            category.
        discount_distance: A discounting distance parameter which must
            be >= 0 such that the cost of misses and false alarms are
            discounted whenever the observation is within distance
            `discount_distance` of the forecast category. A value of 0
            will not a apply any discounting.

    Returns:
        An xarray Dataset with data vars:

            * firm_score: a score for a single category for each coord
              based on the FIRM framework. All dimensions are preserved.
            * overforecast_penalty: Penalty for False Alarms.
            * underforecast_penalty: Penalty for Misses.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    fcst, obs = xr.align(fcst, obs)

    # False Alarms
    condition1 = (obs <= categorical_threshold) & (categorical_threshold < fcst)
    # Misses
    condition2 = (fcst <= categorical_threshold) & (categorical_threshold < obs)

    # Bring back NaNs
    condition1 = condition1.where(~np.isnan(fcst))
    condition1 = condition1.where(~np.isnan(obs))
    condition2 = condition2.where(~np.isnan(fcst))
    condition2 = condition2.where(~np.isnan(obs))

    if discount_distance:
        scale_1 = np.minimum(categorical_threshold - obs, discount_distance)
        scale_2 = np.minimum(obs - categorical_threshold, discount_distance)
    else:
        scale_1 = 1
        scale_2 = 1

    overforecast_penalty = (1 - risk_parameter) * scale_1 * condition1
    underforecast_penalty = risk_parameter * scale_2 * condition2
    firm_score = overforecast_penalty + underforecast_penalty

    score = xr.Dataset(
        {
            "firm_score": firm_score,
            "overforecast_penalty": overforecast_penalty,
            "underforecast_penalty": underforecast_penalty,
        }
    )
    score = score.transpose(*fcst.dims)
    return score
