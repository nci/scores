"""
This module contains methods which may be used for scoring multicategorical forecasts
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import xarray as xr

from scores.functions import apply_weights
from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_dims, gather_dimensions


def firm(  # pylint: disable=too-many-arguments
    fcst: xr.DataArray,
    obs: xr.DataArray,
    risk_parameter: float,
    categorical_thresholds: Union[Sequence[float], Sequence[xr.DataArray]],
    threshold_weights: Sequence[Union[float, xr.DataArray]],
    *,  # Force keywords arguments to be keyword-only
    discount_distance: Optional[float] = 0,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    threshold_assignment: Optional[str] = "lower",
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
            categories. A sequence of xr.DataArrays may be supplied to allow
            for different thresholds at each coordinate (e.g., thresholds
            determined by climatology).
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
        threshold_assignment: Specifies whether the intervals defining the categories are
            left or right closed. That is whether the decision threshold is included in
            the upper (left closed) or lower (right closed) category. Defaults to "lower".

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
        obs, risk_parameter, categorical_thresholds, threshold_weights, discount_distance, threshold_assignment
    )
    total_score = []
    for categorical_threshold, weight in zip(categorical_thresholds, threshold_weights):
        score = weight * _single_category_score(
            fcst,
            obs,
            risk_parameter,
            categorical_threshold,  # type: ignore
            discount_distance=discount_distance,
            threshold_assignment=threshold_assignment,
        )
        total_score.append(score)
    summed_score = sum(total_score)
    reduce_dims = gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )  # type: ignore[assignment]
    summed_score = apply_weights(summed_score, weights=weights)  # type: ignore
    score = summed_score.mean(dim=reduce_dims)  # type: ignore

    return score  # type: ignore


def _check_firm_inputs(
    obs, risk_parameter, categorical_thresholds, threshold_weights, discount_distance, threshold_assignment
):  # pylint: disable=too-many-positional-arguments
    """
    Checks that the FIRM inputs are suitable
    """
    if len(categorical_thresholds) < 1:
        raise ValueError("`categorical_thresholds` must have at least one threshold")

    if not len(categorical_thresholds) == len(threshold_weights):
        raise ValueError("The length of `categorical_thresholds` and `weights` must be equal")
    if risk_parameter <= 0 or risk_parameter >= 1:
        raise ValueError("0 < `risk_parameter` < 1 must be satisfied")

    for count, weight in enumerate(threshold_weights):
        if isinstance(weight, xr.DataArray):
            check_dims(weight, obs.dims, mode="subset")
            if np.any(weight <= 0):
                raise ValueError(
                    f"""
                    No values <= 0 are allowed in `weights`. At least one
                    negative value was found in index {count} of `weights`
                    """
                )
        elif weight <= 0:
            raise ValueError("All values in `weights` must be > 0")

    if discount_distance < 0:
        raise ValueError("`discount_distance` must be >= 0")

    if threshold_assignment not in ["upper", "lower"]:
        raise ValueError(""" `threshold_assignment` must be either \"upper\" or \"lower\" """)


def _single_category_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    risk_parameter: float,
    categorical_threshold: Union[float, xr.DataArray],
    *,  # Force keywords arguments to be keyword-only
    discount_distance: Optional[float] = None,
    threshold_assignment: Optional[str] = "lower",
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
        threshold_assignment: Specifies whether the intervals defining the categories are
            left or right closed. That is whether the decision threshold is included in
            the upper (left closed) or lower (right closed) category. Defaults to "lower".

    Returns:
        An xarray Dataset with data vars:

            * firm_score: a score for a single category for each coord
              based on the FIRM framework. All dimensions are preserved.
            * overforecast_penalty: Penalty for False Alarms.
            * underforecast_penalty: Penalty for Misses.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    fcst, obs = xr.align(fcst, obs)

    if threshold_assignment == "lower":
        # False Alarms
        condition1 = (obs <= categorical_threshold) & (categorical_threshold < fcst)
        # Misses
        condition2 = (fcst <= categorical_threshold) & (categorical_threshold < obs)
    else:
        # False Alarms
        condition1 = (obs < categorical_threshold) & (categorical_threshold <= fcst)
        # Misses
        condition2 = (fcst < categorical_threshold) & (categorical_threshold <= obs)

    # Bring back NaNs
    condition1 = condition1.where(~np.isnan(fcst))
    condition1 = condition1.where(~np.isnan(obs))
    condition1 = condition1.where(~np.isnan(categorical_threshold))
    condition2 = condition2.where(~np.isnan(fcst))
    condition2 = condition2.where(~np.isnan(obs))
    condition2 = condition2.where(~np.isnan(categorical_threshold))

    if discount_distance:
        scale_1 = np.minimum(categorical_threshold - obs, discount_distance)
        scale_2 = np.minimum(obs - categorical_threshold, discount_distance)
    else:
        scale_1 = 1  # type: ignore
        scale_2 = 1  # type: ignore

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


def seeps(
    fcst: XarrayLike,
    obs: XarrayLike,
    dry_threshold: xr.DataArray,
    heavy_precipitation_threshold: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    mask_clim_extremes: bool = True,
    min_masked_value: float = 0.1,
    max_masked_value: float = 0.85,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> XarrayLike:
    """
    Calculates the stable equitable error in probability space (SEEPS) score.

    When used to evaluate precipitation forecasts, the SEEPS score calculates 
    the performance of a forecast across three categories:
    dry weather, light precipitation, and heavy precipitation. The boundary between
    the light and heavy category is based on climatological data.

    The SEEPS penalty matrix is defined as

    .. math::
        \\{s\\} = \frac{1}{2} \\left(
        \\begin{matrix}
        0 & \\frac{1}{1-p_1} & \\frac{1}{p_3}+\\frac{1}{1-p_1} \\\\
        \\frac{1}{p_1} & 0 & \\frac{1}{p_3} \\\\
        \\frac{1}{p_1}+\\frac{1}{1-p_3} & \\frac{1}{1-p_3} & 0
        \\end{matrix}
        \\right)

        
    where 
        - :math:`p_1` is the climatological probability of the dry weather category
        - :math:`p_3` is the climatological probability of the heavy precipitation category.
        - The rows correspond to the forecast category (dry, light, heavy).
        - The columns correspond to the observation category (dry, light, heavy).

    Note that although :math:`p_2`, does not appear in the penalty matrix, it is defined as
    :math:`p_2 = 2p_3` which means that the light precipitation category is twice as likely
    to occur climatologically as the heavy precipitation category.

    This score in this implementation is negatively oriented, meaning that lower scores are better. 
    Sometimes in the literature, a SEEPS skill score is used, which is defined as 1 - SEEPS.

    Args:
        fcst: An array of real-valued forecasts.
        obs: An array of real-valued observations.
        dry_threshold: The climatological probability of the dry weather category.
            This is called p1 in the penalty matrix.
        heavy_precipitation_threshold: The climatological probability of the heavy
            precipitation category. This is called p3 in the penalty matrix
        mask_clim_extremes: If True, mask out the climatological extremes.
        min_masked_value: Points with climatolgical probabilities of dry weather
            less than this value are masked. Defaults to 0.1.
        max_masked_value: Points with climatolgical probabilities of dry weather
            greater than this value are masked. Defaults to 0.85.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the SEEPS score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating SEEPS. All other dimensions will be reduced.
            As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the SEEPS score at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely. Only one of `reduce_dims` and `preserve_dims` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
    
    References:
        Rodwell, M. J., Richardson, D. S., Hewson, T. D., & Haiden, T. (2010). 
        A new equitable score suitable for verifying precipitation in numerical 
        weather prediction. Quarterly Journal of the Royal Meteorological Society, 
        136(650), 1344–1363. Portico. https://doi.org/10.1002/qj.656

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.categorical import seeps
        >>> fcst = xr.DataArray(np.random.rand(4, 6, 8), dims=['time', 'lat', 'lon'])
        >>> obs = xr.DataArray(np.random.rand(4, 6, 8), dims=['time', 'lat', 'lon'])
        >>> dry_threshold = xr.DataArray(np.random.rand(3, 4), dims=['lat', 'lon'])
        >>> heavy_precipitation_threshold = 0.2 * dry_threshold 
        >>> seeps(fcst, obs, dry_threshold, heavy_precipitation_threshold)
    """
    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    fcst, obs = broadcast_and_match_nan(fcst, obs)

    # Penalties for index i, j in the penalty matrix. Row i corresponds to the
    # forecast category while row j corresponds to the observation category
    # row 1 of the penalty matrix
    index_12 = 1 / (1 - dry_threshold)
    index_13 = (1 / heavy_precipitation_threshold) + (1 / (1 - dry_threshold))
    # row 2 of the penalty matrix
    index_21 = 1 / dry_threshold
    index_23 = 1 / heavy_precipitation_threshold
    # row 3 of the penalty matrix
    index_31 = (1 / dry_threshold) + (1 / (1 - heavy_precipitation_threshold))
    index_32 = 1 / (1 - heavy_precipitation_threshold)

    # Get conditions for each category
    fcst1_condition = fcst < dry_threshold
    fcst2_condition = (fcst >= dry_threshold) & (fcst < heavy_precipitation_threshold)
    fcst3_condition = fcst >= heavy_precipitation_threshold

    obs1_condition = obs < dry_threshold
    obs2_condition = (obs >= dry_threshold) & (obs < heavy_precipitation_threshold)
    obs3_condition = obs >= heavy_precipitation_threshold

    # Calculate the penalties
    result = fcst.copy() * 0
    result = result.where(~(fcst1_condition & obs2_condition), index_12)
    result = result.where(~(fcst1_condition & obs3_condition), index_13)
    result = result.where(~(fcst2_condition & obs1_condition), index_21)
    result = result.where(~(fcst2_condition & obs3_condition), index_23)
    result = result.where(~(fcst3_condition & obs1_condition), index_31)
    result = result.where(~(fcst3_condition & obs2_condition), index_32)

    result = result / 2
    # return NaNs
    result = result.where(
        ~np.isnan(fcst) & ~np.isnan(obs) & ~np.isnan(dry_threshold) & ~np.isnan(heavy_precipitation_threshold)
    )

    if mask_clim_extremes:
        result = result.where(np.logical_and(dry_threshold <= max_masked_value, dry_threshold >= min_masked_value))

    result = apply_weights(result, weights=weights)
    result = result.mean(dim=reduce_dims)

    return result
