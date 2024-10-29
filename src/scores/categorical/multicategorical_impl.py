"""
This module contains methods which may be used for scoring multicategorical forecasts
"""

from collections.abc import Sequence
from typing import Optional, Union, Iterable

import numpy as np
import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes
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


def risk_matrix_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    decision_weights: xr.DataArray,
    severity_dim: str,
    prob_threshold_dim: str,
    threshold_assignment: Optional[str] = "lower",
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Calculates the risk matrix score MS of Taggart and Wilke (2024).

    Args:
        fcst: an array of forecast probabilities for the observation lying in each severity
            category. Must have a dimension `severity_dim`.
        obs: an array of binary observations with a value of 1 if the observation was in the
            severity category and 0 otherwise. Must have a dimension `severity_dim`.
        decision_weights: an array of non-negative weights to apply to each matrix decision
            threshold, indexed by coordinates in `severity_dim` and `prob_threshold_dim`.
        severity_dim: the dimension specifying severity cateogories.
        prob_threshold_dim: the dimension in `decision_weights` specifying probability thresholds.
        threshold_assignment: Either "upper" or "lower". Specifies whether the probability
            intervals defining the certainty categories, with endpoints given by the
            decision thresholds in `decision_weights`, are left or right closed. That is,
            whether the probability decision threshold is included in the upper (left closed)
            or lower (right closed) certainty category. Defaults to "lower".
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            mean risk matrix score, where the mean is taken over all forecast cases.
            All other dimensions will be preserved. As a special case, 'all' will allow
            all dimensions to be reduced. Only one of `reduce_dims` and `preserve_dims`
            can be supplied. The default behaviour if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating the
            mean risk matrix score, where the mean is taken over all forecast cases.
            All other dimensions will be reduced. As a special case, 'all' will allow
            all dimensions to be preserved, apart from `severity_dim` and `prob_threshold_dim`.
            Only one of `reduce_dims` and `preserve_dims` can be supplied. The default
            behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom) of scores across all forecast cases.

    Returns:
        An xarray data array of risk matrix scores, averaged according to specified weights
        across appropriate dimensions.

    Raises:
        ValueError: if `severity_dim` is not a dimension of `fcst`, `obs` or `decision_weights`.
        ValueError: if `severity_dim` is a dimension of `weights`.
        ValueError: if `prob_threshold_dim` is not a dimension of `decision_weights`.
        ValueError: if `prob_threshold_dim` is not a dimension of `decision_weights`.
        ValueError: if `decision_weights` does not have exactly 2 dimenions.
        ValueError: if `fcst` values are negative or greater than 1.
        ValueError: if `obs` values are other than 0, 1 or nan.
        ValueError: if `prob_threshold_dim` coordinates are not strictly between 0 and 1.
        ValueError: if `severity_dim` coordinates differ for any of `fcst`, `obs` or `decision_weights`.
        ValueError: if `threshold_assignment` is not "upper" or lower".

    """
    _check_risk_matrix_score_inputs(
        fcst, obs, decision_weights, severity_dim, prob_threshold_dim, threshold_assignment, weights
    )
    # get fcst and obs dims without `severity_dim`
    # this is because `severity_dim` is reduced when scoring individual forecast cases
    fcst_dims0 = [x for x in fcst.dims if x is not severity_dim]
    obs_dims0 = [x for x in obs.dims if x is not severity_dim]
    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims

    reduce_dims = gather_dimensions(
        fcst_dims0, obs_dims0, weights_dims=weights_dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    result = _risk_matrix_score(fcst, obs, decision_weights, severity_dim, prob_threshold_dim, threshold_assignment)

    result = apply_weights(result, weights=weights).mean(dim=reduce_dims)

    return result


def _check_risk_matrix_score_inputs(
    fcst, obs, decision_weights, severity_dim, prob_threshold_dim, threshold_assignment, weights
):
    """
    Checks that the risk matrix score inputs are suitable.
    """
    if severity_dim not in fcst.dims:
        raise ValueError("`severity_dim` must be a dimension of `fcst`")
    if severity_dim not in obs.dims:
        raise ValueError("`severity_dim` must be a dimension of `obs`")
    if severity_dim not in decision_weights.dims:
        raise ValueError("`severity_dim` must be a dimension of `decision_weights`")
    if weights is not None and severity_dim in weights.dims:
        raise ValueError("`severity_dim` must not be a dimension of `weights`")
    if prob_threshold_dim in fcst.dims:
        raise ValueError("`prob_threshold_dim` must not be a dimension of `fcst`")
    if prob_threshold_dim in obs.dims:
        raise ValueError("`prob_threshold_dim` must not be a dimension of `obs`")
    if prob_threshold_dim not in decision_weights.dims:
        raise ValueError("`prob_threshold_dim` must be a dimension of `decision_weights`")
    if weights is not None and prob_threshold_dim in weights.dims:
        raise ValueError("`prob_threshold_dim` must not be a dimension of `weights`")
    if len(decision_weights.dims) != 2:
        raise ValueError("`decision_weights` must have exactly 2 dimensions: `severity_dim` and `prob_threshold_dim`")

    if (fcst.max() > 1) or (fcst.min() < 0):
        raise ValueError("values in `fcst` must lie in the closed interval `[0, 1]`")

    disallowed_obs_values = np.unique(obs.where(obs != 1).where(obs != 0).values)
    disallowed_obs_values = disallowed_obs_values[~np.isnan(disallowed_obs_values)]
    if len(disallowed_obs_values) > 0:
        raise ValueError("values in `obs` can only be 0, 1 or nan")

    if (decision_weights[prob_threshold_dim].min() <= 0) or (decision_weights[prob_threshold_dim].max() >= 1):
        raise ValueError("`prob_threshold_dim` coordinates must be strictly between 0 and 1")

    if not np.array_equal(np.sort(decision_weights[severity_dim].values), np.sort(fcst[severity_dim].values)):
        raise ValueError("`severity_dim` coordinates do not match in `decision_weights` and `fcst`")
    if not np.array_equal(np.sort(decision_weights[severity_dim].values), np.sort(obs[severity_dim].values)):
        raise ValueError("`severity_dim` coordinates do not match in `decision_weights` and `obs`")

    if threshold_assignment not in ["upper", "lower"]:
        raise ValueError(""" `threshold_assignment` must be either \"upper\" or \"lower\" """)


def _risk_matrix_score(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    decision_weights: xr.DataArray,
    severity_dim: str,
    prob_threshold_dim: str,
    threshold_assignment: Optional[str] = "lower",
) -> xr.DataArray:
    """
    Calculates the risk matrix score MS of Taggart and Wilke (2024).

    Args:
        fcst: an array of forecast probabilities for the observation lying in each severity
            category. Must have a dimension `severity_dim`.
        obs: an array of binary observations with a value of 1 if the observation was in the
            severity category and 0 otherwise. Must have a dimension `severity_dim`.
        decision_weights: an array of non=negative weights to apply to each matrix decision
            threshold, indexed by coordinates in `severity_dim` and `prob_threshold_dim`.
        threshold_assignment: Either "upper" or "loewer". Specifies whether the probability
            intervals defining the certainty categories for the decision thresholds in
            `decision_weights` are left or right closed. That is, whether the probability
            decision threshold is included in the upper (left closed) or lower (right closed)
            certainty category. Defaults to "lower".

    Returns:
        xarray data array of matrix scores for each forecast case, preserving all dimensions
        in fcst, obs apart from `severity_dim`.
    """
    da_thresholds = decision_weights[prob_threshold_dim]

    mask = (~np.isnan(fcst)) & (~np.isnan(obs))

    if threshold_assignment == "lower":
        fcst_abv_threshold = fcst >= da_thresholds
    else:
        fcst_abv_threshold = fcst > da_thresholds

    # penalties for over-forecasts
    over = da_thresholds.where((obs == 0) & fcst_abv_threshold, 0).where(mask)
    # penalties for under-forecasts
    under = (1 - da_thresholds).where((obs == 1) & ~fcst_abv_threshold, 0).where(mask)

    result = (over + under) * decision_weights
    result = result.sum([prob_threshold_dim, severity_dim], skipna=False)

    return result


def matrix_weights_to_array(
    matrix_weights: np.ndarray,
    severity_dim: str,
    severity_coords: Iterable,
    prob_threshold_dim: str,
    prob_threshold_coords: Iterable,
) -> xr.DataArray:
    """
    Generates a 2-dimensional xr.DataArray of the decision thresholds for the risk matrix
    score calculation.  Assumes that values toward the left in `matrix_weights` correspond
    to less severe categories, while values towards the top in `matrix_weights` correspond
    to higher probability thresholds.

    Args:
        matrix_weights: array of weights to place on each risk matrix decision threshold,
            with rows (ascending) corresponding to (increasing) probability thresholds,
            and colummns (left to right) corresponding to (increasing) severity categories.
        severity_dim: name of the severity category dimension.
        severity_coords: labels for each of the severity categories, in order of increasing
            severity.
        prob_threshold_dim: name of the probability threshold dimension.
        prob_threshold_coords: list of the probability decision thresholds in the risk matrix,
            strictly between 0 and 1.

    Returns:
        xarray data array of risk matrix decision threshold weights, indexed by prob_threshold_dim
        and severity_dim.

    Raises:
        ValueError: if `matrix_weights` isn't two dimensional.
        ValueError: if number of rows of `matrix_weights` doesn't equal length of `prob_threshold_coords`.
        ValueError: if number of columns of `matrix_weights` doesn't equal length of `severity_coords`.
        ValueError: if `prob_threshold_coords` aren't strictly between 0 and 1.
    """
    matrix_dims = matrix_weights.shape

    if len(matrix_dims) != 2:
        raise ValueError("`matrix_weights` must be two dimensional")
    if matrix_dims[0] != len(prob_threshold_coords):
        raise ValueError("number of `prob_threshold_coords` must equal number of rows of `matrix_weights`")
    if matrix_dims[1] != len(severity_coords):
        raise ValueError("number of `severity_coords` must equal number of columns of `matrix_weights`")

    if (np.max(prob_threshold_coords) >= 1) or (np.min(prob_threshold_coords) <= 0):
        raise ValueError("`prob_threshold_coords` must strictly between 0 and 1")

    prob_threshold_coords = np.flip(np.sort(np.array(prob_threshold_coords)))

    weight_matrix = xr.DataArray(
        data=matrix_weights,
        dims=[prob_threshold_dim, severity_dim],
        coords={
            prob_threshold_dim: prob_threshold_coords,
            severity_dim: severity_coords,
        },
    )

    return weight_matrix


def scaling_to_weight_array(
    scaling_matrix: np.ndarray,
    assessment_weights: Iterable,
    severity_dim: str,
    severity_coords: Iterable,
    prob_threshold_dim: str,
    prob_threshold_coords: Iterable,
) -> xr.DataArray:
    """
    Given warning scaling matrix, assessment weights and other inputs,
    returns the decision weights for the risk matrix score as an xarray data array.

    Comprehensive checks are made on `scaling_matrix` to ensure it satisfies the properties
    of warning scaling in Table 1 of Taggart & Wilke (2004).

    Args:
        scaling_matrix: a 2-dimensional matrix encoding the warning scaling. Warning levels
            are given integer values 0, 1, ..., q. The top row corresponds to the
            highest certainty category while the right-most column corresponds to most
            severe category.
        assessment_weights: positive weights used for warning assessment. The kth weight
            (corresponding to `assessment_weights[k-1]`) is proportional to the importance
            of accuractely discriminating between warning states below level k and
            warning states at or above level k. `len(assessment_weights)` must be at least q.
        severity_dim: name of the severity category dimension.
        severity_coords: labels for each of the severity categories, in order of increasing
            severity.
        prob_threshold_dim: name of the probability threshold dimension.
        prob_threshold_coords: list of the probability decision thresholds in the risk matrix,
            strictly between 0 and 1.

    Returns:
        xarray data array of risk matrix decision threshold weights, indexed by prob_threshold_dim
        and severity_dim.

    Raises:
        ValueError: if `matrix_weights` isn't two dimensional.
        ValueError: if `scaling_matrix` has entries that are not non-negative integers.
        ValueError: if the first column or last row of `scaling_matrix` has nonzero entries.
        ValueError: if `scaling_matrix` decreases along any row (moving left to right).
        ValueError: if `scaling_matrix` increases along any column (moving top to bottom).
        ValueError: if number of rows of `matrix_weights` doesn't equal length of `prob_threshold_coords`.
        ValueError: if number of columns of `matrix_weights` doesn't equal length of `severity_coords`.
        ValueError: `len(assessment_weights)` is less than the maximum value in `scaling_matrix`.
        ValueError: if `prob_threshold_coords` aren't strictly between 0 and 1.
        ValueError: if `assessment_weights` aren't strictly positive.
    """
    scaling_matrix_shape = scaling_matrix.shape

    # checks on the scaling matrix
    if len(scaling_matrix_shape) != 2:
        raise ValueError("`scaling_matrix` should be two dimensional")
    if not np.issubdtype(scaling_matrix.dtype, np.integer):
        raise ValueError("`scaling_matrix` should only have have integer entries")
    if np.min(scaling_matrix) < 0:
        raise ValueError("`scaling_matrix` should only have non-negative integer values")
    if not (np.unique(scaling_matrix[:, 0]) == np.array([0])).all():
        raise ValueError("The first column of `scaling_matrix` should consist of zeros only")
    if not (np.unique(scaling_matrix[-1, :]) == np.array([0])).all():
        raise ValueError("The last row of `scaling_matrix` should consist of zeros only")
    if np.min(np.diff(scaling_matrix, axis=1)) < 0:
        raise ValueError("`scaling_matrix` should be non-decreasing along each row (moving left to right)")
    if np.max(np.diff(scaling_matrix, axis=0)) > 0:
        raise ValueError("`scaling_matrix` should be non-increasing along each column (moving top to bottom)")

    # checks on compatibility between scaling matrix and other inputs
    if scaling_matrix_shape[0] - 1 != len(prob_threshold_coords):
        raise ValueError("Length of `prob_threshold_coords` should be one less than rows of `scaling_matrix`")
    if scaling_matrix_shape[1] - 1 != len(severity_coords):
        raise ValueError("Length of `severity_coords` should be one less than columns of `scaling_matrix`")
    if len(assessment_weights) < np.max(scaling_matrix):
        raise ValueError("length of `assessment_weights` must be at least the highest value in `scaling_matrix`")

    # check on other inputs
    if (np.max(prob_threshold_coords) >= 1) or (np.min(prob_threshold_coords) <= 0):
        raise ValueError("`prob_threshold_coords` must strictly between 0 and 1")
    if np.min(assessment_weights) <= 0:
        raise ValueError("values in `assessment_weights` must be positive")

    weight_matrix = _scaling_to_weight_matrix(scaling_matrix, assessment_weights)
    weight_array = matrix_weights_to_array(
        weight_matrix, severity_dim, severity_coords, prob_threshold_dim, prob_threshold_coords
    )
    return weight_array


def _scaling_to_weight_matrix(scaling_matrix, assessment_weights):
    """
    Given a scaling matrix and assessment weights, outputs the weight matrix for the
    decision thresholds of the corresponding risk matrix.

    This is an implementation of the algorithm of Appendix B, Taggart & Wilke (2024).

    Args:
        scaling_matrix: np.array of warning scaling values. Values must be integers.
        assessment_weights: list of weights for each warning level decision threshold.
    Returns:
        np.array of weights
    """
    max_level = max(np.max(scaling_matrix), len(assessment_weights))

    scaling_matrix_shape = scaling_matrix.shape
    n_sev = scaling_matrix_shape[1] - 1  # number of severity categories for weight matrix
    n_prob = scaling_matrix_shape[0] - 1  # number of probability thresholds for weight matrix

    # initialise the weight matrix wts
    wts = np.zeros((n_prob, n_sev))

    for level in np.arange(1, max_level + 1):
        lowest_prob_index = max_level + 1

        for column in np.arange(1, n_sev + 1):  # column of the scaling matrix
            the_column = np.array(scaling_matrix[:, column])
            column_rev = np.flip(the_column)

            # get the position (i.e. probability threshold index) of level crossover for the column
            # a position of 0 means that there is no crossover for that level
            prob_index = np.argmax(column_rev >= level)
            # only modify weight matrix if
            if prob_index >= lowest_prob_index:
                prob_index = 0
            if prob_index > 0:
                wts[prob_index - 1, column - 1] = wts[prob_index - 1, column - 1] + assessment_weights[level - 1]
                lowest_prob_index = prob_index

    wts = np.flip(wts, axis=0)

    return wts
