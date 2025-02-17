"""
This module contains methods which for emerging scores.
"""

from typing import Iterable, Optional

import numpy as np
import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions


def risk_matrix_score(
    fcst: XarrayLike,
    obs: XarrayLike,
    decision_weights: xr.DataArray,
    severity_dim: str,
    prob_threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    threshold_assignment: Optional[str] = "lower",
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> XarrayLike:
    """
    Caution:
        This is an implementation of a novel metric that is still undergoing mathematical peer review. 
        This implementation may change in line with the peer review process.

    Calculates the risk matrix score of Taggart & Wilke (2025). 
    
    Let :math:`(S_1, \\ldots,S_m)` denote the tuple of nested severity categories,
    let :math:`(p_1, \\ldots,p_n)`  denote the probability thresholds that delineate the
    certainty categories, and let :math:`w_{i,j}` denote the weight applied to the
    decision point associated with severity category :math:`S_i` and probability threshold :math:`p_j`.

    In this implementation of the risk matrix score, a forecast :math:`F` is of the form
    :math:`F=(f_1,\\ldots,f_m)` where :math:`f_i` denotes the forecast probability that
    the observation lies in severity category :math:`S_i`.
    A corresponding observation :math:`y` is given by :math:`y=(y_1,\\ldots,y_m)` where
    :math:`y_i` is 1 if the observation lies in severity category :math:`S_i` and 0 otherwise. 
    Then the risk matrix score :math:`\\text{RMS}` is given by the formula

    .. math::
        \\text{RMS}(F,y) = \\sum_{i=1}^m\\sum_{j=1}^n w_{i,j} \\, s_j(f_i, y_i),

    where

    .. math::
        s_j(f_i,y_i) = \\begin{cases}
            p_j & \\text{if } y_i=0\\text{ and } f_i \\geq p_j \\\\
            1 - p_j & \\text{if } y_i=1\\text{ and } f_i < p_j \\\\
            0 & \\text{otherwise.}
        \\end{cases}

    The formula above is for the case where the ``threshold_assignment`` is "lower", with 
    the adjustments (:math:`f_i > p_j` and :math:`f_i \\leq p_j`) applied when the
    ``threshold_assignment`` is "upper".

    Args:
        fcst: an array of forecast probabilities for the observation lying in each severity
            category. Must have a dimension ``severity_dim``.
        obs: an array of binary observations with a value of 1 if the observation was in the
            severity category and 0 otherwise. Must have a dimension ``severity_dim``.
        decision_weights: an array of non-negative weights to apply to each risk matrix decision
            point, indexed by coordinates in ``severity_dim`` and ``prob_threshold_dim``.
        severity_dim: the dimension specifying severity categories.
        prob_threshold_dim: the dimension in ``decision_weights`` specifying probability thresholds.
        threshold_assignment: Either "upper" or "lower". Specifies whether the probability
            intervals defining the certainty categories, with endpoints given by the
            decision thresholds in ``decision_weights``, are left or right closed. That is,
            whether the probability decision threshold is included in the upper (left closed)
            or lower (right closed) certainty category. Defaults to "lower".
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            mean risk matrix score, where the mean is taken over all forecast cases.
            All other dimensions will be preserved. As a special case, 'all' will allow
            all dimensions to be reduced. Only one of ``reduce_dims`` and ``preserve_dims``
            can be supplied. The default behaviour if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating the
            mean risk matrix score, where the mean is taken over all forecast cases.
            All other dimensions will be reduced. As a special case, 'all' will allow
            all dimensions to be preserved, apart from ``severity_dim`` and ``prob_threshold_dim``.
            Only one of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default
            behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom) of scores across all forecast cases.

    Returns:
        An xarray object of risk matrix scores, averaged according to specified weights
        across appropriate dimensions.

    Raises:
        ValueError: if ``severity_dim`` is not a dimension of ``fcst``, ``obs`` or ``decision_weights``.
        ValueError: if ``severity_dim`` is a dimension of ``weights``.
        ValueError: if ``prob_threshold_dim`` is not a dimension of ``decision_weights``.
        ValueError: if ``prob_threshold_dim`` is a dimension of ``weights``.
        ValueError: if ``decision_weights`` does not have exactly 2 dimensions.
        ValueError: if ``fcst`` values are negative or greater than 1.
        ValueError: if ``obs`` values are other than 0, 1 or nan.
        ValueError: if ``prob_threshold_dim`` coordinates are not strictly between 0 and 1.
        ValueError: if ``severity_dim`` coordinates differ for any of ``fcst``, ``obs`` or ``decision_weights``.
        ValueError: if ``threshold_assignment`` is not "upper" or lower".

    References:
        - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework
          with consistent evaluation. arXiv. https://doi.org/10.48550/arXiv.2502.08891

    See also:
        :py:func:`scores.emerging.matrix_weights_to_array`
        :py:func:`scores.emerging.weights_from_warning_scaling`

    Examples:
        Calculate the risk matrix score where the risk matrix has three nested severity
        categories ("MOD+", "SEV+" and "EXT") and three probability thresholds (0.1, 0.3 and 0.5).
        The decision weights place greater emphasis on higher end severity.

        >>> import xarray as xr
        >>> from scores.emerging import risk_matrix_score
        >>> decision_weights = xr.DataArray(
        >>>     data=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        >>>     dims=["probability_threshold", "severity"],
        >>>     coords={'probability_threshold': [0.1, 0.3, 0.5], 'severity': ["MOD+", "SEV+", "EXT"]}
        >>> )
        >>> fcst = xr.DataArray(
        >>>     data=[[0.45, 0.22, 0.05], [0.65, 0.32, 0.09]],
        >>>     dims=["time", "severity"],
        >>>     coords={'time': [0, 1], 'severity': ["MOD+", "SEV+", "EXT"]}
        >>> )
        >>> obs = xr.DataArray(
        >>>     data=[[1, 1, 0], [1, 0, 0]],
        >>>     dims=["time", "severity"],
        >>>     coords={'time': [0, 1], 'severity': ["MOD+", "SEV+", "EXT"]}
        >>> )
        >>> risk_matrix_score(fcst, obs, decision_weights, "severity", "probability_threshold")
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
    result = _risk_matrix_score(
        fcst, obs, decision_weights, severity_dim, prob_threshold_dim, threshold_assignment=threshold_assignment
    )

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

    if isinstance(fcst, xr.Dataset):
        fcst = fcst.to_array()

    if (fcst.max() > 1) or (fcst.min() < 0):
        raise ValueError("values in `fcst` must lie in the closed interval `[0, 1]`")

    if isinstance(obs, xr.Dataset):
        obs = obs.to_array()

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
    fcst: XarrayLike,
    obs: XarrayLike,
    decision_weights: xr.DataArray,
    severity_dim: str,
    prob_threshold_dim: str,
    *,
    threshold_assignment: Optional[str] = "lower",
) -> XarrayLike:
    """
    Calculates the risk matrix score (MS) of Taggart and Wilke (2025).

    Args:
        fcst: an array of forecast probabilities for the observation lying in each severity
            category. Must have a dimension `severity_dim`.
        obs: an array of binary observations with a value of 1 if the observation was in the
            severity category and 0 otherwise. Must have a dimension `severity_dim`.
        decision_weights: an array of non-negative weights to apply to each decision
            point in the risk matrix, indexed by coordinates in `severity_dim` and `prob_threshold_dim`.
        threshold_assignment: Either "upper" or "lower". Specifies whether the probability
            intervals defining the certainty categories for the decision thresholds in
            `decision_weights` are left or right closed. That is, whether the probability
            decision threshold is included in the upper (left closed) or lower (right closed)
            certainty category. Defaults to "lower".

    Returns:
        xarray object of matrix scores for each forecast case, preserving all dimensions
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
    Caution:
        This function is part of an implementation of a novel metric that is still undergoing
        mathematical peer review. This implementation may change in line with the peer review process.

    Generates a 2-dimensional xr.DataArray of weights for each decision point, which is used for
    the :py:func:`scores.emerging.risk_matrix_score` calculation.
    Assumes that values toward the left in ``matrix_weights`` correspond
    to less severe categories, while values towards the top in ``matrix_weights`` correspond
    to higher probability thresholds.

    Args:
        matrix_weights: array of weights to place on each risk matrix decision point,
            with rows (ascending) corresponding to (increasing) probability thresholds,
            and columns (left to right) corresponding to (increasing) severity categories.
        severity_dim: name of the severity category dimension.
        severity_coords: labels for each of the severity categories, in order of increasing
            severity. Does NOT include the lowest severity category for which no warning would
            be issued.
        prob_threshold_dim: name of the probability threshold dimension.
        prob_threshold_coords: list of the probability decision thresholds in the risk matrix,
            strictly between 0 and 1.

    Returns:
        xarray data array of weights for each risk matrix decision point, indexed by prob_threshold_dim
        and severity_dim.

    Raises:
        ValueError: if ``matrix_weights`` isn't two dimensional.
        ValueError: if number of rows of ``matrix_weights`` doesn't equal length of ``prob_threshold_coords``.
        ValueError: if number of columns of ``matrix_weights`` doesn't equal length of ``severity_coords``.
        ValueError: if ``prob_threshold_coords`` aren't strictly between 0 and 1.

    References:
        - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework
          with consistent evaluation. arXiv. https://doi.org/10.48550/arXiv.2502.08891

    Examples:
        Returns weights for each risk matrix decision point, where weights increase with increasing
        severity category and decrease with increasing probability threshold.

        >>> import numpy as np
        >>> from scores.emerging import matrix_weights_to_array
        >>> matrix_weights = np.array([
        >>>     [1, 2, 3],
        >>>     [2, 4, 6],
        >>>     [3, 6, 9],
        >>> ])
        >>> matrix_weights_to_array(
        >>>     matrix_weights, "severity", ["MOD+", "SEV+", "EXT"], "prob_threshold", [0.1, 0.3, 0.5]
        >>> )
    """
    matrix_dims = matrix_weights.shape

    if len(matrix_dims) != 2:
        raise ValueError("`matrix_weights` must be two dimensional")
    if matrix_dims[0] != len(prob_threshold_coords):  # type: ignore
        raise ValueError("number of `prob_threshold_coords` must equal number of rows of `matrix_weights`")
    if matrix_dims[1] != len(severity_coords):  # type: ignore
        raise ValueError("number of `severity_coords` must equal number of columns of `matrix_weights`")

    if (np.max(prob_threshold_coords) >= 1) or (np.min(prob_threshold_coords) <= 0):  # type: ignore
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


def weights_from_warning_scaling(
    scaling_matrix: np.ndarray,
    evaluation_weights: Iterable,
    severity_dim: str,
    severity_coords: Iterable,
    prob_threshold_dim: str,
    prob_threshold_coords: Iterable,
) -> xr.DataArray:
    """
    Caution:
        This function is part of an implementation of a novel metric that is still undergoing
        mathematical peer review. This implementation may change in line with the peer review process.

    Given a warning scaling matrix, evaluation weights and other inputs,
    returns the weights for each risk matrix decision point as an xarray data array. The returned
    data array is designed to be used for the :py:func:`scores.emerging.risk_matrix_score` calculation.

    Comprehensive checks are made on ``scaling_matrix`` to ensure it satisfies the properties
    of warning scaling in Table 1 of Taggart & Wilke (2025).

    Args:
        scaling_matrix: a 2-dimensional matrix encoding the warning scaling. Warning levels
            are given integer values 0, 1, ..., q. The top row corresponds to the
            highest certainty category while the right-most column corresponds to most
            severe category.
        evaluation_weights: positive weights used for warning evaluation. The kth weight
            (corresponding to ``evaluation_weights[k-1]``) is proportional to the importance
            of accurately discriminating between warning states below level k and
            warning states at or above level k. ``len(evaluation_weights)`` must be at least q.
        severity_dim: name of the severity category dimension.
        severity_coords: labels for each of the severity categories, in order of increasing
            severity. Does NOT include the lowest severity category for which no warning would
            be issued.
        prob_threshold_dim: name of the probability threshold dimension.
        prob_threshold_coords: list of the probability decision thresholds in the risk matrix,
            strictly between 0 and 1.

    Returns:
        xarray data array of weights for each risk matrix decision point, indexed by prob_threshold_dim
        and severity_dim.

    Raises:
        ValueError: if ``matrix_weights`` isn't two dimensional.
        ValueError: if ``scaling_matrix`` has entries that are not non-negative integers.
        ValueError: if the first column or last row of ``scaling_matrix`` has nonzero entries.
        ValueError: if ``scaling_matrix`` decreases along any row (moving left to right).
        ValueError: if ``scaling_matrix`` increases along any column (moving top to bottom).
        ValueError: if number of rows of ``matrix_weights`` doesn't equal length of ``prob_threshold_coords``.
        ValueError: if number of columns of ``matrix_weights`` doesn't equal length of ``severity_coords``.
        ValueError: ``len(evaluation_weights)`` is less than the maximum value in ``scaling_matrix``.
        ValueError: if ``prob_threshold_coords`` aren't strictly between 0 and 1.
        ValueError: if ``evaluation_weights`` aren't strictly positive.

    References:
        - Taggart, R. J., & Wilke, D. J. (2025). Warnings based on risk matrices: a coherent framework
          with consistent evaluation. arXiv. https://doi.org/10.48550/arXiv.2502.08891

    Examples:
        Returns weights for each risk matrix decision point, for the SHORT-RANGE scaling matrix of
        Taggart & Wilke (2025), with ESCALATION evaluation weights.

        >>> import numpy as np
        >>> from scores.emerging import weights_from_warning_scaling
        >>> scaling = np.array([
        >>>     [0, 2, 3, 3],
        >>>     [0, 1, 2, 3],
        >>>     [0, 1, 1, 2],
        >>>     [0, 0, 0, 0],
        >>> ])
        >>> weights_from_warning_scaling(
        >>>     scaling, [1, 2, 3],  "severity", ["MOD+", "SEV+", "EXT"], "prob_threshold", [0.1, 0.3, 0.5]
        >>> )
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
    if scaling_matrix_shape[0] - 1 != len(prob_threshold_coords):  # type: ignore
        raise ValueError("Length of `prob_threshold_coords` should be one less than rows of `scaling_matrix`")
    if scaling_matrix_shape[1] - 1 != len(severity_coords):  # type: ignore
        raise ValueError("Length of `severity_coords` should be one less than columns of `scaling_matrix`")
    if len(evaluation_weights) < np.max(scaling_matrix):  # type: ignore
        raise ValueError("length of `evaluation_weights` must be at least the highest value in `scaling_matrix`")

    # check on other inputs
    if (np.max(prob_threshold_coords) >= 1) or (np.min(prob_threshold_coords) <= 0):  # type: ignore
        raise ValueError("`prob_threshold_coords` must strictly between 0 and 1")
    if np.min(evaluation_weights) <= 0:  # type: ignore
        raise ValueError("values in `evaluation_weights` must be positive")

    weight_matrix = _scaling_to_weight_matrix(scaling_matrix, evaluation_weights)
    weight_array = matrix_weights_to_array(
        weight_matrix, severity_dim, severity_coords, prob_threshold_dim, prob_threshold_coords
    )
    return weight_array


def _scaling_to_weight_matrix(scaling_matrix, evaluation_weights):
    """
    Given a scaling matrix and evaluation weights, outputs the weight matrix for the
    decision points of the corresponding risk matrix.

    This is an implementation of the algorithm of Appendix B, Taggart & Wilke (2025).

    Args:
        scaling_matrix: np.array of warning scaling values. Values must be integers.
        evaluation_weights: list of weights for each warning level decision threshold.
    Returns:
        np.array of weights
    """
    max_level = max(np.max(scaling_matrix), len(evaluation_weights))

    scaling_matrix_shape = scaling_matrix.shape
    n_sev = scaling_matrix_shape[1] - 1  # number of severity categories for weight matrix
    n_prob = scaling_matrix_shape[0] - 1  # number of probability thresholds for weight matrix

    # initialise weight matrix wts
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
                wts[prob_index - 1, column - 1] = wts[prob_index - 1, column - 1] + evaluation_weights[level - 1]
                lowest_prob_index = prob_index

    wts = np.flip(wts, axis=0)

    return wts
