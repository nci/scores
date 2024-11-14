"""
Implementation of quantile interval score and interval score
"""
from typing import Optional

import xarray as xr

from scores.functions import apply_weights
from scores.typing import FlexibleDimensionTypes
from scores.utils import check_dims, gather_dimensions


def quantile_interval_score(  # pylint: disable=R0914
    fcst_lower_qtile: xr.DataArray,
    fcst_upper_qtile: xr.DataArray,
    obs: xr.DataArray,
    lower_qtile_level: float,
    upper_qtile_level: float,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Calculates the quantile interval score for interval forecasts. This score penalises
    the interval's width and whether the observation value lies outside the interval.

    .. math::
        \\text{quantile interval score} = \\underbrace{(q_{u} - q_{l})}_{\\text{interval width penalty}} +
        \\underbrace{\\frac{1}{\\alpha_l} \\cdot (q_{l} - y) \\cdot \\mathbb{1}(y < q_{l})}_{\\text{over-prediction penalty}} +
        \\underbrace{\\frac{1}{1 - \\alpha_u} \\cdot (y - q_{u}) \\cdot \\mathbb{1}(y < q_{u})}_{\\text{under-prediction penalty}}

    where
        - :math:`q_u` is the forecast at the upper quantile
        - :math:`q_l` is the forecast at the lower quantile
        - :math:`\\alpha_u` is the upper quantile level
        - :math:`\\alpha_l` is the lower quantile level
        - :math:`y` is the observation
        - :math:`\\mathbb{1}(condition)` is an indicator function that is 1 when the condition is true, and 0 otherwise.

    Args:
        fcst_lower_qtile: array of forecast values at the lower quantile.
        fcst_upper_qtile: array of forecast values at the upper quantile.
        obs: array of observations.
        lower_qtile_level: Quantile level between 0 and 1 (exclusive) for ``fcst_lower_qtile``.
        upper_qtile_level: Quantile level between 0 and 1 (exclusive) for ``fcst_upper_qtile``.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the quantile interval score. All other dimensions will be preserved.
            As a special case, "all" will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the quantile interval score. All other dimensions will be reduced. As a special case,
            "all" will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the quantile
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom) when aggregating the mean score across dimensions. Alternatively,
            it could be used for masking data.

    Returns:
        A Dataset with the dimensions specified in ``dims``.
        The dataset will have the following data variables:

        - interval_width_penalty: the interval width penalty contribution of the quantile interval score
        - overprediction_penalty: the over-prediction penalty contribution of the quantile interval score
        - underprediction_penalty: the under-prediction penalty contribution of the quantile interval score
        - total: sum of all penalties

    Raises:
        ValueError: If not (0 < lower_qtile_level < upper_qtile_level < 1).
        ValueError: If (fcst_lower_qtile > fcst_upper_qtile).any().

    References:
        Winkler, R. L. (1972). A Decision-Theoretic Approach to Interval Estimation. Journal of the American
        Statistical Association, 67(337), 187. https://doi.org/10.2307/2284720

    Examples:
        Calculate the quantile interval score for forecast intervals with lower and upper
        quantile levels of 0.1 and 0.6, respectively.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.continuous import quantile_interval_score
        >>> fcst_lower_level = xr.DataArray(np.random.uniform(10, 15, size=(30, 15)), dims=['time', 'station'])
        >>> fcst_upper_level = xr.DataArray(np.random.uniform(20, 25, size=(30, 15)), dims=['time', 'station'])
        >>> obs = xr.DataArray(np.random.uniform(8, 27,size=(30, 15)), dims=['time', 'station'])
        >>> quantile_interval_score(fcst_lower_level, fcst_upper_level, obs, 0.1, 0.6)
    """

    if not 0 < lower_qtile_level < upper_qtile_level < 1:
        raise ValueError(
            "Expected 0 < lower_qtile_level < upper_qtile_level < 1. But got "
            f"lower_qtile_level = {lower_qtile_level} and upper_qtile_level = {upper_qtile_level}"
        )
    # Check fcst_lower_qtile and fcst_upper_qtile have the same dimension
    check_dims(fcst_upper_qtile, fcst_lower_qtile.dims, mode="equal")
    specified_dims = reduce_dims or preserve_dims
    # check requested dims are a subset of fcst_lower_qtile (or fcst_upper_qtile) dimensions
    if specified_dims is not None:
        check_dims(xr_data=fcst_lower_qtile, expected_dims=specified_dims, mode="superset")
    # check obs dimensions are a subset of fcst_lower_qtile (or fcst_upper_qtile) dimensions
    check_dims(xr_data=obs, expected_dims=fcst_lower_qtile.dims, mode="subset")  # type: ignore
    if (fcst_lower_qtile > fcst_upper_qtile).any():
        raise ValueError("Input does not satisfy fcst_lower_qtile < fcst_upper_qtile condition.")
    interval_width = fcst_upper_qtile - fcst_lower_qtile
    lower_diff = (1 / lower_qtile_level) * (fcst_lower_qtile - obs)
    lower_diff = lower_diff.clip(min=0.0)
    upper_diff = (1 / (1 - upper_qtile_level)) * (obs - fcst_upper_qtile)
    upper_diff = upper_diff.clip(min=0.0)
    total_score = interval_width + lower_diff + upper_diff
    components = {
        "interval_width_penalty": interval_width,
        "overprediction_penalty": lower_diff,
        "underprediction_penalty": upper_diff,
        "total": total_score,
    }
    result = xr.Dataset(components)
    reduce_dims = gather_dimensions(fcst_lower_qtile.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)  # type: ignore[assignment]
    results = apply_weights(result, weights=weights)
    score = results.mean(dim=reduce_dims)
    return score


def interval_score(
    fcst_lower_qtile: xr.DataArray,
    fcst_upper_qtile: xr.DataArray,
    obs: xr.DataArray,
    interval_range: float,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Calculates the interval score for interval forecasts.
    This function calls the :py:func:`scores.continuous.quantile_interval_score` function
    to calculate the interval score for cases where the quantile level range is symmetric.

    .. math::
        \\text{interval score} = \\underbrace{(q_{u} - q_{l})}_{\\text{interval width penalty}} +
        \\underbrace{\\frac{2}{\\alpha} \\cdot (q_{l} - y) \\cdot \\mathbb{1}(y < q_{l})}_{\\text{over-prediction penalty}} +
        \\underbrace{\\frac{2}{\\alpha} \\cdot (y - q_{u}) \\cdot \\mathbb{1}(y < q_{u})}_{\\text{under-prediction penalty}}

    where
        - :math:`q_u` is the forecast at the upper quantile
        - :math:`q_l` is the forecast at the lower quantile
        - :math:`\\alpha` is the confidence level that is equal to :math:`1 - \\text{interval_range}`
        - :math:`y` is the observation
        - :math:`\\mathbb{1}(condition)` is an indicator function that is 1 when the condition is true, and 0 otherwise.


    Args:
        fcst_lower_qtile: array of forecast values at the lower quantile.
        fcst_upper_qtile: array of forecast values at the upper quantile.
        obs: array of observations.
        interval_range: Range (length) of interval (e.g., 0.9 for 90% confidence level, which
            will result in lower quantile of 0.05 and upper quantile of 0.95). Must be strictly
            between 0 and 1.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the interval score. All other dimensions will be preserved.
            As a special case, "all" will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating
            the interval score. All other dimensions will be reduced. As a special case,
            "all" will allow all dimensions to be preserved. In this case, the result will be in
            the same shape/dimensionality as the forecast, and the errors will be the quantile
            score at each point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely. Only one of ``reduce_dims``
            and ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom) when aggregating the mean score across dimensions. Alternatively,
            it could be used for masking data.

    Returns:
        A Dataset with the dimensions specified in ``dims``.
        The dataset will have the following data variables:

        - interval_width_penalty: the interval width penalty contribution of the interval score
        - overprediction_penalty: the over-prediction penalty contribution of the interval score
        - underprediction_penalty: the under-prediction penalty contribution of the interval score
        - total: sum of all penalties

        As can be seen in the interval score equation, the lower and upper quantile levels are
        derived from the interval range: ``lower_qtile_level = (1 - interval_range) / 2``
        and ``upper_qtile_level = (1 + interval_range) / 2``.

    Raises:
        ValueError: If not 0 < interval_range < 1.
        ValueError: If (fcst_lower_qtile > fcst_upper_qtile).any().

    References:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction,
        and estimation. Journal of the American Statistical Association, 102(477), 359-378.
        Section 6.2. https://doi.org/10.1198/016214506000001437

     See also:
        :py:func:`scores.continuous.quantile_interval_score`

    Examples:
        Calculate the interval score for forecast intervals with an interval range of 0.5
        (i.e., lower and upper quantile levels are 0.25 and 0.75, respectively).

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.continuous import interval_score
        >>> fcst_lower_level = xr.DataArray(np.random.uniform(10, 15, size=(30, 15)), dims=['time', 'station'])
        >>> fcst_upper_level = xr.DataArray(np.random.uniform(20, 25, size=(30, 15)), dims=['time', 'station'])
        >>> obs = xr.DataArray(np.random.uniform(8, 27,size=(30, 15)), dims=['time', 'station'])
        >>> interval_score(fcst_lower_level, fcst_upper_level, obs, 0.5)
    """
    if interval_range <= 0 or interval_range >= 1:
        raise ValueError("`interval_range` must be strictly between 0 and 1")
    score = quantile_interval_score(
        fcst_lower_qtile=fcst_lower_qtile,
        fcst_upper_qtile=fcst_upper_qtile,
        obs=obs,
        lower_qtile_level=(1 - interval_range) / 2,
        upper_qtile_level=(1 + interval_range) / 2,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    return score
