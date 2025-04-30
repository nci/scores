"""
This module contains methods related to the Brier score
"""

import operator
from collections.abc import Sequence
from numbers import Real
from typing import Callable, Optional, cast

import xarray as xr

from scores.continuous import mse
from scores.functions import apply_weights
from scores.processing import binary_discretise
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_binary, gather_dimensions


def brier_score(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: bool = True,
) -> XarrayLike:
    """
    Calculates the Brier score for forecast and observed data.

    .. math::
        \\text{brier score} = \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2

    If you want to speed up performance with Dask and are confident of your input data,
    or if you want observations to take values between 0 and 1, set ``check_args`` to ``False``.

    Args:
        fcst: Forecast or predicted variables in xarray.
        obs: Observed variables in xarray.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Brier score. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Brier score. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom).
        check_args: will perform some tests on the data if set to True.
    Raises:
        ValueError: if ``fcst`` contains non-nan values outside of the range [0, 1]
        ValueError: if ``obs`` contains non-nan values not in the set {0, 1}

    References:
        - Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
          Monthly Weather Review, 78(1), 1-3. https://doi.org/fp62r6
        - https://en.wikipedia.org/wiki/Brier_score

    See Also:
        - :py:func:`scores.probability.brier_score_for_ensemble`
    """
    if check_args:
        error_msg = ValueError("`fcst` contains values outside of the range [0, 1]")
        if isinstance(fcst, xr.Dataset):
            # The .values is required as item() is not yet a valid method on Dask
            if fcst.to_array().max().values.item() > 1 or fcst.to_array().min().values.item() < 0:
                raise error_msg
        else:
            if fcst.max().values.item() > 1 or fcst.min().values.item() < 0:
                raise error_msg
        check_binary(obs, "obs")

    return mse(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights)


def brier_score_for_ensemble(
    fcst: XarrayLike,  # type: ignore
    obs: XarrayLike,  # type: ignore
    ensemble_member_dim: str,
    event_thresholds: Real | Sequence[Real],
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    fair_correction: bool = True,
    event_threshold_operator: Callable = operator.ge,
    threshold_dim: str = "threshold",
) -> XarrayLike:  # type: ignore
    """
    Calculates the Brier score for an ensemble forecast for the provided thresholds. By default,
    the fair Brier score is calculated, which is a modified version of the Brier score that
    applies a correction related to the number of ensemble members and the number of
    ensemble members that predict the event.

    Given an event defined by ``event_threshold``, the fair Brier score for ensembles is defined as


    .. math::
        s_{i,y} = \\left(\\frac{i}{m} - y\\right)^2 - \\frac{i(m-i)}{m^2(m-1)}

    Where:

    - :math:`i` is the number of ensemble members that predict the event
    - :math:`m` is the total number of ensemble members
    - :math:`y` is the binary observation (0 if the event was not observed or 1 otherwise)

    When the fair correction is not applied (i.e., ``fair_correction=False``),
    the Brier score is calculated as:


    .. math::
        s_{i,y} = \\left(\\frac{i}{m} - y\\right)^2

    The fair correction will only be applied at coordinates where the number of ensemble members
    is greater than 1 and ``fair_correction=True``.
    When the ``fair_correction`` is set to ``True`` and the number of ensemble members is 1, the
    fair correction will not be applied to avoid division by zero. If you want to
    set the minimum number of ensemble members required to calculate the Brier score,
    we recommend preprocessing your data to remove data points with fewer than the
    minimum number of ensemble members that you want.

    Args:
        fcst: Real valued ensemble forecasts in xarray. The proportion of ensemble members
        above a threshold is calculated within this function.
        obs: Real valued observations in xarray. These values are converted to binary
            values in this function based on the supplied event thresholds and operators.
        ensemble_member_dim: The dimension name of the ensemble member.
        event_thresholds: The threshold(s) that define what is considered an event. If
            multiple thresholds are provided, they should be monotonically increasing with no NaNs.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Brier score. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the score. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude)
            when calculating the mean score across specified dimensions via ``reduce_dims`` or ``preserve_dims``.
        fair_correction: Whether or not to apply the fair Brier score correction. Default is True.
        event_threshold_operator: The mode to use to define an event relative to a supplied threshold.
            Default is ``operator.ge``, which means that an event occurs
            if the observation is greater than or equal to the threshold.
            Similarly, an event will have been considered to be forecast by an ensemble member
            if the forecast is greater than or equal to the threshold.
            Passing in ``operator.lt`` will give the same results as it is mathematically
            equivalent in this case. Alternatively, you can provide
            ``operator.gt`` which means that an event occurs if the observation
            is strictly greater than the threshold. Again, an event for an ensemble member
            will have been considered to be forecast if the forecast is greater than the threshold.
            Using``operator.le`` will give the same results as ``operator.gt`` as it is mathematically
            equivalent in this case.
        threshold_dim: The name of the threshold dimension in the output. Default is 'threshold'.
           It must not exist as a dimension in the forecast or observation data.


    Returns:
        The Brier score for the ensemble forecast.

    Raises:
        ValueError: if values in ``event_thresholds`` are not monotonically increasing.
        ValueError: if ``fcst`` contains the dimension provided in ``threshold_dim``.
        ValueError: if ``obs`` contains the dimension provided in ``threshold_dim``.
        ValueError: if ``weights`` contains the dimension provided in ``threshold_dim``.
        ValueError: if ``ensemble_member_dim`` is not in ``fcst`` dimensions.

    References:
        - Ferro, C. A. T. (2013). Fair scores for ensemble forecasts. Quarterly
            Journal of the Royal Meteorological Society, 140(683), 1917–1923.
            https://doi.org/10.1002/qj.2270

    See Also:
        - :py:func:`scores.probability.brier_score`

    Examples:
        Calculate the Brier score for an ensemble forecast for a single threshold:

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import brier_score_for_ensemble
        >>> fcst = 10 * xr.DataArray(np.random.rand(10, 10), dims=['time', 'ensemble'])
        >>> obs = 10 * xr.DataArray(np.random.rand(10), dims=['time'])
        >>> brier_score_for_ensemble(fcst, obs, ensemble_member_dim='ensemble', thresholds=0.5)

        Calculate the Brier score for an ensemble forecast for multiple thresholds:
        >>> thresholds = [0.1, 5, 9]
        >>> brier_score_for_ensemble(fcst, obs, ensemble_member_dim='ensemble', thresholds=thresholds)

    """
    if event_threshold_operator not in [operator.ge, operator.gt, operator.le, operator.lt]:
        raise ValueError("event_threshold_operator must be one of operator.ge, operator.gt, operator.le, operator.lt")
    if threshold_dim in fcst.dims:
        raise ValueError("threshold_dim is not allowed to be the same as a dim in fcst")
    if threshold_dim in obs.dims:
        raise ValueError("threshold_dim is not allowed to be the same as a dim in obs")

    weights_dims = None
    if weights is not None:
        if threshold_dim in weights.dims:
            raise ValueError("threshold_dim is not allowed to be the same as a dim in weights")
        weights_dims = weights.dims

    dims_for_mean = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=ensemble_member_dim,
    )
    if isinstance(event_thresholds, (float, int)):
        event_thresholds = [event_thresholds]  # type: ignore
    thresholds_xr = xr.DataArray(event_thresholds, dims=[threshold_dim], coords={threshold_dim: event_thresholds})

    # calculate i term in equation
    member_event_count = event_threshold_operator(fcst, thresholds_xr).sum(dim=ensemble_member_dim)
    # calculate m term in equation
    total_member_count = fcst.notnull().sum(dim=ensemble_member_dim)

    binary_obs = binary_discretise(obs, event_thresholds, event_threshold_operator)

    result = (member_event_count / total_member_count - binary_obs) ** 2
    if fair_correction:
        fair_corr = (member_event_count * (total_member_count - member_event_count)) / (
            total_member_count**2 * (total_member_count - 1)
        )
        # For coordinates where the ensemble member count is 1, the fair correction will
        # have a NaN value. In this situation, we set the fair correction to 0 at these coordinates.
        # This may be important if you have a lagged ensemble where at least one time step
        # has only a single ensemble member
        fair_correction = fair_corr.fillna(0)
        result -= fair_correction

    # apply weights and take means across specified dims
    result = apply_weights(result, weights=weights).mean(dim=dims_for_mean)

    return result  # type: ignore
