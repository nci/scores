"""
Relative Economic Value metrics for forecast evaluation
"""

import warnings
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import xarray as xr

from scores.categorical import probability_of_detection, probability_of_false_detection
from scores.processing import binary_discretise, broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions


def relative_economic_value_from_rates(
    pod: XarrayLike,
    pofd: XarrayLike,
    climatology: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    cost_loss_dim: str = "cost_loss_ratio",
):
    """
    Calculates Relative Economic Value (REV) from pre-computed detection rates.

    REV measures the economic benefit of using forecasts compared to climatology,
    relative to perfect forecasts. This function computes REV directly from
    probability of detection (POD), probability of false detection (POFD), and
    climatological frequency.

    The relative economic value is calculated using:

    .. math::
        \\begin{split}
        \\text{REV} = \\frac{\\min(\\alpha, \\bar{o}) - F\\alpha(1-\\bar{o})
                              + H\\bar{o}(1-\\alpha) - \\bar{o}}
                             {\\min(\\alpha, \\bar{o}) - \\bar{o}\\alpha}
        \\end{split}

    where:
        - :math:`\\alpha` is the cost-loss ratio
        - :math:`\\bar{o}` is the climatological frequency (base rate)
        - :math:`F` is the probability of false detection (false alarm rate)
        - :math:`H` is the probability of detection (hit rate)

    Args:
        pod: Probability of detection (hit rate). Values should be between 0 and 1,
            where 1 indicates all events were correctly forecast.
        pofd: Probability of false detection (false alarm rate). Values should be
            between 0 and 1, where 0 indicates no false alarms.
        climatology: Climatological frequency of the event (base rate). Values should
            be between 0 and 1, representing the proportion of time the event occurs.
        cost_loss_ratios: Cost-loss ratio(s) at which to calculate REV. Must be
            strictly monotonically increasing values between 0 and 1. Can be a single
            float or sequence of floats. The cost-loss ratio represents the ratio of
            the cost of taking protective action to the loss incurred if the event
            occurs without protection.
        cost_loss_dim: Name of the cost-loss ratio dimension in output. Default is
            'cost_loss_ratio'. Must not exist as a dimension in any input array.

    Returns:
        xarray.DataArray: REV values with dimensions from broadcasting the input
            arrays, plus an additional 'cost_loss_ratio' dimension with coordinates
            matching the supplied cost-loss ratios.

    Raises:
        ValueError: If cost_loss_ratios is not strictly monotonically increasing,
            is not one-dimensional, or contains values outside [0, 1].
        ValueError: If 'cost_loss_ratio' dimension already exists in any input array.

    Notes:
        - REV = 1 indicates perfect forecast value (as good as perfect information)
        - REV = 0 indicates no value over climatology
        - REV < 0 indicates the forecast performs worse than climatology
        - This function is typically called internally by `relative_economic_value()`
          after computing POD and POFD from forecasts and observations

    References:
        Richardson, D. S. (2000). Skill and relative economic value of the ECMWF
        ensemble prediction system. Quarterly Journal of the Royal Meteorological
        Society, 126(563), 649-667. https://doi.org/10.1002/qj.49712656313

    Examples:
        Calculate REV from pre-computed rates:

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import relative_economic_value_from_rates
        >>>
        >>> # Pre-computed detection rates
        >>> pod = xr.DataArray([0.8, 0.6, 0.4], dims=['threshold'])
        >>> pofd = xr.DataArray([0.2, 0.1, 0.05], dims=['threshold'])
        >>> climatology = xr.DataArray(0.3)  # 30% base rate
        >>>
        >>> # Calculate REV at multiple cost-loss ratios
        >>> cost_loss_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> rev = relative_economic_value_from_rates(
        ...     pod, pofd, climatology, cost_loss_ratios
        ... )
        >>> rev.dims
        ('cost_loss_ratio', 'threshold')

        Calculate REV values:

        >>> # Perfect forecast scenario
        >>> pod_perfect = xr.DataArray(1.0)
        >>> pofd_perfect = xr.DataArray(0.0)
        >>> climatology = xr.DataArray(0.5)
        >>>
        >>> rev = relative_economic_value_from_rates(
        ...     pod_perfect, pofd_perfect, climatology,
        ...     cost_loss_ratios=[0.5],
        ... )
        >>> rev.values  # Returns finite value close to 1
        array([1.])

    See Also:
        - :py:func:`scores.probability.relative_economic_value`
        - :py:func:`scores.categorical.probability_of_detection`
        - :py:func:`scores.categorical.probability_of_false_detection`
    """

    # From Richardson, D.S., QJR Meteorol Soc 2000, 126, pp 649-667
    # Hit rate (probability of detection) equation 4
    # False alarm rate (probability of false detection) equation 5

    try:
        check_monotonic_array(cost_loss_ratios)
    except Exception as ex:
        raise type(ex)("for cost_loss_ratios, " + str(ex))

    if cost_loss_dim in (set(pod.dims) | set(pofd.dims) | set(climatology.dims)):
        raise ValueError(f"dimension '{cost_loss_dim}' must not be in input data")

    # Handle Dataset inputs by applying to each variable
    if isinstance(pod, xr.Dataset):
        result_dict = {}
        for var in pod.data_vars:
            result_dict[var] = relative_economic_value_from_rates(
                pod[var],
                pofd[var] if isinstance(pofd, xr.Dataset) else pofd,
                climatology[var] if isinstance(climatology, xr.Dataset) else climatology,
                cost_loss_ratios,
                cost_loss_dim=cost_loss_dim,
            )
        return xr.Dataset(result_dict)

    alphas = xr.DataArray(
        cost_loss_ratios,
        dims=[cost_loss_dim],
        coords={cost_loss_dim: cost_loss_ratios},
    )

    obar, alphas = xr.broadcast(climatology, alphas)
    climatological_term = obar.copy()
    climatological_term.values = np.minimum(obar.values, alphas.values)

    # calculate the relative economic value (equation 8)

    rev = ((climatological_term) - (pofd * alphas * (1 - obar)) + (pod * obar * (1 - alphas)) - obar) / (
        climatological_term - obar * alphas
    )

    # tidy up floating point infinities - necessary because you can get
    # near-zero in the denominator for alpha=0 or alpha=1
    rev = rev.where(~np.isinf(rev))

    return rev


def _create_output_dataset(
    rev: xr.DataArray,
    thresholds: Sequence[float],
    cost_loss_ratios: Sequence[float],
    derived_metrics: Optional[Sequence[str]],
    threshold_outputs: Optional[Sequence[float]],
    threshold_dim: str,
    cost_loss_dim: str,
) -> xr.Dataset:
    """
    Create output Dataset with derived metrics and threshold slices.

    Args:
        rev: Full REV DataArray with threshold and cost_loss_ratio dimensions
        thresholds: List of threshold values
        cost_loss_ratios: List of cost-loss ratio values
        derived_metrics: Derived metrics to compute
        threshold_outputs: Specific thresholds to extract
        threshold_dim: Name of threshold dimension
        cost_loss_dim: Name of cost-loss ratio dimension

    Returns:
        Dataset with requested outputs

    Raises:
        ValueError: If invalid derived_metrics values provided
        ValueError: If 'rational_user' requested but thresholds don't match cost_loss_ratios
    """
    derived_metrics = [] if derived_metrics is None else list(derived_metrics)
    threshold_outputs = [] if threshold_outputs is None else list(threshold_outputs)

    # Store original attributes
    attributes = rev.attrs
    rev.attrs = {}

    result = xr.Dataset()
    result.attrs = attributes

    # Add derived metrics
    for mode in derived_metrics:
        if mode == "maximum":
            result["maximum"] = rev.max(dim=threshold_dim)
        elif mode == "rational_user":
            # Check that thresholds match cost_loss_ratios exactly
            if list(thresholds) != list(cost_loss_ratios):
                raise ValueError(
                    "Can only specify derived_metrics 'rational_user' if thresholds and "
                    "cost_loss_ratios are identical"
                )
            # Extract diagonal where threshold == cost_loss_ratio
            actual_values = []
            for alpha in cost_loss_ratios:
                val = rev.sel({threshold_dim: alpha, cost_loss_dim: alpha})
                actual_values.append(val)

            # Concat and assign proper coordinates
            result["rational_user"] = xr.concat(actual_values, dim=cost_loss_dim)
            result["rational_user"][cost_loss_dim] = cost_loss_ratios

            # Drop threshold coordinate since it's redundant
            if threshold_dim in result["rational_user"].coords:
                result["rational_user"] = result["rational_user"].drop_vars(threshold_dim)

        else:
            raise ValueError(
                f"Invalid derived_metrics value: '{mode}'. Valid options are " "'maximum' and 'rational_user'"
            )

    # Add threshold-specific outputs
    for thresh in threshold_outputs:
        # Use string formatting that's valid for variable names
        var_name = f"threshold_{thresh}".replace(".", "_")
        result[var_name] = rev.sel({threshold_dim: thresh}).drop_vars(threshold_dim)

    return result


def _check_dask_arrays(fcst: XarrayLike, obs: XarrayLike) -> None:
    """
    Warn if Dask arrays detected during validation.

    Note: This check forces computation of Dask arrays.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import dask.array as da

        is_dask_fcst = isinstance(fcst.data if hasattr(fcst, "data") else fcst, da.Array)
        is_dask_obs = isinstance(obs.data if hasattr(obs, "data") else obs, da.Array)
        if is_dask_fcst or is_dask_obs:
            warnings.warn(
                "check_args=True will force computation on Dask arrays. "
                "Set check_args=False to skip validation and preserve lazy evaluation.",
                UserWarning,
                stacklevel=4,  # Adjusted for extra function call depth
            )
    except ImportError:
        pass  # Dask not available, proceed with validation


def _validate_dimensions(
    fcst: XarrayLike,
    obs: XarrayLike,
    weights: Optional[xr.DataArray],
    threshold_dim: str,
    cost_loss_dim: str,
) -> None:
    """Check for dimension conflicts in inputs."""
    # Check fcst dimensions
    if threshold_dim in fcst.dims:
        raise ValueError(f"'{threshold_dim}' cannot be a dimension in fcst")
    if cost_loss_dim in fcst.dims:
        raise ValueError(f"'{cost_loss_dim}' cannot be a dimension in fcst")

    # Check obs dimensions
    if threshold_dim in obs.dims:
        raise ValueError(f"'{threshold_dim}' cannot be a dimension in obs")
    if cost_loss_dim in obs.dims:
        raise ValueError(f"'{cost_loss_dim}' cannot be a dimension in obs")

    # Check weights dimensions
    if weights is not None:
        if threshold_dim in weights.dims:
            raise ValueError(f"'{threshold_dim}' cannot be a dimension in weights")
        if cost_loss_dim in weights.dims:
            raise ValueError(f"'{cost_loss_dim}' cannot be a dimension in weights")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")


def _validate_cost_loss_ratios(cost_loss_ratios: Union[float, Sequence[float]]) -> None:
    """Validate cost-loss ratio values are in [0,1] and monotonically increasing."""
    clr_array = np.atleast_1d(cost_loss_ratios)

    if not np.all((clr_array >= 0) & (clr_array <= 1)):
        raise ValueError("cost_loss_ratios must be between 0 and 1")

    if len(clr_array) > 1 and not np.all(np.diff(clr_array) > 0):
        raise ValueError("cost_loss_ratios must be strictly monotonically increasing")


def _validate_observations(obs: XarrayLike) -> None:
    """Check observations contain only binary values (0, 1, or NaN)."""
    if isinstance(obs, xr.Dataset):
        obs_vals = obs.to_array().values
    else:
        obs_vals = obs.values

    unique_obs = np.unique(obs_vals[~np.isnan(obs_vals)])
    if not np.all(np.isin(unique_obs, [0, 1])):
        raise ValueError("obs must contain only 0, 1, or NaN values")


def _validate_forecasts(
    fcst: XarrayLike,
    threshold: Optional[Union[float, Sequence[float]]],
    threshold_outputs: Optional[Sequence[float]],
) -> None:
    """Validate forecast values and threshold configuration."""
    # Extract forecast values
    if isinstance(fcst, xr.Dataset):
        fcst_vals = fcst.to_array().values
    else:
        fcst_vals = fcst.values

    fcst_min = np.nanmin(fcst_vals)
    fcst_max = np.nanmax(fcst_vals)

    if threshold is not None:
        # Probabilistic forecasts: validate range
        if fcst_min < 0 or fcst_max > 1:
            raise ValueError("When threshold is provided, fcst must contain values between 0 and 1")

        # Validate threshold values
        thresh_array = np.atleast_1d(threshold)
        if not np.all((thresh_array >= 0) & (thresh_array <= 1)):
            raise ValueError("threshold values must be between 0 and 1")
        if len(thresh_array) > 1 and not np.all(np.diff(thresh_array) > 0):
            raise ValueError("threshold values must be strictly monotonically increasing")

        # Validate threshold_outputs
        if threshold_outputs is not None:
            if not set(threshold_outputs) <= set(thresh_array):
                raise ValueError("values in threshold_outputs must be in the supplied threshold parameter")
    else:
        # Binary forecasts: validate values
        unique_fcst = np.unique(fcst_vals[~np.isnan(fcst_vals)])
        if not np.all(np.isin(unique_fcst, [0, 1])):
            raise ValueError(
                "When threshold is None, fcst must contain only 0, 1, or NaN values. "
                "For probabilistic forecasts, provide threshold parameter."
            )

        # threshold_outputs requires thresholds
        if threshold_outputs:
            raise ValueError("threshold_outputs can only be used when threshold parameter is provided")


def _validate_derived_metrics(
    derived_metrics: Optional[Sequence[str]],
    threshold: Optional[Union[float, Sequence[float]]],
) -> None:
    """Validate derived metrics configuration."""
    if derived_metrics is None:
        return

    valid_special = {"maximum", "rational_user"}
    invalid = set(derived_metrics) - valid_special
    if invalid:
        raise ValueError(f"Invalid derived_metrics values: {invalid}. Valid options are {valid_special}")

    if "rational_user" in derived_metrics and threshold is None:
        raise ValueError("derived_metrics 'rational_user' can only be used when threshold parameter is provided")


def _validate_rev_inputs(
    fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    threshold: Optional[Union[float, Sequence[float]]],
    threshold_dim: str,
    cost_loss_dim: str,
    weights: Optional[xr.DataArray],
    derived_metrics: Optional[Sequence[str]],
    threshold_outputs: Optional[Sequence[float]],
) -> None:
    """
    Validate inputs for REV calculation.

    Note: This function forces computation of Dask arrays. For Dask workflows,
    set check_args=False to skip validation.

    Raises:
        ValueError: For various input validation failures
        UserWarning: If Dask arrays detected with check_args=True
    """
    _check_dask_arrays(fcst, obs)
    _validate_dimensions(fcst, obs, weights, threshold_dim, cost_loss_dim)
    _validate_cost_loss_ratios(cost_loss_ratios)
    _validate_observations(obs)
    _validate_forecasts(fcst, threshold, threshold_outputs)
    _validate_derived_metrics(derived_metrics, threshold)


def _calculate_rev_core(
    binary_fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    dims_to_reduce: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    cost_loss_dim: str = "cost_loss_ratio",
) -> XarrayLike:
    """
    Core REV calculation from binary forecasts using scores library functions.
    """

    # Ensure that we're censoring off obs where forecasts are missing,
    # mainly for climatology calculations.
    binary_fcst, obs = broadcast_and_match_nan(binary_fcst, obs)

    pod = probability_of_detection(binary_fcst, obs, reduce_dims=dims_to_reduce, weights=weights, check_args=False)

    pofd = probability_of_false_detection(
        binary_fcst, obs, reduce_dims=dims_to_reduce, weights=weights, check_args=False
    )

    climatology = calculate_climatology(
        obs,
        reduce_dims=dims_to_reduce,
        weights=weights,
    )

    result = relative_economic_value_from_rates(
        pod=pod, pofd=pofd, climatology=climatology, cost_loss_ratios=cost_loss_ratios, cost_loss_dim=cost_loss_dim
    )

    return result


def calculate_climatology(
    obs: XarrayLike,
    *,  # Force keyword arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: bool = True,
) -> XarrayLike:
    """
    Calculates the climatological base rate (mean of observations).

    Handles weighted and unweighted means, including the case where weights
    don't span all dimensions being reduced.

    Args:
        obs: An array containing binary values (typically {0, 1, np.nan})
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating climatology. All other dimensions will be preserved. As a
            special case, 'all' will reduce all dimensions. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating climatology. All other dimensions will be reduced.
            As a special case, 'all' will preserve all dimensions. Only one of
            `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        weights: An array of weights to apply (e.g., weighting a grid by latitude).
            If None, unweighted mean is calculated. Weights must be broadcastable
            to the data dimensions and should not contain negative or NaN values.
        check_args: If True, validates input arguments. Set to False to improve performance.

    Returns:
        A DataArray of the climatological base rate.
    """

    if check_args and weights is not None:
        if (weights < 0).any():
            raise ValueError("'weights' contains negative values")
        if weights.isnull().any():
            raise ValueError("'weights' contains NaN values")

    # Use gather_dimensions to determine which obs dimensions to reduce
    # gather_dimensions will validate reduce_dims/preserve_dims automatically.
    dims_to_reduce = gather_dimensions(
        fcst_dims=(),  # Not needed for climatology
        obs_dims=obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights_dims=weights.dims if weights is not None else None,
    )

    obs_reduce_dims = tuple(d for d in dims_to_reduce if d in obs.dims)

    if weights is not None:
        # Identify which dimensions can be handled with weighted sum
        weight_reduce_dims = tuple(d for d in obs_reduce_dims if d in weights.dims)

        if weight_reduce_dims:
            # Weighted mean over dimensions that weights span
            obs_weighted = obs * weights
            climatology = obs_weighted.sum(dim=weight_reduce_dims, skipna=True) / weights.sum(dim=weight_reduce_dims)

            # Unweighted mean over remaining dimensions
            remaining_dims = tuple(d for d in obs_reduce_dims if d not in weight_reduce_dims)
            if remaining_dims:
                climatology = climatology.mean(dim=remaining_dims, skipna=True)
        else:
            # Weights exist but don't span any reduction dimensions - ignore them
            climatology = obs.mean(dim=obs_reduce_dims, skipna=True)
    else:
        # Simple unweighted mean
        climatology = obs.mean(dim=obs_reduce_dims, skipna=True)

    return climatology


def check_monotonic_array(array: Union[Sequence[float], np.ndarray]) -> None:
    """
    Checks array values are in range [0, 1] and monotonically increasing.
    """
    try:
        np_array = np.array(array, dtype=float)
    except Exception as ex:
        raise TypeError("could not convert array into a numpy ndarray of floats") from ex

    if len(np_array.shape) != 1:
        raise ValueError("array must be one-dimensional")

    if max(np_array) > 1 or min(np_array) < 0:
        raise ValueError("array values should be between 0 and 1.")

    if len(np_array) > 1 and not (np_array[1:] - np_array[:-1] >= 0).all():
        raise ValueError("the supplied array is not monotonic increasing.")


# pylint: disable=too-many-arguments,too-many-locals
def relative_economic_value(
    fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    *,  # Force keyword arguments to be keyword-only
    threshold: Optional[Union[float, Sequence[float]]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    threshold_dim: str = "threshold",
    cost_loss_dim: str = "cost_loss_ratio",
    derived_metrics: Optional[Sequence[str]] = None,
    threshold_outputs: Optional[Sequence[float]] = None,
    check_args: bool = True,
) -> XarrayLike:
    """
    Calculates the Relative Economic Value (REV) for probabilistic or binary forecasts.

    REV measures the economic benefit of using a forecast compared to a baseline strategy
    (climatology), relative to a perfect forecast. It evaluates forecasts across different
    cost-loss scenarios, where users must decide whether to take protective action based
    on forecast information.

    For probabilistic forecasts, multiple decision thresholds are evaluated to find the
    optimal strategy. For binary forecasts, a single decision has already been made.

    For ensemble forecasts, consider using the scores.processing.binary_discretise_proportion
    function to convert ensembles to empirical probabilities before calculating REV. See the
    tutorial notebook for an example.

    .. math::
        \\begin{split}
        \\text{REV} = \\frac{\\min(\\alpha, \\bar{o}) - F\\alpha(1-\\bar{o})
                              + H\\bar{o}(1-\\alpha) - \\bar{o}}
                             {\\min(\\alpha, \\bar{o}) - \\bar{o}\\alpha}
        \\end{split}

    where:
        - :math:`\\bar{o}` is the climatological frequency (base rate) of the event
        - :math:`\\alpha` is the cost-loss ratio
        - :math:`F` is the probability of false detection (false alarm rate)
        - :math:`H` is the probability of detection (hit rate)

    Args:
        fcst: Forecast data. Can be:
            - Probabilistic: values between 0 and 1 representing event probabilities
            - Binary: values of 0 or 1 representing categorical yes/no forecasts
        obs: Binary observations with values of 0 (no event) or 1 (event occurred).
        cost_loss_ratios: The cost-loss ratio(s) at which to calculate REV. Must be
            strictly monotonically increasing values between 0 and 1. The cost-loss
            ratio represents the ratio of the cost of taking protective action to
            the loss incurred if the event occurs without protection.
        threshold: Decision threshold(s) for converting probabilistic forecasts to
            binary decisions. Required for probabilistic forecasts. Each threshold
            converts forecasts to 1 where fcst >= threshold, 0 otherwise. Must be
            strictly monotonically increasing values between 0 and 1. If None, assumes
            fcst is already binary (0 or 1).
        reduce_dims: Dimensions to reduce when calculating REV. All other dimensions
            will be preserved. Cannot be used with preserve_dims.
        preserve_dims: Dimensions to preserve when calculating REV. All other dimensions
            will be reduced. As a special case, 'all' preserves all dimensions. Cannot
            be used with reduce_dims.
        weights: Optional array of weights for weighted averaging. Must be broadcastable
            to the data dimensions and cannot contain negative or NaN values. Weights
            need not cover all dimensions being reduced; unweighted averaging is applied
            to dimensions not in weights.
        threshold_dim: Name of the threshold dimension in output. Default is 'threshold'.
            Must not exist as a dimension in fcst or obs.
        cost_loss_dim: Name of the cost-loss ratio dimension in output. Default is
            'cost_loss_ratio'. Must not exist as a dimension in fcst or obs.
        derived_metrics: Optional list of derived metrics to compute. Options are:
            - 'maximum': Maximum REV across all thresholds (envelope of value curves,
              also called "potential value")
            - 'rational_user': REV when threshold equals cost-loss ratio (requires thresholds
              to match cost_loss_ratios exactly). The user is assuming that the forecast
              probabilities are reliable.
            If None and threshold is provided, returns full DataArray with threshold
            dimension. If None and threshold is None, returns DataArray with
            cost_loss_ratio dimension only.
        threshold_outputs: Optional list of specific threshold values to extract as
            separate variables in the output Dataset. Values must exist in the threshold
            parameter. Only used when threshold is provided.
        check_args: If True, validates input arguments. Set to False to improve performance
            with Dask (validation forces computation of lazy arrays). When working with
            Dask arrays, it is recommended to set check_args=False.

    Returns:
        XarrayLike:
            - If derived_metrics or threshold_outputs is specified: xr.Dataset with data
              variables for each requested output
            - If threshold is provided: xr.DataArray with dimensions from
              reduce_dims/preserve_dims, plus cost_loss_dim and threshold_dim
            - If threshold is None: xr.DataArray with dimensions from
              reduce_dims/preserve_dims, plus cost_loss_dim

    Raises:
        ValueError: If fcst contains probabilistic values but threshold is None
        ValueError: If fcst contains values outside [0, 1] when threshold is provided
        ValueError: If fcst contains values other than 0 or 1 when threshold is None
        ValueError: If obs contains values other than 0 or 1
        ValueError: If cost_loss_ratios not strictly monotonically increasing or not in [0, 1]
        ValueError: If threshold values not strictly monotonically increasing or not in [0, 1]
        ValueError: If threshold_dim or cost_loss_dim already exist in fcst or obs
        ValueError: If both reduce_dims and preserve_dims are specified
        ValueError: If threshold_outputs values are not in threshold parameter
        ValueError: If 'rational_user' in derived_metrics but thresholds don't match cost_loss_ratios

    Notes:
        - REV = 1 indicates perfect forecast value (as good as perfect information)
        - REV = 0 indicates no value over climatology
        - REV < 0 indicates the forecast is worse than using climatology
        - For probabilistic forecasts, 'maximum' value represents the maximum achievable
          value with perfect calibration (max across all decision thresholds)
        - 'rational_user' represents what a user would achieve by using the forecast probability
          directly as their decision threshold (only valid when thresholds equal
          cost-loss ratios). This assumes the forecasts are probabilistically reliable.
        - Negative REV values can occur with small samples due to random chance, or when
          forecasts genuinely perform worse than climatology
        - Validation (check_args=True) forces computation of Dask arrays. For Dask
          workflows, set check_args=False and ensure inputs are valid.

    References:
        Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble
        prediction system. Quarterly Journal of the Royal Meteorological Society, 126(563),
        649-667. https://doi.org/10.1002/qj.49712656313

    Examples:
        Calculate REV for binary forecasts:

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import relative_economic_value
        >>> fcst = xr.DataArray([0, 1, 1, 0, 1], dims=['time'])
        >>> obs = xr.DataArray([0, 1, 0, 0, 1], dims=['time'])
        >>> cost_loss_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        >>> rev = relative_economic_value(fcst, obs, cost_loss_ratios)

        Calculate REV for probabilistic forecasts with maximum value:

        >>> fcst_prob = xr.DataArray([0.2, 0.8, 0.6, 0.1, 0.9], dims=['time'])
        >>> thresholds = [0.3, 0.5, 0.7]
        >>> result = relative_economic_value(
        ...     fcst_prob, obs, cost_loss_ratios,
        ...     threshold=thresholds,
        ...     derived_metrics=['maximum']
        ... )
        >>> # result is a Dataset with 'maximum' variable

    See Also:
        - :py:func:`scores.probability.brier_score`
        - :py:func:`scores.probability.brier_score_for_ensemble`
        - :py:func:`scores.categorical.probability_of_detection`
        - :py:func:`scores.categorical.probability_of_false_detection`
        - :py:func:`scores.processing.binary_discretise`
    """

    if isinstance(fcst, xr.Dataset) and isinstance(obs, xr.Dataset):
        raise ValueError("Both fcst and obs cannot be Datasets. Convert one to a DataArray or calculate separately.")

    if isinstance(weights, xr.Dataset):
        raise ValueError("Weights cannot be Datasets. Convert to a DataArray or calculate separately.")

    # Input validation
    if check_args:
        _validate_rev_inputs(
            fcst,
            obs,
            cost_loss_ratios,
            threshold,
            threshold_dim,
            cost_loss_dim,
            weights,
            derived_metrics,
            threshold_outputs,
        )

    if isinstance(fcst, xr.Dataset):
        result_dict = {}
        for var in fcst.data_vars:
            result_dict[var] = relative_economic_value(
                fcst[var],
                obs,
                cost_loss_ratios,
                threshold=threshold,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                threshold_dim=threshold_dim,
                cost_loss_dim=cost_loss_dim,
                derived_metrics=derived_metrics,
                threshold_outputs=threshold_outputs,
                check_args=False,  # Already validated
            )
        return xr.Dataset(result_dict)

    if isinstance(obs, xr.Dataset):
        result_dict = {}
        for var in obs.data_vars:
            result_dict[var] = relative_economic_value(
                fcst,
                obs[var],
                cost_loss_ratios,
                threshold=threshold,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                threshold_dim=threshold_dim,
                cost_loss_dim=cost_loss_dim,
                derived_metrics=derived_metrics,
                threshold_outputs=threshold_outputs,
                check_args=False,  # Already validated
            )
        return xr.Dataset(result_dict)

    # Handle cost-loss ratios
    if isinstance(cost_loss_ratios, (float, int)):
        cost_loss_ratios = [cost_loss_ratios]
    cost_loss_xr = xr.DataArray(cost_loss_ratios, dims=[cost_loss_dim], coords={cost_loss_dim: cost_loss_ratios})

    # Determine dimensions for aggregation
    weights_dims = weights.dims if weights is not None else None
    dims_to_reduce = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    # Handle probabilistic vs binary forecasts
    if threshold is not None:
        # Probabilistic forecast: discretize at multiple thresholds
        if isinstance(threshold, (float, int)):
            threshold = [threshold]

        # Discretize forecasts at each threshold, adding a threshold dim
        binary_fcst = binary_discretise(fcst, threshold, ">=")

        # Rename the threshold dimension if needed
        if "threshold" in binary_fcst.dims and threshold_dim != "threshold":
            binary_fcst = binary_fcst.rename({"threshold": threshold_dim})

        # Calculate REV for each threshold
        # The threshold dimension should be PRESERVED in output
        rev = _calculate_rev_core(
            binary_fcst, obs, cost_loss_xr, dims_to_reduce=dims_to_reduce, weights=weights, cost_loss_dim=cost_loss_dim
        )

        # Post-process for derived metrics or threshold outputs
        if derived_metrics or threshold_outputs:
            result = _create_output_dataset(
                rev, threshold, cost_loss_ratios, derived_metrics, threshold_outputs, threshold_dim, cost_loss_dim
            )
            return result
        return rev

    # Assume we already have a binary set of forecasts
    rev = _calculate_rev_core(
        fcst,
        obs,
        cost_loss_xr,
        dims_to_reduce=dims_to_reduce,
        weights=weights,
    )
    return rev
