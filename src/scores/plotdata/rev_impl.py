"""
Relative Economic Value metrics for forecast evaluation
"""

import warnings
from collections.abc import Sequence
from typing import Optional, Union, cast

import numpy as np
import xarray as xr

from scores.categorical import probability_of_detection, probability_of_false_detection
from scores.processing import aggregate, binary_discretise, broadcast_and_match_nan
from scores.typing import FlexibleDimensionTypes, XarrayLike, all_same_xarraylike
from scores.utils import check_binary, check_weights, gather_dimensions

REV_DASK_MESSAGE = """
"REV relies on xarray methods not yet implemented in Dask. Calling compute() on input arrays.
Alternatively, use check_args=False to skip the incompatible code and avoid the .compute() call."
"""

# INPUT VALIDATION


def _validate_dimensions(
    fcst: XarrayLike,
    obs: XarrayLike,
    weights: Optional[xr.DataArray],
    threshold_dim: str,
    cost_loss_dim: str,
) -> None:
    """Check for dimension conflicts in inputs."""
    inputs = [("fcst", fcst), ("obs", obs)]
    if weights is not None:
        inputs.append(("weights", weights))

    for dim_name in (threshold_dim, cost_loss_dim):
        for input_name, input_data in inputs:
            if dim_name in input_data.dims:
                raise ValueError(f"'{dim_name}' cannot be a dimension in {input_name}")


def _validate_thresholds(
    probability_thresholds: Optional[Union[float, Sequence[float]]],
    probability_threshold_outputs: Optional[Sequence[float]],
) -> None:
    """Validate probability_thresholds and probability_threshold_outputs configuration."""
    if probability_thresholds is not None:
        try:
            check_monotonic_array(np.atleast_1d(probability_thresholds))
        except (ValueError, TypeError) as ex:
            raise type(ex)(f"for probability_thresholds, {ex}") from ex

        if probability_threshold_outputs is not None:
            thresh_array = np.atleast_1d(probability_thresholds)
            if not set(probability_threshold_outputs) <= set(thresh_array):
                raise ValueError(
                    "values in probability_threshold_outputs must be in the supplied probability_thresholds parameter"
                )
    else:
        if probability_threshold_outputs:
            raise ValueError(
                "probability_threshold_outputs can only be used when probability_thresholds parameter is provided"
            )


def _validate_cost_loss_ratios(cost_loss_ratios: Union[float, Sequence[float]]) -> None:
    """Validate cost-loss ratio values are in [0,1] and monotonically increasing."""

    try:
        check_monotonic_array(np.atleast_1d(cost_loss_ratios))
    except (ValueError, TypeError) as ex:
        raise type(ex)(f"for cost_loss_ratios, {ex}") from ex


def _validate_forecasts(
    fcst: XarrayLike,
    probability_thresholds: Optional[Union[float, Sequence[float]]],
) -> None:
    """
    Validate forecast values and probability_thresholds configuration.

    Raises ValueError if the probability_thresholds is provided but forecast is not
    between 0 and 1 or probability_thresholds is None but the forecast is not 0, 1 or NaN.

    """

    # Get Data Stats (Min/Max)
    if isinstance(fcst, xr.Dataset):
        fcst_min = min(var.min().item() for var in fcst.data_vars.values())
        fcst_max = max(var.max().item() for var in fcst.data_vars.values())
    else:
        fcst_min = fcst.min().item()
        fcst_max = fcst.max().item()

    # Validate based on configuration
    if probability_thresholds is not None:
        # Probabilistic forecasts: validate range
        if fcst_min < 0 or fcst_max > 1:
            raise ValueError("When probability_thresholds is provided, fcst must contain values between 0 and 1")
    else:
        check_binary(fcst, "fcst")


def _validate_rev_inputs(
    fcst: XarrayLike,
    obs: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    probability_thresholds: Optional[Union[float, Sequence[float]]],
    probability_threshold_dim: str,
    cost_loss_dim: str,
    weights: Optional[xr.DataArray],
    generate_equilibrium_point_rev: bool,
    probability_threshold_outputs: Optional[Sequence[float]],
) -> None:
    """
    Validate inputs for REV calculation.

    Raises ValueError if weights are datasets or one of the validations fails
    """

    if isinstance(weights, xr.Dataset):
        raise ValueError("Weights cannot be Datasets. Convert to a DataArray or calculate separately.")

    _validate_dimensions(fcst, obs, weights, probability_threshold_dim, cost_loss_dim)
    _validate_cost_loss_ratios(cost_loss_ratios)
    _validate_thresholds(probability_thresholds, probability_threshold_outputs)

    if generate_equilibrium_point_rev and probability_thresholds is None:
        raise ValueError(
            "generate_equilibrium_point_rev=True can only be used when probability_thresholds parameter is provided"
        )

    check_binary(obs, "obs")
    _validate_forecasts(fcst, probability_thresholds)

    # Weights validation
    if weights is not None:
        check_weights(weights)


# REV CALCULATION CORE


def _calculate_rev_core(
    binary_fcst: xr.DataArray,
    obs: xr.DataArray,
    cost_loss_ratios: Union[float, Sequence[float]],
    dims_to_reduce: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    cost_loss_dim: str = "cost_loss_ratio",
) -> xr.DataArray:
    """
    Core REV calculation from binary forecasts using scores library functions.
    """

    # Ensure that we're censoring off obs where forecasts are missing,
    # mainly for base rate calculations.
    binary_fcst, obs = broadcast_and_match_nan(binary_fcst, obs)

    pod = probability_of_detection(binary_fcst, obs, reduce_dims=dims_to_reduce, weights=weights, check_args=False)

    pofd = probability_of_false_detection(
        binary_fcst, obs, reduce_dims=dims_to_reduce, weights=weights, check_args=False
    )

    base_rate = calculate_base_rate(
        obs,
        reduce_dims=dims_to_reduce,
        weights=weights,
    )

    result = relative_economic_value_from_rates(
        pod=pod,
        pofd=pofd,
        base_rate=base_rate,
        cost_loss_ratios=cost_loss_ratios,
        cost_loss_dim=cost_loss_dim,
    )

    return cast(xr.DataArray, result)


def calculate_base_rate(
    obs: XarrayLike,
    *,  # Force keyword arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
) -> XarrayLike:
    """
    Calculates the climatological base rate (mean of observations).

    Handles weighted and unweighted means, including the case where weights
    don't span all dimensions being reduced.

    Args:
        obs: An array containing binary values (typically {0, 1, np.nan})
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating base rates. All other dimensions will be preserved. As a
            special case, 'all' will reduce all dimensions. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating base rates. All other dimensions will be reduced.
            As a special case, 'all' will preserve all dimensions. Only one of
            `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        weights: An array of weights to apply (e.g., weighting a grid by latitude).
            If None, unweighted mean is calculated. Weights must be broadcastable
            to the data dimensions and should not contain negative or NaN values.

    Returns:
        A DataArray of the climatological base rate.
    """

    # Use gather_dimensions to determine which obs dimensions to reduce
    # gather_dimensions will validate reduce_dims/preserve_dims automatically.
    dims_to_reduce = gather_dimensions(
        fcst_dims=(),  # Not needed for base rates
        obs_dims=obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights_dims=weights.dims if weights is not None else None,
    )
    base_rate = aggregate(
        obs,
        reduce_dims=dims_to_reduce,
        weights=weights,
        method="mean",
    )

    return base_rate


def _create_output_dataset(
    rev: xr.DataArray,
    thresholds: Sequence[float],
    cost_loss_ratios: Sequence[float],
    generate_maximum_rev: bool,
    generate_equilibrium_point_rev: bool,
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
        generate_maximum_rev: If True, include maximum REV across all thresholds
        generate_equilibrium_point_rev: If True, include the equilibrium point REV
        threshold_outputs: Specific thresholds to extract
        threshold_dim: Name of threshold dimension
        cost_loss_dim: Name of cost-loss ratio dimension

    Returns:
        Dataset with requested outputs

    Raises:
        ValueError: If generate_equilibrium_point_rev is True but thresholds don't match cost_loss_ratios
    """
    threshold_outputs = [] if threshold_outputs is None else list(threshold_outputs)

    result = xr.Dataset(attrs=rev.attrs)

    if generate_maximum_rev:
        result["maximum"] = rev.max(dim=threshold_dim)

    if generate_equilibrium_point_rev:  # pragma: no cover
        # Check that thresholds match cost_loss_ratios exactly
        if list(thresholds) != list(cost_loss_ratios):
            raise ValueError(
                "Can only use generate_equilibrium_point_rev=True if thresholds and cost_loss_ratios are identical"
            )
        # Extract diagonal where threshold == cost_loss_ratio
        coords = xr.DataArray(cost_loss_ratios, dims=[cost_loss_dim])
        result["equilibrium_point"] = rev.sel({threshold_dim: coords, cost_loss_dim: coords})

    # Add threshold-specific outputs
    for thresh in threshold_outputs:
        # Use string formatting that's valid for variable names
        var_name = f"threshold_{thresh}".replace(".", "_")
        result[var_name] = rev.sel({threshold_dim: thresh}).drop_vars(threshold_dim)

    return result


def check_monotonic_array(array: Union[Sequence[float], np.ndarray]) -> None:
    """
    Checks array values are in range [0, 1] and monotonically increasing.
    """
    try:
        np_array = np.array(array, dtype=float)
    except Exception as ex:
        raise TypeError("could not convert array into a numpy ndarray of floats") from ex

    if np_array.ndim != 1:
        raise ValueError("array must be one-dimensional")

    if np_array.max() > 1 or np_array.min() < 0:
        raise ValueError("array values should be between 0 and 1.")

    if len(np_array) > 1 and not (np_array[1:] - np_array[:-1] >= 0).all():
        raise ValueError("the supplied array is not monotonically increasing.")


# PUBLIC API


def relative_economic_value_from_rates(
    pod: XarrayLike,
    pofd: XarrayLike,
    base_rate: XarrayLike,
    cost_loss_ratios: Union[float, Sequence[float]],
    cost_loss_dim: str = "cost_loss_ratio",
) -> XarrayLike:
    """
    Calculates Relative Economic Value (REV) from pre-computed detection rates.

    REV measures the economic benefit of using forecasts compared to a
    baseline strategy (based on base rates), relative to perfect forecasts.
    This function computes REV directly from probability of detection (POD),
    probability of false detection (POFD), and base rates.

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
        base_rate: Climatological frequency of the event (base rate). Values should
            be between 0 and 1, representing the proportion of time the event occurs.
        cost_loss_ratios: Cost-loss ratio(s) at which to calculate REV. Must be
            strictly monotonically increasing values between 0 and 1. Can be a single
            float or sequence of floats. The cost-loss ratio represents the ratio of
            the cost of taking protective action to the loss incurred if the event
            occurs without protection.
        cost_loss_dim: Name of the cost-loss ratio dimension in output. Default is
            'cost_loss_ratio'. Must not exist as a dimension in any input array.

    Returns:
        xarray.DataArray or xarray.Dataset: REV values with dimensions from broadcasting the
            input arrays, plus an additional 'cost_loss_ratio' dimension with coordinates
            matching the supplied cost-loss ratios. Returns a Dataset if 'pod' and 'pofd' are Datasets,
            otherwise returns a DataArray.

    Raises:
        ValueError: If cost_loss_ratios is not strictly monotonically increasing,
            is not one-dimensional, or contains values outside [0, 1].
        ValueError: If 'cost_loss_ratio' dimension already exists in any input array.
        TypeError: If 'pod' and 'pofd' are not both xarray DataArrays or both xarray Datasets.

    Notes:
        - REV = 1 indicates perfect forecast value (as good as perfect information)
        - REV = 0 indicates no value over a baseline strategy based on base rates
        - REV < 0 indicates the forecast performs worse than the baseline strategy
        - This function is typically called internally by `relative_economic_value()`
          after computing POD and POFD from forecasts and observations

    References:
        - Richardson, D. S. (2000). Skill and relative economic value of the ECMWF
          ensemble prediction system. Quarterly Journal of the Royal Meteorological
          Society, 126(563), 649-667. https://doi.org/10.1002/qj.49712656313

    Examples:
        Calculate REV from pre-computed rates:

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.plotdata import relative_economic_value_from_rates

        >>> # Using pre-computed detection rates
        >>> pod = xr.DataArray([0.8, 0.6, 0.4], dims=["threshold"])
        >>> pofd = xr.DataArray([0.2, 0.1, 0.05], dims=["threshold"])
        >>> base_rate = xr.DataArray(0.3)  # 30% base rate
        >>> cost_loss_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

        >>> relative_economic_value_from_rates(
        ...     pod, pofd, base_rate, cost_loss_ratios
        ... )
        <xarray.DataArray (cost_loss_ratio: 5, threshold: 3)> Size: 120B
        array([[ 0.02857143, -0.64285714, -1.36428571],
            [ 0.6       ,  0.5       ,  0.35      ],
            [ 0.33333333,  0.36666667,  0.28333333],
            [-0.28888889,  0.05555556,  0.12777778],
            [-3.4       , -1.5       , -0.65      ]])
        Coordinates:
        * cost_loss_ratio  (cost_loss_ratio) float64 40B 0.1 0.3 0.5 0.7 0.9
        Dimensions without coordinates: threshold
        >>> # Perfect forecast scenario (returns values close to 1)
        >>> pod_perfect = xr.DataArray(1.0)
        >>> pofd_perfect = xr.DataArray(0.0)
        >>> base_rate = xr.DataArray(0.5)

        >>> relative_economic_value_from_rates(
        ...     pod_perfect,
        ...     pofd_perfect,
        ...     base_rate,
        ...     cost_loss_ratios=[0.5],
        ... )
        <xarray.DataArray (cost_loss_ratio: 1)> Size: 8B
        array([1.])
        Coordinates:
        * cost_loss_ratio  (cost_loss_ratio) float64 8B 0.5

    See Also:
        - :py:func:`scores.plotdata.relative_economic_value`
        - :py:func:`scores.categorical.probability_of_detection`
        - :py:func:`scores.categorical.probability_of_false_detection`
    """

    # if pod.chunks is not None or pofd.chunks is not None or base_rate.chunks is not None:
    #     warnings.warn(REV_DASK_MESSAGE)
    #     pod = pod.compute()
    #     obs = obs.compute()
    #     base_rate = base_rate.compute()

    if not all_same_xarraylike([pod, pofd]):
        raise TypeError("Both pod and pofd must be either xarray DataArrays or xarray Datasets.")

    if cost_loss_dim in (set(pod.dims) | set(pofd.dims) | set(base_rate.dims)):
        raise ValueError(f"dimension '{cost_loss_dim}' must not be in input data")

    # Handle Dataset inputs by applying to each variable
    if isinstance(pod, xr.Dataset):
        result_dict = {}
        for var in pod.data_vars:
            result_dict[var] = relative_economic_value_from_rates(
                pod[var],
                pofd[var] if isinstance(pofd, xr.Dataset) else pofd,
                (base_rate[var] if isinstance(base_rate, xr.Dataset) else base_rate),
                cost_loss_ratios,
                cost_loss_dim=cost_loss_dim,
            )
        return xr.Dataset(result_dict)

    alphas = xr.DataArray(
        cost_loss_ratios,
        dims=[cost_loss_dim],
        coords={cost_loss_dim: cost_loss_ratios},
    )

    obar, alphas = xr.broadcast(base_rate, alphas)
    climatological_term = xr.where(obar < alphas, obar, alphas)

    # calculate the relative economic value (equation 8)

    rev = ((climatological_term) - (pofd * alphas * (1 - obar)) + (pod * obar * (1 - alphas)) - obar) / (
        climatological_term - obar * alphas
    )

    # tidy up floating point infinities - necessary because you can get
    # near-zero in the denominator for alpha=0 or alpha=1
    rev = rev.where(~np.isinf(rev))

    return rev


def relative_economic_value(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keyword arguments to be keyword-only
    cost_loss_ratios: Optional[Union[float, Sequence[float]]] = None,
    probability_thresholds: Optional[Union[float, Sequence[float]]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    probability_threshold_dim: str = "probability_threshold",
    cost_loss_dim: str = "cost_loss_ratio",
    generate_maximum_rev: bool = False,
    generate_equilibrium_point_rev: bool = False,
    probability_threshold_outputs: Optional[Sequence[float]] = None,
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
            the loss incurred if the event occurs without protection. If None, assumes
            each percentile between 0 to 1, i.e. 0.01, 0.02...0.98, 0.99.
        probability_thresholds: The minimum probability of an event occurring that implies
            a decision would be taken. A sequence of probability thresholds may be
            provided. Each threshold converts forecasts to 1 where fcst >= probability_threshold,
            0 otherwise. Must be strictly monotonically increasing values between 0 and 1.
            If None, assumes fcst is already binary (0 or 1).
        reduce_dims: Dimensions to reduce when calculating REV. All other dimensions
            will be preserved. Cannot be used with preserve_dims.
        preserve_dims: Dimensions to preserve when calculating REV. All other dimensions
            will be reduced. As a special case, 'all' preserves all dimensions. Cannot
            be used with reduce_dims.
        weights: Optional array of weights for weighted averaging. Must be broadcastable
            to the data dimensions and cannot contain negative or NaN values. Weights
            need not cover all dimensions being reduced; unweighted averaging is applied
            to dimensions not in weights.
        probability_threshold_dim: Name of the probability_threshold dimension in output.
            Default is 'probability_threshold'.
            Must not exist as a dimension in fcst or obs.
        cost_loss_dim: Name of the cost-loss ratio dimension in output. Default is
            'cost_loss_ratio'. Must not exist as a dimension in fcst or obs.
        generate_maximum_rev: If True, include the maximum REV across all thresholds
            (the envelope of value curves, also called "potential value") as a variable
            in the output Dataset. Only used when threshold is provided.
        generate_equilibrium_point_rev: If True, include the equilibrium point REV as
            a variable in the output Dataset. The equilibrium point is the REV value for
            each user whose decision threshold equals their cost-loss ratio — i.e. the
            point where the decision threshold and cost-loss ratio are in balance. Requires
            thresholds and cost_loss_ratios to be identical. Only used when threshold
            is provided.
        probability_threshold_outputs: Optional list of specific probability_thresholds values to extract as
            separate variables in the output Dataset. Values must exist in the probability_thresholds
            parameter. Only used when probability_thresholds is provided.
        check_args: If True (default), validates input arguments.

    Returns:
        XarrayLike:
            - If generate_maximum_rev, generate_equilibrium_point_rev, or probability_threshold_outputs
              is specified: xr.Dataset with data variables for each requested output
            - If probability_thresholds is provided: xr.DataArray with dimensions from
              reduce_dims/preserve_dims, plus cost_loss_dim and probability_threshold_dim
            - If probability_thresholds is None: xr.DataArray with dimensions from
              reduce_dims/preserve_dims, plus cost_loss_dim

    Raises:
        ValueError: If fcst contains probabilistic values but probability_thresholds is None
        ValueError: If fcst contains values outside [0, 1] when probability_thresholds is provided
        ValueError: If fcst contains values other than 0 or 1 when probability_thresholds is None
        ValueError: If obs contains values other than 0 or 1
        ValueError: If cost_loss_ratios not strictly monotonically increasing or not in [0, 1]
        ValueError: If probability_thresholds values not strictly monotonically increasing or not in [0, 1]
        ValueError: If probability_threshold_dim or cost_loss_dim already exist in fcst or obs
        ValueError: If both reduce_dims and preserve_dims are specified
        ValueError: If probability_threshold_outputs values are not in probability_thresholds parameter
        ValueError: If generate_equilibrium_point_rev=True but thresholds don't match cost_loss_ratios

    Notes:
        - REV = 1 indicates perfect forecast value (as good as perfect information)
        - REV = 0 indicates no value over a baseline strategy based on historical base rates
        - REV < 0 indicates the forecast is worse than the baseline strategy
        - generate_maximum_rev represents the maximum achievable value with perfect calibration
          (max REV across all decision thresholds)
        - generate_equilibrium_point_rev represents what a user would achieve by using the
          forecast probability directly as their decision threshold (only valid when thresholds
          equal cost-loss ratios). This assumes the forecasts are probabilistically reliable.
        - Negative REV values can occur with small samples due to random chance, or when
          forecasts genuinely perform worse than the baseline strategy.

    References:
        - Richardson, D. S. (2000). Skill and relative economic value of the ECMWF ensemble
          prediction system. Quarterly Journal of the Royal Meteorological Society, 126(563),
          649-667. https://doi.org/10.1002/qj.49712656313

    Examples:
        Calculate REV for binary forecasts:

        >>> import xarray as xr
        >>> from scores.plotdata import relative_economic_value

        >>> fcst = xr.DataArray([0, 1, 1, 0, 1], dims=["time"])
        >>> obs = xr.DataArray([0, 1, 0, 0, 1], dims=["time"])
        >>> cost_loss_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

        >>> rev = relative_economic_value(
        ...     fcst, obs, cost_loss_ratios=cost_loss_ratios
        ... )

        Calculate REV for probabilistic forecasts with maximum value:

        >>> fcst_prob = xr.DataArray([0.2, 0.8, 0.6, 0.1, 0.9], dims=["time"])
        >>> thresholds = [0.3, 0.5, 0.7]
        >>> relative_economic_value(
        ...     fcst_prob,
        ...     obs,
        ...     cost_loss_ratios=cost_loss_ratios,
        ...     probability_thresholds=thresholds,
        ...     generate_maximum_rev=True,
        ... )
        <xarray.Dataset> Size: 80B
        Dimensions:          (cost_loss_ratio: 5)
        Coordinates:
          * cost_loss_ratio  (cost_loss_ratio) float64 40B 0.1 0.3 0.5 0.7 0.9
        Data variables:
            maximum          (cost_loss_ratio) float64 40B 1.0 1.0 1.0 1.0 1.0...

    See Also:
        - :py:func:`scores.probability.brier_score`
        - :py:func:`scores.probability.brier_score_for_ensemble`
        - :py:func:`scores.categorical.probability_of_detection`
        - :py:func:`scores.categorical.probability_of_false_detection`
        - :py:func:`scores.processing.binary_discretise`
    """

    if check_args and fcst.chunks is not None or obs.chunks is not None:
        warnings.warn(REV_DASK_MESSAGE)
        fcst = fcst.compute()
        obs = obs.compute()

    if cost_loss_ratios is None:
        cost_loss_ratios = list(np.arange(0.01, 1.0, 0.01))

    # Input validation
    if check_args:
        _validate_rev_inputs(
            fcst,
            obs,
            cost_loss_ratios,
            probability_thresholds,
            probability_threshold_dim,
            cost_loss_dim,
            weights,
            generate_equilibrium_point_rev,
            probability_threshold_outputs,
        )

    if isinstance(fcst, xr.Dataset) and isinstance(obs, xr.Dataset):
        # Align datasets so coords match (drops non-matching coords)
        fcst_aligned, obs_aligned = xr.align(fcst, obs, join="inner")

        # Choose a separator for combined variable names
        name_sep = "__vs__"

        result_dict = {}
        # Cross-product of variables: produce one output variable per pair
        for fvar in sorted(fcst_aligned.data_vars):
            for ovar in sorted(obs_aligned.data_vars):
                out_name = f"{fvar}{name_sep}{ovar}"
                # Recurse into scalar-DataArray path; pass check_args=False since
                # we already validated at top-level
                result_dict[out_name] = relative_economic_value(
                    fcst_aligned[fvar],
                    obs_aligned[ovar],
                    cost_loss_ratios=cost_loss_ratios,
                    probability_thresholds=probability_thresholds,
                    reduce_dims=reduce_dims,
                    preserve_dims=preserve_dims,
                    weights=weights,
                    probability_threshold_dim=probability_threshold_dim,
                    cost_loss_dim=cost_loss_dim,
                    generate_maximum_rev=generate_maximum_rev,
                    generate_equilibrium_point_rev=generate_equilibrium_point_rev,
                    probability_threshold_outputs=probability_threshold_outputs,
                    check_args=False,  # Already validated
                )

        result = xr.Dataset(result_dict)
        return result

    if isinstance(fcst, xr.Dataset):
        result_dict = {}
        for var in fcst.data_vars:
            result_dict[var] = relative_economic_value(
                fcst[var],
                obs,
                cost_loss_ratios=cost_loss_ratios,
                probability_thresholds=probability_thresholds,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                probability_threshold_dim=probability_threshold_dim,
                cost_loss_dim=cost_loss_dim,
                generate_maximum_rev=generate_maximum_rev,
                generate_equilibrium_point_rev=generate_equilibrium_point_rev,
                probability_threshold_outputs=probability_threshold_outputs,
                check_args=False,  # Already validated
            )
        result = xr.Dataset(result_dict)
        return result

    if isinstance(obs, xr.Dataset):
        result_dict = {}
        for var in obs.data_vars:
            result_dict[var] = relative_economic_value(
                fcst,
                obs[var],
                cost_loss_ratios=cost_loss_ratios,
                probability_thresholds=probability_thresholds,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
                probability_threshold_dim=probability_threshold_dim,
                cost_loss_dim=cost_loss_dim,
                generate_maximum_rev=generate_maximum_rev,
                generate_equilibrium_point_rev=generate_equilibrium_point_rev,
                probability_threshold_outputs=probability_threshold_outputs,
                check_args=False,  # Already validated
            )
        result = xr.Dataset(result_dict)
        return result

    # Handle cost-loss ratios
    if isinstance(cost_loss_ratios, (float, int)):
        cost_loss_ratios = [cost_loss_ratios]

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
    if probability_thresholds is not None:
        # Probabilistic forecast: discretize at multiple thresholds
        if isinstance(probability_thresholds, (float, int)):
            probability_thresholds = [probability_thresholds]

        # Discretize forecasts at each threshold, adding a threshold dim
        binary_fcst = cast(xr.DataArray, binary_discretise(fcst, probability_thresholds, ">="))

        # Rename the threshold dimension if needed
        if probability_threshold_dim != "threshold":
            binary_fcst = binary_fcst.rename({"threshold": probability_threshold_dim})

        # Calculate REV for each threshold
        # The threshold dimension should be PRESERVED in output
        rev = _calculate_rev_core(
            binary_fcst,
            obs,
            cost_loss_ratios,
            dims_to_reduce=dims_to_reduce,
            weights=weights,
            cost_loss_dim=cost_loss_dim,
        )

        # Post-process for derived metrics or threshold outputs
        if generate_maximum_rev or generate_equilibrium_point_rev or probability_threshold_outputs:
            result = _create_output_dataset(
                rev,
                probability_thresholds,
                cost_loss_ratios,
                generate_maximum_rev,
                generate_equilibrium_point_rev,
                probability_threshold_outputs,
                probability_threshold_dim,
                cost_loss_dim,
            )
            return result

        return rev

    # Assume we already have a binary set of forecasts
    rev = _calculate_rev_core(
        fcst,
        obs,
        cost_loss_ratios,
        dims_to_reduce=dims_to_reduce,
        weights=weights,
    )

    return rev
