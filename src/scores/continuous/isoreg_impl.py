"""
An implementation of isotonic regression using the pool adjacent violators (PAV)
algorithm. In the context of forecast verification, the regression explanatory variable
is the forecast and the response variable is the observation. Forecasts and observations
are assumed to be real-valued quantities.

Isotonic regression exposes conditional forecast biases. Confidence bands for the
regression fit can also be returned. This implementation includes option to specify what
functional the forecast is targeting, so that appropriate regression can be performed on
different types of real-valued forecasts, such as those which are a quantile of the
predictive distribution.
"""

from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy import interpolate
from sklearn.isotonic import IsotonicRegression


def isotonic_fit(  # pylint: disable=too-many-locals, too-many-arguments
    fcst: Union[np.ndarray, xr.DataArray],
    obs: Union[np.ndarray, xr.DataArray],
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[Union[np.ndarray, xr.DataArray]] = None,
    functional: Optional[Literal["mean", "quantile"]] = "mean",
    bootstraps: Optional[int] = None,
    quantile_level: Optional[float] = None,
    solver: Optional[Union[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray], float]]] = None,
    confidence_level: Optional[float] = 0.9,
    min_non_nan: Optional[int] = 1,
    report_bootstrap_results: Optional[bool] = False,
) -> dict:
    """
    Calculates an isotonic regression fit to observations given forecasts as the
    explanatory values. Optionally returns confidence bands on the fit via bootstrapping.
    Forecasts and observations are scalar-valued (i.e, integers, floats or NaN).

    If forecasts target the mean or a quantile functional, then the user can select that
    functional. Otherwise the user needs to supply a `solver` function

        - of one variable (if no weights are supplied), or
        - of two variables (if weights are supplied).

    The solver function is used to find an optimal regression fit in each block, subject
    to a non-decreasing monotonicity constraint, using the pool adjacent violators (PAV)
    algorithm.

    An example of a solver function of one variable is obs -> np.mean(obs)
    An example of a solver function of two variables is (obs, weight) -> np.average(obs, weights=weight).

    Ties (where one forecast value has more than one observation value) are handled as follows.
    Forecasts are sorted in ascending order, and observations sorted to match. Where ties occur,
    observations are sorted in descending order within each subset of ties. Isotonic regression
    is then performed on this sorted observation sequence.

    This implementation uses sklearn.isotonic.IsotonicRegression when `functional="mean"`.

    Args:
        fcst: np.array or xr.DataArray of forecast (or explanatory) values. Values must
            be float, integer or NaN.
        obs: np.array or xr.DataArray of corresponding observation (or response) values.
            Must be the same shape as `fcst`. Values must be float, integer or NaN.
        weight: positive weights to apply to each forecast-observation pair. Must be the
            same shape as `fcst`, or `None` (which is equivalent to applying equal weights).
        functional: Functional that the forecasts are targeting. Either "mean" or
            "quantile" or `None`. If `None` then `solver` must be supplied. The current
            implementation for "quantile" does not accept weights. If weighted quantile
            regression is desired then the user should should supply an appropriate solver.
        bootstraps: the number of bootstrap samples to perform for calculating the
            regression confidence band. Set to `None` if a confidence band is not required.
        quantile_level: the level of the quantile functional if `functional='quantile'`.
            Must be strictly between 0 and 1.
        solver: function that accepts 1D numpy array of observations and returns
            a float. Function values give the regression fit for each block as determined
            by the PAV algorithm. If weights are supplied, the function must have a second
            argument that accepts 1D numpy array of weights of same length as the observations.
        confidence_level: Confidence level of the confidence band, strictly between 0 and 1.
            For example, a confidence level of 0.5 will calculate the confidence band by
            computing the 25th and 75th percentiles of the bootstrapped samples.
        min_non_nan: The minimum number of non-NaN bootstrapped values to calculate
            confidence bands at a particular forecast value.
        report_bootstrap_results: This specifies to whether keep the bootstrapped
            regression values in return dictionary. Default is False to keep the output
            result small.

    Returns:
        Dictionary with the following keys:

        - "unique_fcst_sorted": 1D numpy array of remaining forecast values sorted in
            ascending order, after any NaNs from `fcst`, `obs` and `weight` are removed,
            and only unique values are kept to keep the output size reasonably small.
        - "fcst_counts": 1D numpy array of forecast counts for unique values of forecast sorted
        - "regression_values": 1D numpy array of regression values corresponding to
            "unique_fcst_sorted" values.
        - "regression_func": function that returns the regression fit based on linear
            interpolation of ("fcst_sorted", "regression_values"), for any supplied
            argument (1D numpy array) of potential forecast values.
        - "bootstrap_results": in the case of `report_bootstrap_results=True`, 2D numpy
            array of bootstrapped regression values is included in return dictionary.
            Each row gives the interpolated regression values from a particular bootstrapped
            sample, evaluated at "fcst_sorted" values. If `m` is the number of bootstrap
            samples and `n = len(fcst_sorted)` then it is has shape `(m, n)`. We emphasise
            that this array corresponds to `fcst_sorted` not `unique_fcst_sorted`.
        - "confidence_band_lower_values": values of lower confidence band threshold, evaluated
            at "unique_fcst_sorted" values.
        - "confidence_band_upper_values": values of upper confidence band threshold, evaluated
            at "unique_fcst_sorted" values.
        - "confidence_band_lower_func": function that returns regression fit based on linear
            interpolation of ("fcst_sorted", "confidence_band_lower_values"), given any
            argument (1D numpy array) of potential forecast values.
        - "confidence_band_upper_func": function that returns regression fit based on linear
            interpolation of ("fcst_sorted", "confidence_band_upper_values"), given any
            argument (1D numpy array) of potential forecast values.
        - "confidence_band_levels": tuple giving the quantile levels used to calculate the
            confidence band.

    Raises:
        ValueError: if `fcst` and `obs` are np.arrays and don't have the same shape.
        ValueError: if `fcst` and `weight` are np.arrays and don't have the same shape.
        ValueError: if `fcst` and `obs` are xarray.DataArrays and don't have the same dimensions.
        ValueError: if `fcst` and `weight` are xarray.DataArrays and don't have the same dimensions.
        ValueError: if any entries of `fcst`, `obs` or `weight` are not an integer, float or NaN.
        ValueError: if there are no `fcst` and `obs` pairs after NaNs are removed.
        ValueError: if `functional` is not one of "mean", "quantile" or `None`.
        ValueError: if `quantile_level` is not strictly between 0 and 1 when `functional="quantile"`.
        ValueError: if `weight` is not `None` when `functional="quantile"`.
        ValueError: if `functional` and `solver` are both `None`.
        ValueError: if not exactly one of `functional` and `solver` is `None`.
        ValueError: if `bootstraps` is not a positive integer.
        ValueError: if `confidence_level` is not strictly between 0 and 1.

    Note: This function only keeps the unique values of `fcst_sorted` to keep the volume of
        the return dictionary small. The forecast counts is also included, so users can it to
        create forecast histogram (usually displayed in the reliability diagrams).

    References
        - de Leeuw, Hornik and Mair. "Isotone Optimization in R: Pool-Adjacent-Violators Algorithm (PAVA)
          and Active Set Methods", Journal of Statistical Software, 2009.
        - Dimitriadis, Gneiting and Jordan. "Stable reliability diagrams for probabilistic classifiers",
          PNAS, Vol. 118 No. 8, 2020. Available at https://www.pnas.org/doi/10.1073/pnas.2016191118
        - Jordan, Mühlemann, and Ziegel. "Optimal solutions to the isotonic regression problem",
          2020 (version 2), available on arxiv at https://arxiv.org/abs/1904.04761
    """

    if isinstance(fcst, xr.DataArray):
        fcst, obs, weight = _xr_to_np(fcst, obs, weight)  # type: ignore
    # now fcst, obs and weight (unless None) are np.arrays

    _iso_arg_checks(
        fcst,  # type: ignore
        obs,  # type: ignore
        weight=weight,  # type: ignore
        functional=functional,
        quantile_level=quantile_level,
        solver=solver,
        bootstraps=bootstraps,
        confidence_level=confidence_level,
    )
    fcst_tidied, obs_tidied, weight_tidied = _tidy_ir_inputs(fcst, obs, weight=weight)  # type: ignore
    y_out = _do_ir(
        fcst_tidied,
        obs_tidied,
        weight=weight_tidied,
        functional=functional,
        quantile_level=quantile_level,
        solver=solver,
    )

    # calculate the fitting function
    ir_func = interpolate.interp1d(fcst_tidied, y_out, bounds_error=False)

    if bootstraps is not None:
        boot_results = _bootstrap_ir(
            fcst=fcst_tidied,
            obs=obs_tidied,
            weight=weight_tidied,
            functional=functional,
            quantile_level=quantile_level,
            solver=solver,
            bootstraps=bootstraps,
        )

        lower_pts, upper_pts = _confidence_band(boot_results, confidence_level, min_non_nan)  # type: ignore

        lower_func = interpolate.interp1d(fcst_tidied, lower_pts, bounds_error=False)
        upper_func = interpolate.interp1d(fcst_tidied, upper_pts, bounds_error=False)

        confband_levels = ((1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2)  # type: ignore

    else:
        boot_results = lower_pts = upper_pts = None  # type: ignore
        lower_func = upper_func = partial(np.full_like, fill_value=np.nan)
        confband_levels = (None, None)  # type: ignore
    # To reduce the size of output dictionary, we only keep the unique values of
    # forecasts and accordingly regression and confidence band values calculate by using
    # unique forecast values. We also calculate forecast counts that can be used to create
    # the forecast histogram.
    unique_fcst_sorted, fcst_counts = np.unique(fcst_tidied, return_counts=True)
    regression_values = ir_func(unique_fcst_sorted)
    if bootstraps:
        lower_pts = lower_func(unique_fcst_sorted)
        upper_pts = upper_func(unique_fcst_sorted)
    results = {
        "fcst_sorted": unique_fcst_sorted,
        "fcst_counts": fcst_counts,
        "regression_values": regression_values,
        "regression_func": ir_func,
        "confidence_band_lower_values": lower_pts,
        "confidence_band_upper_values": upper_pts,
        "confidence_band_lower_func": lower_func,
        "confidence_band_upper_func": upper_func,
        "confidence_band_levels": confband_levels,
    }
    if report_bootstrap_results:
        results["bootstrap_results"] = boot_results

    return results


def _xr_to_np(
    fcst: xr.DataArray, obs: xr.DataArray, weight: Optional[xr.DataArray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conducts basic dimension checks to `isotonic_fit` inputs if they are xr.DataArray.
    Then converts xr.DataArray objects into corresponding numpy arrays, with
    corresponding datapoints.

    Args:
        fcst: DataArray of forecast values
        obs: DataArray of observed values
        weight: weights to apply to each forecast-observation pair. Must be the same
            shape as `fcst`, or `None`.
    Returns:
        numpy arrays of `fcst`, `obs` and `weight`.

    Raises:
        ValueError: if `fcst` and `obs` do not have the same dimension.
        ValueError: if `fcst` and `weight` do not have the same dimension.
    """
    fcst_dims = fcst.dims

    if set(fcst_dims) != set(obs.dims):
        raise ValueError("`fcst` and `obs` must have same dimensions.")

    if weight is not None:
        if set(fcst_dims) != set(weight.dims):
            raise ValueError("`fcst` and `weight` must have same dimensions.")
        merged_ds = xr.merge([fcst.rename("fcst"), obs.rename("obs"), weight.rename("weight")])
        weight = merged_ds["weight"].transpose(*fcst_dims).values  # type: ignore
    else:
        merged_ds = xr.merge([fcst.rename("fcst"), obs.rename("obs")])

    fcst = merged_ds["fcst"].transpose(*fcst_dims).values  # type: ignore
    obs = merged_ds["obs"].transpose(*fcst_dims).values  # type: ignore

    return fcst, obs, weight  # type: ignore


def _iso_arg_checks(  # pylint: disable=too-many-arguments, too-many-branches
    fcst: np.ndarray,
    obs: np.ndarray,
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[Union[np.ndarray, xr.DataArray]] = None,
    functional: Optional[str] = None,
    quantile_level: Optional[float] = None,
    solver: Optional[Union[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray], float]]] = None,
    bootstraps: Optional[int] = None,
    confidence_level: Optional[float] = None,
) -> None:
    """Raises ValueError if isotonic_fit arguments are invalid."""
    if fcst.shape != obs.shape:
        raise ValueError("`fcst` and `obs` must have same shape.")

    if not (np.issubdtype(fcst.dtype, np.integer) or np.issubdtype(fcst.dtype, np.floating)):
        raise ValueError("`fcst` must be an array of floats or integers.")

    if not (np.issubdtype(obs.dtype, np.integer) or np.issubdtype(obs.dtype, np.floating)):
        raise ValueError("`obs` must be an array of floats or integers.")

    if weight is not None:
        if fcst.shape != weight.shape:
            raise ValueError("`fcst` and `weight` must have same shape.")

        if not (np.issubdtype(weight.dtype, np.integer) or np.issubdtype(weight.dtype, np.floating)):
            raise ValueError("`weight` must be an array of floats or integers, or else `None`.")

        if np.any(weight <= 0):
            raise ValueError("`weight` values must be either positive or NaN.")

    if functional not in ["mean", "quantile", None]:
        raise ValueError("`functional` must be one of 'mean', 'quantile' or `None`.")

    if functional == "quantile" and not 0 < quantile_level < 1:  # type: ignore
        raise ValueError("`quantile_level` must be strictly between 0 and 1.")

    if functional == "quantile" and weight is not None:
        msg = "Weighted quantile isotonic regression has not been implemented. "
        msg += "Either (i) set `weight=None` or (ii) set `functional=None` and supply an appropriate "
        msg += "quantile `solver` function with two arguments."
        raise ValueError(msg)

    if functional is None and solver is None:
        raise ValueError("`functional` and `solver` cannot both be `None`.")

    if None not in (functional, solver):
        raise ValueError("One of `functional` or `solver` must be `None`.")

    if bootstraps is not None:
        if not isinstance(bootstraps, int) or bootstraps < 1:
            raise ValueError("`bootstraps` must be a positive integer.")
        if not 0 < confidence_level < 1:  # type: ignore
            raise ValueError("`confidence_level` must be strictly between 0 and 1.")


def _tidy_ir_inputs(
    fcst: np.ndarray,
    obs: np.ndarray,
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tidies array inputs to `isotonic_fit` in preparation for the regression routine.

    Arrays are flattened to 1D arrays. NaNs are jointly removed from fcst, obs, weights.
    The array data is then sorted by fcst (ascending) then by obs (descending).

    Args:
        fcst: numpy array of forecast values.
        obs: numpy array of corresponding observed values, with the same shape as `fcst`.
        weight: Either `None`, or numpy array of corresponding weights with the same
            shape as `fcst`.

    Returns:
        Tidied fcst, obs and weight arrays. If input weight is `None` then returned weight
        is `None`.

    Raises:
        ValueError: if no forecast and observation pairs remains after removing NaNs.
    """
    fcst = fcst.flatten()
    obs = obs.flatten()

    if weight is not None:
        weight = weight.flatten()

    nan_locs = np.isnan(fcst) | np.isnan(obs)

    if weight is not None:
        nan_locs = nan_locs | np.isnan(weight)

    new_fcst = fcst[~nan_locs]
    new_obs = obs[~nan_locs]

    if len(new_fcst) == 0:
        raise ValueError("No (fcst, obs) pairs remaining after NaNs removed.")

    # deal with ties in fcst: sort by fcst (ascending) then by obs (descending)
    # this ensures that ties in fcst have same regression value in PAV algorithm
    sorter = np.lexsort((-new_obs, new_fcst))
    new_fcst = new_fcst[sorter]
    new_obs = new_obs[sorter]

    if weight is None:
        new_weight = None
    else:
        new_weight = weight[~nan_locs]
        new_weight = new_weight[sorter]

    return new_fcst, new_obs, new_weight  # type: ignore


def _do_ir(  # pylint: disable=too-many-arguments
    fcst: np.ndarray,
    obs: np.ndarray,
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[np.ndarray] = None,
    functional: Optional[Literal["mean", "quantile"]] = None,
    quantile_level: Optional[float] = None,
    solver: Optional[Union[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray], float]]] = None,
) -> np.ndarray:
    """
    Returns the isotonic regression (IR) fit for specified functional or solver,
    by passing the inputs to the appropriate IR algorithm.

    The inputs `fcst`, `obs` and `weight` are 1D numpy arrays assumed to have been
    mutually tidied via `_tidy_ir_inputs`.

    Args:
        fcst: tidied array of forecast values.
        obs: tidied array of observed values.
        weight: tidied array of weights.
        functional: either "mean", "quantile" or None.
        quantile_level: float strictly between 0 and 1 if functional="quantile",
            else None.
        solver: solver function as described in docstring of `isotonic_fit`, or `None`.
            Cannot be `None` if `functional` is `None`.

    Returns:
        1D numpy array of values fitted using isotonic regression.
    """
    if functional == "mean":
        y_out = _contiguous_mean_ir(fcst, obs, weight=weight)
    elif functional == "quantile":
        y_out = _contiguous_quantile_ir(obs, quantile_level)  # type: ignore
    else:
        y_out = _contiguous_ir(obs, solver, weight=weight)  # type: ignore

    return y_out


def _contiguous_ir(
    y: np.ndarray,
    solver: Union[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray], float]],
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Given a 1D numpy array `y` of response values, returns fitted isotonic regression
    with values for each valid regression block determined by `solver`.

    This implementation uses the pool adjacent violators (PAV) algorithm, and is based on the squared loss
    implementation at sklearn._isotonic._inplace_contiguous_isotonic_regression
    https://github.com/scikit-learn/scikit-learn/blob/a6574655478a24247189236ce5f824e65ad0d369/sklearn/_isotonic.pyx
    For an animated visualisation of the algorithm, see
    https://www.r-bloggers.com/2020/05/what-is-isotonic-regression/

    Args:
        y: 1D numpy array of response (i.e., observed) values.
        solver: function which determines values of each valid regression block
            in the PAV algorithm, given response values in the block.
        weight: 1D numpy array of weights, or else `None`.
            If `weight` is not `None` then `solver` takes in two arrays as arguments
            (response values and weights). If `weight` is `None` then `solver` takes
            in a single array as argument (response values).

    """
    len_y = len(y)

    if weight is not None and len(weight) != len_y:
        raise ValueError("`y` and `weight` must have same length.")

    # y_out will be the final regression output
    y_out = y.copy()

    # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    # an active block, then target[i] := j and target[j] := i.
    target = np.arange(len_y, dtype=np.intp)

    index = 0
    while index < len_y:
        # idx_next_block is the index of beginning of the next block
        idx_next_block = target[index] + 1

        if idx_next_block == len_y:  # there are no more blocks
            break
        if (
            y_out[index] < y_out[idx_next_block]
        ):  # the regression value of the next block is greater than the active one
            index = idx_next_block  # that block becomes the active block
            continue

        while True:
            # We are within a decreasing subsequence.
            prev_y = y_out[idx_next_block]  # this is the smaller y value after the active block
            idx_next_block = target[idx_next_block] + 1  # advance index of next block
            if idx_next_block == len_y or prev_y < y_out[idx_next_block]:
                # update block indices
                target[index] = idx_next_block - 1
                target[idx_next_block - 1] = index

                # regression value for current block
                if weight is None:
                    y_out[index] = solver(y[index:idx_next_block])  # type: ignore
                else:
                    y_out[index] = solver(y[index:idx_next_block], weight[index:idx_next_block])  # type: ignore

                if index > 0:
                    # Backtrack if we can.  This makes the algorithm
                    # single-pass and ensures O(n) complexity (subject to solver complexity)
                    index = target[index - 1]
                # Otherwise, restart from the same point.
                break
    # Reconstruct the solution.
    index = 0
    while index < len_y:
        idx_next_block = target[index] + 1
        y_out[index + 1 : idx_next_block] = y_out[index]
        index = idx_next_block

    return y_out


def _contiguous_quantile_ir(y: np.ndarray, alpha: float) -> np.ndarray:
    """Performs contiguous quantile IR on tidied data y, for quantile-level alpha, with no weights."""
    return _contiguous_ir(y, partial(np.quantile, q=alpha))  # type: ignore


def _contiguous_mean_ir(
    x: np.ndarray, y: np.ndarray, *, weight: Optional[np.ndarray] = None  # Force keywords arguments to be keyword-only
) -> np.ndarray:
    """
    Performs classical (i.e. for mean functional) contiguous quantile IR on tidied data x, y.
    Uses sklearn implementation rather than supplying the mean solver function to `_contiguous_ir`,
    as it is about 4 times faster (since it is optimised for mean).
    """
    return IsotonicRegression().fit_transform(x, y, sample_weight=weight)  # type: ignore


def _bootstrap_ir(  # pylint: disable=too-many-arguments, too-many-locals
    fcst: np.ndarray,
    obs: np.ndarray,
    bootstraps: int,
    *,  # Force keywords arguments to be keyword-only
    weight: Optional[np.ndarray] = None,
    functional: Optional[Literal["mean", "quantile"]] = None,
    quantile_level: Optional[float] = None,
    solver: Optional[Union[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray], float]]] = None,
):
    """
    Gives the isotonic fits of bootstrapped samples.

    Args:
        fcst: 1D numpy array of forecast values.
        obs: 1D numpy array of observed values corresponding to `fcst`.
        bootstraps: the number of samples to bootstrap.
        weight: either `None`, or 1D numpy array of weights corresponding to `fcst`.
        functional: either `None` or the target functional (one of "mean" or "quantile").
        quantile_level: the quantile level if `functional="quantile"`.
        solver: the regression solver function if `functional=None`.

    Returns:
        2D numpy array with one row for each bootstrapped sample and one column for each `fcst` value.
        The [i,j] entry is the isotonic fit of the ith bootstrapped sample evaluated at the point
        `fcst[j]`. Evaluations are determined using interpolation when necessary. If the [i,j] entry is
        NaN then `fcst[j]` lay outside the range of the ith bootstrapped sample.
    """
    # initialise
    fc_values_count = len(fcst)
    result = np.full((bootstraps, fc_values_count), np.nan)

    for boostrap_sample_num in range(bootstraps):
        selection = np.random.randint(0, fc_values_count, fc_values_count)
        fcst_sample = fcst[selection]
        obs_sample = obs[selection]
        if weight is None:
            weight_sample = None
        else:
            weight_sample = weight[selection]

        fcst_sample, obs_sample, weight_sample = _tidy_ir_inputs(fcst_sample, obs_sample, weight=weight_sample)

        ir_results = _do_ir(
            fcst_sample,
            obs_sample,
            weight=weight_sample,
            functional=functional,
            quantile_level=quantile_level,
            solver=solver,
        )

        approximation_func = interpolate.interp1d(fcst_sample, ir_results, bounds_error=False)

        result[boostrap_sample_num] = approximation_func(fcst)

    return result


def _confidence_band(
    bootstrap_results: np.ndarray, confidence_level: float, min_non_nan: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given bootstrapped results, return the lower and upper values of the corresponding confidence band
    with specified `confidence_level`.

    Args:
        bootstrap_results: matrix output from `_bootstrap_ir`.
        confidence_level: float between 0 and 1.
        min_non_nan: the minimum number of non-NaN values a column of `bootstrap_results`
            for which the confidence band will be calculated. If the minimum is not met,
            a NaN value is returned.

    Returns:
        Tuple of two 1D numpy arrays. The first array gives is the
        `(1-confidence_level)/2`-quantile of each column of `bootstrap_results`.
        The second array gives is the `1-(1-confidence_level)/2`-quantile of each column.
    """
    sig_level = 1 - confidence_level
    upper_level = 1 - sig_level / 2
    lower_level = sig_level / 2

    # confidence band values ignoring NaNs
    upper = _nanquantile(bootstrap_results, upper_level)
    lower = _nanquantile(bootstrap_results, lower_level)

    # masking values with too many NaNs in bootstrap results
    non_nan_count = np.count_nonzero(~np.isnan(bootstrap_results), axis=0)
    upper = np.where(non_nan_count >= min_non_nan, upper, np.nan)
    lower = np.where(non_nan_count >= min_non_nan, lower, np.nan)

    return lower, upper


def _nanquantile(arr: np.ndarray, quant: float) -> np.ndarray:
    """Returns same* output as np.nanquantile but faster.

    *almost equal
    See https://github.com/numpy/numpy/issues/16575, this implementation is about 100
    times faster compared to numpy 1.24.3.

    Args:
        arr: 2D numpy array with one row for each bootstrapped sample and one column for
            each `fcst` value.
        quant: Value between 0 and 1.

    Returns:
        1D numpy array of shape `arr.shape[0]` of values at each quantile.
    """
    arr = np.copy(arr)
    valid_obs_count = np.sum(np.isfinite(arr), axis=0)
    # Replace NaN with maximum (these values will be ignored)
    if not np.isnan(arr).all():
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
    # Sort forecast values - former NaNs will move to the end
    arr = np.sort(arr, axis=0)
    result = np.zeros(shape=[arr.shape[1]])

    # Get the index of the values at the requested quantile
    desired_position = (valid_obs_count - 1) * quant
    index_floor = np.floor(desired_position).astype(np.int32)
    index_ceil = np.ceil(desired_position).astype(np.int32)
    same_index = index_floor == index_ceil

    # Linear interpolation - take the fractional part of desired position
    floor_val = arr[index_floor, np.arange(arr.shape[1])]
    floor_frac = index_ceil - desired_position
    floor_frac_val = arr[index_floor, np.arange(arr.shape[1])] * floor_frac
    ceil_frac = desired_position - index_floor
    ceil_frac_val = arr[index_ceil, np.arange(arr.shape[1])] * ceil_frac
    result = np.where(same_index, floor_val, floor_frac_val + ceil_frac_val)
    return result
