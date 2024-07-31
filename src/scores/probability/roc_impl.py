"""
Implementation of Reciever Operating Characteristic (ROC) calculations
"""

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np
import xarray as xr

from scores.categorical import probability_of_detection, probability_of_false_detection
from scores.processing import binary_discretise
from scores.utils import gather_dimensions

# trapz was deprecated in numpy 2.0, but trapezoid was not backported to
# earlier versions. As numpy 2.0 contains some API changes, `scores`
# will try to support both interchangeably for the time being
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # pragma: no cover # tested manually for old numpy versions


def roc_curve_data(  # pylint: disable=too-many-arguments
    fcst: xr.DataArray,
    obs: xr.DataArray,
    thresholds: Iterable[float],
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[xr.DataArray] = None,
    check_args: bool = True,
) -> xr.Dataset:
    """
    Calculates data required for plotting a Receiver (Relative) Operating Characteristic (ROC)
    curve, including the area under the curve (AUC). The ROC curve is used as a way to measure
    the discrimination ability of a particular forecast.

    The AUC is the probability that the forecast probability of a random event is higher
    than the forecast probability of a random non-event.

    Args:
        fcst: An array of probabilistic forecasts for a binary event in the range [0, 1].
        obs: An array of binary values where 1 is an event and 0 is a non-event.
        thresholds: Monotonic increasing values between 0 and 1, the thresholds at and
          above which to convert the probabilistic forecast to a value of 1 (an 'event')
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the ROC curve data. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of `reduce_dims` and `preserve_dims` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating ROC curve data. All other dimensions will be reduced.
            As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the values will be
            the ROC curve at each point (i.e. single-value comparison
            against observed) for each threshold, and the forecast and observed dimensions
            must match precisely. Only one of `reduce_dims` and `preserve_dims` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom).
        check_args: Checks if `obs` data only contains values in the set
            {0, 1, np.nan}. You may want to skip this check if you are sure about your
            input data and want to improve the performance when working with dask.

    Returns:
        An xarray.Dataset with data variables:

        - 'POD' (the probability of detection)
        - 'POFD' (the probability of false detection)
        - 'AUC' (the area under the ROC curve)

        `POD` and `POFD` have dimensions `dims` + 'threshold', while `AUC` has
        dimensions `dims`.

    Raises:
        ValueError: if `fcst` contains values outside of the range [0, 1]
        ValueError: if `obs` contains non-nan values not in the set {0, 1}
        ValueError: if 'threshold' is a dimension in `fcst`.
        ValueError: if values in `thresholds` are not monotonic increasing or are outside
          the range [0, 1]


    Notes:
        The probabilistic `fcst` is converted to a deterministic forecast
        for each threshold in `thresholds`. If a value in `fcst` is greater
        than or equal to the threshold, then it is converted into a
        'forecast event' (fcst = 1), and a 'forecast non-event' (fcst = 0)
        otherwise. The probability of detection (POD) and probability of false
        detection (POFD) are calculated for the converted forecast. From the
        POD and POFD data, the area under the ROC curve is calculated.

        Ideally concave ROC curves should be generated rather than traditional
        ROC curves.

    """
    if check_args:
        if fcst.max().item() > 1 or fcst.min().item() < 0:
            raise ValueError("`fcst` contains values outside of the range [0, 1]")

        if np.max(thresholds) > 1 or np.min(thresholds) < 0:  # type: ignore
            raise ValueError("`thresholds` contains values outside of the range [0, 1]")

        if not np.all(np.array(thresholds)[1:] >= np.array(thresholds)[:-1]):
            raise ValueError("`thresholds` is not monotonic increasing between 0 and 1")

    # make a discrete forecast for each threshold in thresholds
    # discrete_fcst has an extra dimension 'threshold'
    discrete_fcst = binary_discretise(fcst, thresholds, ">=")

    all_dims = set(fcst.dims).union(set(obs.dims))
    final_reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    final_preserve_dims = all_dims - set(final_reduce_dims)  # type: ignore
    auc_dims = () if final_preserve_dims is None else tuple(final_preserve_dims)
    final_preserve_dims = auc_dims + ("threshold",)  # type: ignore[assignment]

    pod = probability_of_detection(
        discrete_fcst, obs, preserve_dims=final_preserve_dims, weights=weights, check_args=check_args
    )

    pofd = probability_of_false_detection(
        discrete_fcst, obs, preserve_dims=final_preserve_dims, weights=weights, check_args=check_args
    )

    # Need to ensure ordering of dims is consistent for xr.apply_ufunc
    pod = pod.transpose(*final_preserve_dims)
    pofd = pofd.transpose(*final_preserve_dims)

    auc = -1 * xr.apply_ufunc(
        np.trapezoid,
        pod,
        pofd,
        input_core_dims=[pod.dims, pofd.dims],  # type: ignore
        output_core_dims=[auc_dims],
        dask="parallelized",
    )

    return xr.Dataset({"POD": pod, "POFD": pofd, "AUC": auc})
