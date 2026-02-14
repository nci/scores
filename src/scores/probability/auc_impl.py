"""
This module contains methods for calculating area under curve metrics, such as the ROC AUC.
"""

import warnings
from typing import Optional

import numpy as np
import xarray as xr
from scipy.stats import rankdata

from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_binary, gather_dimensions


def _roc_auc_mann_whitney(fcst_flat: np.ndarray, obs_flat: np.ndarray) -> float:
    """Core ROC AUC computation using the Mann-Whitney U statistic.

    Computes AUC = U / (n_pos * n_neg) where U is the Mann-Whitney U statistic.
    This is equivalent to P(fcst_positive > fcst_negative) + 0.5 * P(fcst_positive == fcst_negative)
    and runs in O(n log n) time due to sorting-based ranking.

    Args:
        fcst_flat: 1-D array of forecast probabilities.
        obs_flat: 1-D array of binary observations (0 or 1).

    Returns:
        The area under the ROC curve as a float, or NaN if there are
        fewer than one positive or one negative observation.
    """
    # Mask out NaNs from both arrays jointly
    valid = ~(np.isnan(fcst_flat) | np.isnan(obs_flat))
    fcst_valid = fcst_flat[valid]
    obs_valid = obs_flat[valid]

    n_pos = np.sum(obs_valid == 1)
    n_neg = np.sum(obs_valid == 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Compute average ranks of forecast values (handles ties correctly)
    ranks = rankdata(fcst_valid, method="average")

    # Sum of ranks for the positive class
    rank_sum_pos = np.sum(ranks[obs_valid == 1])

    # Mann-Whitney U statistic
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0

    return u_stat / (n_pos * n_neg)


def roc_auc(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,  # Force keyword arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    check_args: bool = True,
) -> XarrayLike:
    """Calculates the area under the Receiver Operating Characteristic (ROC) curve.

    The ROC AUC measures the discrimination ability of a probabilistic forecast
    for a binary event. It equals the probability that the forecast value for a
    randomly chosen event is higher than the forecast value for a randomly chosen
    non-event.

    This implementation uses the Mann-Whitney U statistic, which is equivalent to
    the trapezoidal ROC AUC but runs in O(n log n) time instead of O(n × T) where
    T is the number of unique thresholds. Ties in forecast values are handled
    correctly via average ranking.

    .. math::
        \\text{AUC} = \\frac{U}{n_1 \\cdot n_0}

    where :math:`U` is the Mann-Whitney U statistic, :math:`n_1` is the number of
    positive observations, and :math:`n_0` is the number of negative observations.
    :math:`U` is calculated from the rank sum of the positive class forecasts:

    .. math::
        U = R_1 - \\frac{n_1(n_1 + 1)}{2}

    where :math:`R_1 = \\sum_{i : y_i = 1} \\text{rank}(\\hat{y}_i)` is the sum of ranks
    of the forecast values corresponding to positive observations.

    Args:
        fcst: An array of probabilistic forecasts for a binary event in the range [0, 1].
        obs: An array of binary observations where 1 is an event and 0 is a non-event.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the ROC AUC. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the ROC AUC. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the forecast and observed dimensions must match
            precisely.
        check_args: If True, checks that ``fcst`` values are in [0, 1] and
            ``obs`` values are in {0, 1, NaN}. Set to False to skip these checks
            for improved performance, especially with Dask-backed arrays.

    Returns:
        An xarray object containing ROC AUC values. The result has dimensions
        equal to the preserved dimensions. An AUC of 1.0 indicates perfect
        discrimination, 0.5 indicates no discrimination (equivalent to random
        guessing), and 0.0 indicates perfectly reversed discrimination.

    Raises:
        ValueError: if ``fcst`` contains values outside of the range [0, 1].
        ValueError: if ``obs`` contains non-NaN values not in the set {0, 1}.

    Warns:
        UserWarning: If ``fcst`` or ``obs`` is backed by a Dask array and
            ``check_args`` is True, a warning will be issued to inform the user
            that eager computation will occur for input validation.

    Notes:
        - NaN values in ``fcst`` or ``obs`` are excluded pairwise before computing
          the AUC.
        - If all observations are positive or all are negative along the
          reduced dimensions, NaN is returned for that slice.
        - This function supports Dask-backed xarray objects for lazy evaluation.

    References:
        - Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area
          under a receiver operating characteristic (ROC) curve. Radiology, 143(1),
          29-36. https://doi.org/10.1148/radiology.143.1.7063747
        - Mason, S. J. and Graham, N. E. (2002). Areas beneath the relative operating
          characteristics (ROC) and relative operating levels (ROL) curves: Statistical
          significance and interpretation. Quarterly Journal of the Royal
          Meteorological Society, 128(584), 2145-2166. https://doi.org/10.1256/003590002320603584
        - Mann, H. B. and Whitney, D. R. (1947). On a Test of Whether one of Two Random
          Variables is Stochastically Larger than the Other. The Annals of Mathematical
          Statistics, 18(1), 50-60. https://doi.org/10.1214/aoms/1177730491

    Examples:
        Calculate ROC AUC reducing all dimensions:

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import roc_auc
        >>> fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        >>> obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        >>> roc_auc(fcst, obs)
        <xarray.DataArray ()> Size: 8B
        array(1.)

        Calculate ROC AUC preserving the 'station' dimension:

        >>> fcst = xr.DataArray(
        ...     np.random.rand(3, 100),
        ...     dims=["station", "time"],
        ... )
        >>> obs = xr.DataArray(
        ...     np.random.randint(0, 2, size=(3, 100)),
        ...     dims=["station", "time"],
        ... )
        >>> result = roc_auc(fcst, obs, preserve_dims=["station"])
    """
    if check_args:
        if isinstance(fcst, (xr.DataArray, xr.Dataset)) and (
            getattr(fcst, "chunks", None) is not None or getattr(obs, "chunks", None) is not None
        ):
            warnings.warn(
                "`fcst` or `obs` is an xarray object backed by a Dask array. "
                "Input validation requires computing the min and max of the arrays "
                "which triggers immediate computation. Set `check_args=False` to avoid this.",
                UserWarning,
            )

        if isinstance(fcst, xr.Dataset):
            fcst_max = fcst.to_array().max().values.item()
            fcst_min = fcst.to_array().min().values.item()
        else:
            fcst_max = fcst.max().values.item()
            fcst_min = fcst.min().values.item()

        if fcst_max > 1 or fcst_min < 0:
            raise ValueError("`fcst` contains values outside of the range [0, 1]")

        check_binary(obs, "obs")

    reduce_dims = gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )

    reduce_dims_tuple = tuple(reduce_dims)

    # If there are no dims to reduce, return element-wise (degenerate case)
    if not reduce_dims_tuple:
        return fcst * np.nan  # AUC is undefined for a single point

    # Stack reduce dims into a single sample dimension for the core computation
    sample_dim = "__roc_auc_sample__"
    fcst_stacked = fcst.stack({sample_dim: reduce_dims_tuple})
    obs_stacked = obs.stack({sample_dim: reduce_dims_tuple})

    result = xr.apply_ufunc(
        _roc_auc_mann_whitney,
        fcst_stacked,
        obs_stacked,
        input_core_dims=[[sample_dim], [sample_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return result
