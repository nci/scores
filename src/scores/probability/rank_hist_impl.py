"""
This module contains methods related to the rank histogram.
"""
import warnings
from typing import Optional

import numpy as np
import xarray as xr

from scores.processing import aggregate
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions


def _value_at_rank(fcst: XarrayLike, obs: XarrayLike, ens_member_dim: str) -> XarrayLike:
    """
    Calculates the mass that each (fcst, obs) pair contributes to each bar of the rank histogram.
    We call this mass at each rank `value_at_rank`, which is what is returned.

    For an ensemble of size n, observations receive a min_rank and max_rank between 1 and n+1 as follows:
        min_rank = (count of ensemble members where obs > fcst) + 1
        max_rank = (count of ensemble members where obs >= fcst) + 1

    Then the `value_at_rank` is
        1 / (max_rank - min_rank + 1)  if min_rank <= rank <= max_rank
        0 otherwise.

    Returns NaN for forecast cases where an ensemble member is missing.

    Args:
        fcst: xarray object of ensemble forecasts, including dimension `ens_member_dim`
        obs: xarray object of observations
        ens_member_dim: name of the ensemble memeber dimension in `fcst`

    Returns:
        the value of the rank for each (fcst, obs) across all possible ranks, as an xarray
        object which includes the dimension 'rank'
    """
    # ranks only make sense when there are no missing values from the ensemble
    # need to screen out cases where missing members exist.
    ens_member_count = fcst.count(ens_member_dim)
    ens_size = len(fcst[ens_member_dim])
    no_missing_members = ens_member_count == ens_size

    min_rank = (obs > fcst).sum(ens_member_dim) + 1
    max_rank = (obs >= fcst).sum(ens_member_dim) + 1

    rank = np.arange(1, ens_size + 2)
    rank = xr.DataArray(rank, dims=["rank"], coords={"rank": rank})

    value_at_rank = 1 / (max_rank - min_rank + 1)
    value_at_rank = (
        value_at_rank.where((rank >= min_rank) & (rank <= max_rank), 0).where(no_missing_members).where(obs.notnull())
    )

    return value_at_rank


def rank_histogram(
    fcst: XarrayLike,
    obs: XarrayLike,
    ens_member_dim: str,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
):
    """
    Calculates the values of the rank histogram for a set of ensemble forecasts and
    corresponding observations.

    In this case, `rank_histogram` returns the relative frequencies of observations'
    rank against the sorted values of an ensemble forecast. If an ensemble has n members,
    each observation is ranked from 1 to n + 1, assuming that the observation does not
    equal any of the values of the ensemble. For example, if the observation is less than
    all the ensemble members, it is assigned a rank of 1, while if it is greater than
    all the ensemble members, it is assigned a rank of n + 1.

    In the case where the observation shares the same value as one or more of the ensemble members,
    the contribution of this observation to the relative frequencies is shared equally
    among all relevant ranks. For example, if the observation value equals the 2nd, 3rd and 4th
    sorted members of an ensemble, then the observation is assigned a joint rank of
    2, 3, 4 and 5, and a relative weighting of 1/4 is placed on its contribution to each of these
    ranks when computing the final relative frequency. This method of dealing with "ties"
    ensures that the expected rank histogram of a probabilistically calibrated ensemble is flat.

    In the case when there is a NaN in one of the ensemble members, the entire ensemble is
    treates as NaN for this particular forecast case.

    Args:
        fcst: forecast of ensemble values, containing the dimension ``ens_member_dim``
        obs: observations
        ens_member_dim: name of the dimension in ``fcst`` that indexes the ensemble member
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            rank histogram values. All other dimensions will be preserved. As a special case,
            'all' will allow all dimensions to be reduced. Only one of ``reduce_dims`` and
            ``preserve_dims`` can be supplied. The default behaviour if neither are supplied
            is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating the
            rank histogram values. All other dimensions will be reduced. As a special case,
            'all' will allow all dimensions to be preserved, apart from ``severity_dim`` and
            ``prob_threshold_dim``. Only one of ``reduce_dims`` and ``preserve_dims`` can be supplied.
            The default behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighting contributions to the rank histogram
        calculations (e.g. by latitude).

    Returns:
        xarray of rank histogram values, including the dimension 'rank'.

    Raises:
        ValueError if any of the dimensions in the inputs have the name 'rank'.

    Warns:
        UserWarning if there are any NaNs in ``fcst``.

    References:
        - Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble forecasts.
            Monthly Weather Review, 129(3), 550-560.
            https://doi.org/10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2
        - Talagrand, O. (1999). Evaluation of probabilistic prediction systems.
            In Workshop proceedings "Workshop on predictability", 20-22 October 1997, ECMWF, Reading, UK.

    See also:
        - :py:func:`scores.probability.Pit`
        - :py:func:`scores.probability.Pit_fcst_at_obs`

    Examples:
        Calculate and plot the rank histogram for an under-dispersive ensemble forecast:

        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> from scores.probability import rank_histogram
        >>> fcst = xr.DataArray(norm.rvs(size=(500, 10)), dims=['time', 'ensemble'])
        >>> obs = xr.DataArray(norm.rvs(scale=2, size=(500)), dims=['time'])
        >>> rank_relative_frequencies = rank_histogram(fcst, obs, ensemble_member_dim='ensemble')
    """
    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims

    dims_for_mean = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=ens_member_dim,
    )

    any_null = fcst.isnull().any()
    if isinstance(any_null, xr.Dataset):
        any_null = any([bool(any_null[var]) for var in any_null.data_vars])
    if any_null:
        warnings.warn(
            "Encountered a NaN in `fcst`. Any forecast case with NaN for one ensemble member "
            "will be treated as NaN for all ensemble members.",
            UserWarning,
        )

    result = _value_at_rank(fcst, obs, ens_member_dim)
    result = aggregate(result, reduce_dims=dims_for_mean, weights=weights)
    # result = apply_weights(result, weights=weights).mean(dim=dims_for_mean)
    # rescale so that the values across the ranks sum to 1
    # may be needed if supplied weights don't sum to 1
    result = result / result.sum("rank")
    return result
