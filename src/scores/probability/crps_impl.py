"""
This module supports the implementation of the CRPS scoring function, drawing from additional functions.
The two primary methods, `crps_cdf` and `crps_for_ensemble` are imported into 
the probability module to be part of the probability API.
"""

from collections.abc import Iterable
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

import scores.utils
from scores.probability.checks import coords_increasing
from scores.processing.cdf import (
    add_thresholds,
    cdf_envelope,
    decreasing_cdfs,
    integrate_square_piecewise_linear,
    observed_cdf,
    propagate_nan,
)
from scores.typing import XarrayLike


# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
def check_crps_cdf_inputs(
    fcst,
    obs,
    threshold_dim,
    threshold_weight,
    fcst_fill_method,
    threshold_weight_fill_method,
    integration_method,
    dims,
):
    """Checks that `crps_cdf` inputs are valid."""
    if threshold_dim not in fcst.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `fcst`")

    if threshold_weight is not None and threshold_dim not in threshold_weight.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `threshold_weight`")

    if threshold_dim in obs.dims:
        raise ValueError(f"'{threshold_dim}' is a dimension of `obs`")

    if not set(obs.dims).issubset(fcst.dims):
        raise ValueError("Dimensions of `obs` must be a subset of dimensions of `fcst`")

    if threshold_weight is not None and not set(threshold_weight.dims).issubset(fcst.dims):
        raise ValueError("Dimensions of `threshold_weight` must be a subset of dimensions of `fcst`")

    if dims is not None and not set(dims).issubset(fcst.dims):
        raise ValueError("`dims` must be a subset of `fcst` dimensions")  # pragma: no cover

    if fcst_fill_method not in ["linear", "step", "forward", "backward"]:
        raise ValueError("`fcst_fill_method` must be 'linear', 'step', 'forward' or 'backward'")

    if threshold_weight is not None and threshold_weight_fill_method not in [
        "linear",
        "step",
        "forward",
        "backward",
    ]:
        msg = "`threshold_weight_fill_method` must be 'linear', 'step', 'forward' or "
        msg += "'backward' if `threshold_weight` is supplied"
        raise ValueError(msg)

    if integration_method not in ["exact", "trapz"]:
        raise ValueError("`integration_method` must be 'exact' or 'trapz'")

    if len(fcst[threshold_dim]) < 2:
        raise ValueError("`threshold_dim` in `fcst` must have at least 2 values to calculate CRPS")

    if not coords_increasing(fcst, threshold_dim):
        raise ValueError("`threshold_dim` coordinates in `fcst` must be increasing")

    if threshold_weight is not None and not coords_increasing(threshold_weight, threshold_dim):
        raise ValueError("`threshold_dim` coordinates in `threshold_weight` must be increasing")

    if threshold_weight is not None and (threshold_weight < 0).any():
        raise ValueError("`threshold_weight` has negative values")


def crps_cdf_reformat_inputs(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    threshold_weight: Optional[xr.DataArray] = None,
    additional_thresholds: Optional[Iterable[float]] = None,
    fcst_fill_method: Literal["linear", "step", "forward", "backward"] = "linear",
    threshold_weight_fill_method: Literal["linear", "step", "forward", "backward"] = "forward",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Takes `fcst`, `obs` and `threshold_weight` inputs from `crps_cdf` and reformats them
    for use in `crps_cdf`; that is, `cdf_fcst`, `cdf_obs` and `threshold_weight` are
    returned with matching dimensions and coordinates.

    `additional_thresholds` contains additional thresholds to use.
    Useful if wanting to increase density of thresholds for `crps_cdf_trapz`.

    NaNs are filled, unless there are too few non-NaNs for filling.
    """
    # will use all thresholds from fcst, obs and (if applicable) weight

    fcst_thresholds = fcst[threshold_dim].values
    obs_thresholds = pd.unique(obs.values.flatten())

    weight_thresholds = []  # type: ignore
    if threshold_weight is not None:
        weight_thresholds = threshold_weight[threshold_dim].values  # type: ignore

    if additional_thresholds is None:
        additional_thresholds = []

    thresholds = np.concatenate((weight_thresholds, fcst_thresholds, obs_thresholds, additional_thresholds))  # type: ignore
    thresholds = np.sort(pd.unique(thresholds))
    thresholds = thresholds[~np.isnan(thresholds)]

    # get obs in cdf form with correct thresholds
    obs_cdf = observed_cdf(
        obs,
        threshold_dim,
        threshold_values=thresholds,
        include_obs_in_thresholds=False,  # thresholds already has rounded obs values
        precision=0,
    )

    # get fcst with correct thresholds
    fcst = add_thresholds(fcst, threshold_dim, thresholds, fcst_fill_method)

    # get weight with correct thresholds
    if threshold_weight is None:
        # the weight is 1
        weight_cdf = xr.full_like(fcst, 1.0)
    else:
        weight_cdf = add_thresholds(threshold_weight, threshold_dim, thresholds, threshold_weight_fill_method)

    fcst_cdf, obs_cdf, weight_cdf = xr.broadcast(fcst, obs_cdf, weight_cdf)

    return fcst_cdf, obs_cdf, weight_cdf


# pylint: disable=too-many-locals
def crps_cdf(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    threshold_dim: str = "threshold",
    threshold_weight: Optional[xr.DataArray] = None,
    additional_thresholds: Optional[Iterable[float]] = None,
    propagate_nans: bool = True,
    fcst_fill_method: Literal["linear", "step", "forward", "backward"] = "linear",
    threshold_weight_fill_method: Literal["linear", "step", "forward", "backward"] = "forward",
    integration_method: Literal["exact", "trapz"] = "exact",
    reduce_dims: Optional[Iterable[str]] = None,
    preserve_dims: Optional[Iterable[str]] = None,
    weights=None,
    include_components=False,
):
    """Calculates the CRPS probabilistic metric given CDF input.

    Calculates the continuous ranked probability score (CRPS), or the mean CRPS over
    specified dimensions, given forecasts in the form of predictive cumulative
    distribution functions (CDFs). Can also calculate threshold-weighted versions of the
    CRPS by supplying a `threshold_weight`.

    Predictive CDFs here are described by an indexed set of values rather than by
    closed forumulae. As a result, the resolution or number of points at which the CDF
    is realised has an impact on the calculation of areas under and over the curve to
    obtain the CRPS result. The term 'threshold' is used to describe the dimension which
    is used as an index for predictive CDF values. Various techniques are used to
    interpolate CDF values between indexed thresholds.

    Given:
        - a predictive CDF `fcst` indexed at thresholds by variable x
        - an observation in CDF form `obs_cdf` (i.e., :math:`\\text{obs_cdf}(x) = 0` if \
         :math:`x < \\text{obs}` and 1 if :math:`x >= \\text{obs}`)
        - a `threshold_weight` array indexed by variable x

    The threshold-weighted CRPS is given by:
        - :math:`twCRPS = \\int_{-\\infty}^{\\infty}{[\\text{threshold_weight}(x) \\times \
        (\\text{fcst}(x) - \\text{obs_cdf}(x))^2]\\text{d}x}`, over all thresholds x.
        - The usual CRPS is the threshold-weighted CRPS with :math:`\\text{threshold_weight}(x) = 1` for all x.

    This can be decomposed into an over-forecast penalty:
        :math:`\\int_{-\\infty}^{\\infty}{[\\text{threshold_weight}(x) \\times \\text{fcst}(x) - 
        \\text{obs_cdf}(x))^2]\\text{d}x}`, over all thresholds x where x >= obs
    
    and an under-forecast penalty:
        :math:`\\int_{-\\infty}^{\\infty}{[\\text{threshold_weight}(x) \\times \\text{(fcst}(x) - 
        \\text{obs_cdf}(x)^2]\\text{d}x}`, over all thresholds x where x <= obs.

    To obtain the components of the CRPS score, set ``include_components`` to ``True``.  

    Note that there are several ways to decompose the CRPS and this decomposition differs from
    the one used in the :py:func:`scores.probability.crps_for_ensemble` function. 

    Note that the function `crps_cdf` is designed so that the `obs` argument contains
    actual observed values. `crps_cdf` will convert `obs` into CDF form in order to
    calculate the CRPS.

    To calculate CRPS, integration is applied over the set of thresholds x taken from:
        - `fcst[threshold_dim].values`,
        - `obs.values`.
        - `threshold_weight[threshold_dim].values` if applicable.
        - `additional_thresholds` if applicable.
        - (with NaN values excluded)

    There are two methods of integration:
        - "exact" gives the exact integral under that assumption that that `fcst` is 
          continuous and piecewise linear between its specified values, and that 
          `threshold_weight` (if supplied) is piecewise constant and right-continuous 
          between its specified values.
        - "trapz" simply uses a trapezoidal rule using the specified values, and so is 
          an approximation of the CRPS. To get an accurate approximation, the density 
          of threshold values can be increased by supplying `additional_thresholds`.

    Both methods of calculating CRPS may require adding additional values to the
    `threshold_dim` dimension in `fcst` and (if supplied) `threshold_weight`.
    `fcst_fill_method` and `weight_fill_method` specify how `fcst` and `weight` are to
    be filled at these additional points.

    The function `crps_cdf` calculates the CRPS using forecast CDF values 'as is'.
    No checks are performed to ensure that CDF values in `fcst` are nondecreasing along
    `threshold_dim`. Checks are conducted on `fcst` and `threshold_weight` (if applicable)
    to ensure that coordinates are increasing along `threshold_dim`.

    Args:
        fcst: array of forecast CDFs, with the threshold dimension given by `threshold_dim`.
        obs: array of observations, not in CDF form.
        threshold_dim: name of the dimension in `fcst` that indexes the thresholds.
        threshold_weight: weight to be applied along `threshold_dim` to calculate 
            threshold-weighted CRPS. Must contain `threshold_dim` as a dimension, and may 
            also include other dimensions from `fcst`. If `weight=None`, a weight of 1 
            is applied everywhere, which gives the usual CRPS.
        additional_thresholds: additional thresholds values to add to `fcst` and (if 
            applicable) `threshold_weight` prior to calculating CRPS.
        propagate_nans: If `True`, propagate NaN values along `threshold_dim` in `fcst` 
            and `threshold_weight` prior to calculating CRPS. This will result in CRPS 
            being NaN for these cases. If `False`, NaN values in `fcst` and `weight` will 
            be replaced, wherever possible, with non-NaN values using the fill method 
            specified by `fcst_fill_method` and `threshold_weight_fill_method`.
        fcst_fill_method: how to fill values in `fcst` when NaNs have been introduced 
            (by including additional thresholds) or are specified to be removed (by 
            setting `propagate_nans=False`). Select one of: 

            - "linear": use linear interpolation, then replace any leading or \
            trailing NaNs using linear extrapolation. Afterwards, all values are \
            clipped to the closed interval [0, 1].
            - "step": apply forward filling, then replace any leading NaNs with 0. 
            - "forward": first apply forward filling, then remove any leading NaNs by \
            back filling.
            - "backward": first apply back filling, then remove any trailing NaNs by \
            forward filling.
            - (In most cases, "linear" is likely the appropriate choice.)

        threshold_weight_fill_method: how to fill values in `threshold_weight` when NaNs
            have been introduced (by including additional thresholds) or are specified
            to be removed (by setting `propagate_nans=False`). Select one of "linear", 
            "step", "forward" or "backward". If the weight function is continuous, 
            "linear" is probably the best choice. If it is an increasing step function, 
            "forward" may be best.
                  
        integration_method (str): one of "exact" or "trapz".
        preserve_dims (Tuple[str]): dimensions to preserve in the output. All other dimensions are collapsed
            by taking the mean.
        reduce_dims (Tuple[str]): dimensions to reduce in the output by taking the mean. All other dimensions are
            preserved.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        include_components (bool): if True, include the under and over forecast components of
            the score in the returned dataset.

    Returns:
        xr.Dataset: The following are the produced Dataset variables:
            - "total" the total CRPS.
            - "underforecast_penalty": the under-forecast penalty contribution of the CRPS.
            - "overforecast_penalty": the over-forecast penalty contribution of the CRPS.

    Raises:
        ValueError: if `threshold_dim` is not a dimension of `fcst`.
        ValueError: if `threshold_dim` is not a dimension of `threshold_weight` when
            `threshold_weight` is not `None`.
        ValueError: if `threshold_dim` is a dimension of `obs`.
        ValueError: if dimensions of `obs` are not also dimensions of `fcst`.
        ValueError: if dimensions of `threshold_weight` are not also dimensions of `fcst`
            when `threshold_weight` is not `None`.
        ValueError: if `dims` is not a subset of dimensions of `fcst`.
        ValueError: if `fcst_fill_method` is not one of 'linear', 'step', 'forward' or
            'backward'.
        ValueError: if `weight_fill_method` is not one of 'linear', 'step', 'forward' or
            'backward'.
        ValueError: if `fcst[threshold_dim]` has less than 2 values.
        ValueError: if coordinates in `fcst[threshold_dim]` are not increasing.
        ValueError: if `threshold_weight` is not `None` and coordinates in
            `threshold_weight[threshold_dim]` are not increasing.
        ValueError: if `threshold_weight` has negative values.

    See also:
        - :py:func:`scores.probability.crps_cdf_brier_decomposition`
        - :py:func:`scores.probability.crps_for_ensemble`

    References:
        - Matheson, J. E., and R. L. Winkler, 1976: Scoring rules for continuous probability distributions. \
            Management Science, 22(10), 1087–1095. https://doi.org/10.1287/mnsc.22.10.1087
        - Gneiting, T., & Ranjan, R. (2011). Comparing Density Forecasts Using Threshold- and \
            Quantile-Weighted Scoring Rules. \
            Journal of Business & Economic Statistics, 29(3), 411–422. https://doi.org/10.1198/jbes.2010.08110
    """

    dims = scores.utils.gather_dimensions(
        fcst.dims,
        obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    check_crps_cdf_inputs(
        fcst,
        obs,
        threshold_dim,
        threshold_weight,
        fcst_fill_method,
        threshold_weight_fill_method,
        integration_method,
        dims,
    )

    if propagate_nans:
        fcst = propagate_nan(fcst, threshold_dim)  # type: ignore
        if threshold_weight is not None:
            threshold_weight = propagate_nan(threshold_weight, threshold_dim)  # type: ignore

    fcst, obs, threshold_weight = crps_cdf_reformat_inputs(
        fcst,
        obs,
        threshold_dim,
        threshold_weight=threshold_weight,
        additional_thresholds=additional_thresholds,
        fcst_fill_method=fcst_fill_method,
        threshold_weight_fill_method=threshold_weight_fill_method,
    )

    if integration_method == "exact":
        result = crps_cdf_exact(
            fcst,
            obs,
            threshold_weight,
            threshold_dim,
            include_components=include_components,
        )

    if integration_method == "trapz":
        result = crps_cdf_trapz(
            fcst,
            obs,
            threshold_weight,
            threshold_dim,
            include_components=include_components,
        )

    weighted = scores.functions.apply_weights(result, weights=weights)

    dims.remove(threshold_dim)  # type: ignore

    result = weighted.mean(dim=dims)  # type: ignore

    return result


def crps_cdf_exact(
    cdf_fcst: xr.DataArray,
    cdf_obs: xr.DataArray,
    threshold_weight: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    include_components=False,
) -> xr.Dataset:
    """
    Calculates exact value of CRPS assuming that:
        - the forecast CDF is continuous piecewise linear, with join points given by
          values in `cdf_fcst`,
        - the observation CDF is right continuous with values in {0,1} given by `cdf_obs`,
        - the threshold weight function is right continuous with values in {0,1} given
          by `threshold_weight`.

    If these assumptions do not hold, it might be best to use `crps_approximate`, with a
    sufficiently high resolution along `threshold_dim`.

    This function assumes that `cdf_fcst`, `cdf_obs`, `threshold_weight` have same shape.
    Also assumes that values along the `threshold_dim` dimension are increasing.

    Returns:
        (xr.Dataset): Dataset with `threshold_dim` collapsed containing DataArrays with
        CRPS and its decomposition, labelled "total", "underforecast_penalty" and
        "overforecast_penalty". NaN is returned if there is a NaN in the corresponding
        `cdf_fcst`, `cdf_obs` or `threshold_weight`.
    """

    # identify where input arrays have no NaN, collapsing `threshold_dim`
    # Mypy doesn't realise the isnan and any come from xarray not numpy
    inputs_without_nan = (
        ~np.isnan(cdf_fcst).any(threshold_dim)  # type: ignore
        & ~np.isnan(cdf_obs).any(threshold_dim)  # type: ignore
        & ~np.isnan(threshold_weight).any(threshold_dim)  # type: ignore
    )

    # thresholds in the closure of the interval (i.e. including endpoints) where
    # weight is 1
    interval_where_weight_one = (threshold_weight == 1) | ((threshold_weight.shift(**{threshold_dim: 1})) == 1)  # type: ignore

    # thresholds in the closure of the interval where cdf_obs is 1
    interval_where_obs_one = cdf_obs == 1

    # thresholds in the closure of the interval where obs cdf is 0
    interval_where_obs_zero = (cdf_obs == 0) | ((cdf_obs.shift(**{threshold_dim: 1})) == 0)  # type: ignore

    # over-forecast penalty contribution to CRPS: integral(w(x) * (F(x) - 1)^2) where x >= obs
    over = (cdf_fcst - 1).where(interval_where_obs_one).where(interval_where_weight_one)
    over = integrate_square_piecewise_linear(over, threshold_dim)
    # If zero penalty, could be NaN. Replace with 0, then NaN using inputs_without_nan
    over = over.where(~np.isnan(over), 0).where(inputs_without_nan)

    # under-forecast penalty contribution to CRPS: integral(w(x) * F(x)^2) where x < obs
    under = (cdf_fcst).where(interval_where_obs_zero).where(interval_where_weight_one)
    under = integrate_square_piecewise_linear(under, threshold_dim)
    # If zero penalty, could be NaN. Replace with 0, then nan using inputs_without_nan
    under = under.where(~np.isnan(under), 0).where(inputs_without_nan)

    total = over + under
    result = total.to_dataset(name="total")

    if include_components:
        result = xr.merge(
            [
                total.rename("total"),
                under.rename("underforecast_penalty"),
                over.rename("overforecast_penalty"),
            ]
        )

    return result


def crps_cdf_brier_decomposition(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    threshold_dim: str = "threshold",
    additional_thresholds: Optional[Iterable[float]] = None,
    fcst_fill_method: Literal["linear", "step", "forward", "backward"] = "linear",
    reduce_dims: Optional[Iterable[str]] = None,
    preserve_dims: Optional[Iterable[str]] = None,
) -> xr.Dataset:
    """
    Given an array `fcst` of predictive CDF values indexed along `threshold_dim`, and
    an array `obs` of  observations, calculates the mean Brier score for each index along
    `threshold_dim`. Since the mean CRPS is the integral of the mean Brier score over
    all thresholds, this gives a threshold decomposition of the mean CRPS.

    If any there are any NaNs along the threshold dimension of `fcst`, then NaNs are
    propagated along this dimension prior to calculating the decomposition. If
    propagating NaNs is not desired, the user may first fill NaNs in `fcst` using
    `scores.probability.functions.fill_cdf`.

    Args:
        fcst (xr.DataArray): DataArray of CDF values with threshold dimension `threshold_dim`.
        obs (xr.DataArray): DataArray of observations, not in CDF form.
        threshold_dim (str): name of the threshold dimension in `fcst`.
        additional_thresholds (Optional[Iterable[float]]): additional thresholds \
            at which to calculate the mean Brier score.
        fcst_fill_method (Literal["linear", "step", "forward", "backward"]): How to fill NaN
            values in `fcst` that arise from new user-supplied thresholds or thresholds derived
            from observations.

            - "linear": use linear interpolation, and if needed also extrapolate linearly. \
              Clip to 0 and 1. Needs at least two non-NaN values for interpolation, \
              so returns NaNs where this condition fails.
            - "step": use forward filling then set remaining leading NaNs to 0. \
              Produces a step function CDF (i.e. piecewise constant).
            - "forward": use forward filling then fill any remaining leading NaNs with \
              backward filling.
            - "backward": use backward filling then fill any remaining trailing NaNs with \
              forward filling.
        dims: dimensions to preserve in the output. The dimension `threshold_dim` is always \
            preserved, even if not specified here.

    Returns:
        An xarray Dataset with data_vars:
            - "total_penalty": the mean Brier score for each threshold.
            - "underforecast_penalty": the mean of the underforecast penalties for the Brier score. For a particular
                forecast case, this component equals 0 if the event didn't occur and the Brier score if it did.
            - "overforecast_penalty": the mean of the overforecast penalties for the Brier score. For a particular
                forecast case, this component equals 0 if the event did occur and the Brier score if it did not.

    Raises:
        ValueError: if `threshold_dim` is not a dimension of `fcst`.
        ValueError: if `threshold_dim` is a dimension of `obs`.
        ValueError: if dimensions of `obs` are not also among the dimensions of `fcst`.
        ValueError: if dimensions in `dims` is not among the dimensions of `fcst`.
        ValueError: if `fcst_fill_method` is not one of 'linear', 'step', 'forward' or 'backward'.
        ValueError: if coordinates in `fcst[threshold_dim]` are not increasing.
    """
    dims = scores.utils.gather_dimensions(
        fcst.dims,
        obs.dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    check_crps_cdf_brier_inputs(fcst, obs, threshold_dim, fcst_fill_method, dims)

    fcst = propagate_nan(fcst, threshold_dim)  # type: ignore

    fcst, obs, _ = crps_cdf_reformat_inputs(
        fcst,
        obs,
        threshold_dim,
        threshold_weight=None,
        additional_thresholds=additional_thresholds,
        fcst_fill_method=fcst_fill_method,
        threshold_weight_fill_method="forward",
    )

    dims.remove(threshold_dim)  # type: ignore

    # brier score for each forecast case
    bscore = (fcst - obs) ** 2
    not_nan = ~np.isnan(bscore)

    # `obs` here is the empirical CDF of the observation
    # when `obs == 1` the observation was lower than the threshold considered
    # and hence the Brier score at this threshold penalises an over-forecast
    over = bscore.where(np.isclose(obs, 1), 0).where(not_nan).mean(dims)
    under = bscore.where(np.isclose(obs, 0), 0).where(not_nan).mean(dims)
    total = over + under

    result = xr.merge(
        [
            total.rename("total_penalty"),
            under.rename("underforecast_penalty"),
            over.rename("overforecast_penalty"),
        ]
    )

    return result


def check_crps_cdf_brier_inputs(fcst, obs, threshold_dim, fcst_fill_method, dims):
    """Checks inputs to 'crps_cdf_brier_decomposition'."""

    if threshold_dim not in fcst.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `fcst`")

    if threshold_dim in obs.dims:
        raise ValueError(f"'{threshold_dim}' is a dimension of `obs`")

    if not set(obs.dims).issubset(fcst.dims):
        raise ValueError("Dimensions of `obs` must be a subset of dimensions of `fcst`")

    if dims is not None and not set(dims).issubset(fcst.dims):
        raise ValueError("`dims` must be a subset of `fcst` dimensions")  # pragma: no cover

    if fcst_fill_method not in ["linear", "step", "forward", "backward"]:
        raise ValueError("`fcst_fill_method` must be 'linear', 'step', 'forward' or 'backward'")

    if not coords_increasing(fcst, threshold_dim):
        raise ValueError("`threshold_dim` coordinates in `fcst` must be increasing")


def adjust_fcst_for_crps(
    fcst: xr.DataArray,
    threshold_dim: str,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    decreasing_tolerance: float = 0,
    additional_thresholds: Optional[Iterable[float]] = None,
    fcst_fill_method: Literal["linear", "step", "forward", "backward"] = "linear",
    integration_method: Literal["exact", "trapz"] = "exact",
) -> xr.DataArray:
    """
    This function takes a forecast cumulative distribution functions (CDF) `fcst`.

    If `fcst` is not decreasing outside of specified tolerance, it returns `fcst`.

    Otherwise, the CDF envelope for `fcst` is computed, and the CDF from among:
        - `fcst`,
        - the upper envelope, and
        - the lower envelope

    that has the higher (i.e. worse) CRPS is returned. In the event of a tie,
    preference is given in the order `fcst` then upper.

    See `scores.probability.functions.cdf_envelope` for details about the CDF envelope.

    The use case for this is when, either due to rounding or poor forecast process, the
    forecast CDF `fcst` fails to be nondecreasing. Rather than simply replacing `fcst`
    with NaN, `adjust_fcst_for_crps` returns a CDF for which CRPS can be calculated, but
    possibly with a predictive performance cost as measured by CRPS.

    Whether a CDF is decreasing outside specified tolerance is determined as follows.
    For each CDF in `fcst`, the sum of incremental decreases along the threshold dimension
    is calculated. For example, if the CDF values are 

    `[0, 0.4, 0.3, 0.9, 0.88, 1]`

    then the sum of incremental decreases is -0.12. This CDF decreases outside specified
    tolerance if 0.12 > `decreasing_tolerance`.

    The adjusted array of forecast CDFs is determined as follows:
        - any NaN values in `fcst` are propagated along `threshold_dim` so that in each case \
            the entire CDF is NaN;
        - any CDFs in `fcst` that are decreasing within specified tolerance are unchanged;
        - any CDFs in `fcst` that are decreasing outside specified tolerance are replaced with \
            whichever of the upper or lower CDF envelope gives the highest CRPS, unless the original \
            values give a higher CRPS in which case original values are kept.

    See `scores.probability.functions.cdf_envelope` for a description of the 'CDF envelope'.
 
    If propagating NaNs is not desired, the user may first fill NaNs in `fcst` using
    `scores.probability.functions.fill_cdf`.

    The CRPS for each forecast case is calculated using `crps`, with a weight of 1.

    Args:
        fcst: DataArray of CDF values with threshold dimension `threshold_dim`.
        threshold_dim: name of the threshold dimension in `fcst`.
        obs: DataArray of observations.
        decreasing_tolerance: nonnegative tolerance value.
        additional_thresholds: optional additional thresholds passed on to `crps` when calculating CRPS.
        fcst_fill_method: `fcst` fill method passed on to `crps` when calculating CRPS.
        integration_method: integration method passed on to `crps` when calculating CRPS.

    Returns:
        An xarray DataArray of possibly adjusted forecast CDFs, where adjustments are made
        to penalise CDFs that decrease outside tolerance.

    Raises:
        ValueError: If `threshold_dim` is not a dimension of `fcst`.
        ValueError: If `decreasing_tolerance` is negative.
    """
    if threshold_dim not in fcst.dims:
        raise ValueError(f"'{threshold_dim}' is not a dimension of `fcst`")

    if decreasing_tolerance < 0:
        raise ValueError("`decreasing_tolerance` must be nonnegative")

    fcst = propagate_nan(fcst, threshold_dim)  # type: ignore

    is_decreasing = decreasing_cdfs(fcst, threshold_dim, decreasing_tolerance)

    if not is_decreasing.any():
        return fcst

    fcst_env = cdf_envelope(fcst, threshold_dim)

    # dimensions to preserve when calculating CRPS
    crps_dims = [x for x in fcst_env.dims if x != threshold_dim]

    crps_fcst_env = crps_cdf(
        fcst_env,
        obs,
        threshold_dim=threshold_dim,
        additional_thresholds=additional_thresholds,
        fcst_fill_method=fcst_fill_method,
        integration_method=integration_method,
        preserve_dims=crps_dims,  # type: ignore
    )

    fcst_type_to_use = crps_fcst_env["total"].idxmax("cdf_type")

    fcst_to_return = fcst_env.where(fcst_env["cdf_type"] == fcst_type_to_use).max("cdf_type")

    fcst_to_return = fcst_to_return.where(is_decreasing).combine_first(fcst)

    return fcst_to_return


def crps_cdf_trapz(
    cdf_fcst: xr.DataArray,
    cdf_obs: xr.DataArray,
    threshold_weight: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    include_components=False,
) -> xr.Dataset:
    """
    Returns dataset with CRPS estimate and decomposition using a trapezoidal rule on
    points from the integrand
        threshold_weight(x) * (cdf_fcst(x) - cdf_obs(x)) ** 2

    The error in the estimate improves as the distance between adjacent thresholds in
    `threshold_dim` approaches zero.

    NaN is returned if there is a NaN in the corresponding `cdf_fcst`, `cdf_obs` or
    `threshold_weight`.
    """

    # identify where input arrays have no NaN, collapsing `threshold_dim`
    # mypy doesn't realise the isnan and any come from xarray not numpy
    inputs_without_nan = (
        ~np.isnan(cdf_fcst).any(dim=threshold_dim)  # type: ignore
        & ~np.isnan(cdf_obs).any(dim=threshold_dim)  # type: ignore
        & ~np.isnan(threshold_weight).any(dim=threshold_dim)  # type: ignore
    )

    # total error measured by CRPS
    total = (threshold_weight * (cdf_fcst - cdf_obs) ** 2).integrate(threshold_dim).where(inputs_without_nan)

    over = (cdf_obs * threshold_weight * (cdf_fcst - cdf_obs) ** 2).integrate(threshold_dim).where(inputs_without_nan)

    under = total - over

    result = total.to_dataset(name="total")

    if include_components:
        result = xr.merge(
            [
                total.rename("total"),
                under.rename("underforecast_penalty"),
                over.rename("overforecast_penalty"),
            ]
        )

    return result


def crps_step_threshold_weight(
    step_points: xr.DataArray,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    threshold_values: Optional[Iterable[float]] = None,
    steppoints_in_thresholds: bool = True,
    steppoint_precision: float = 0,
    weight_upper: bool = True,
) -> xr.DataArray:
    """Generates an array of weights based on DataArray step points.

    Creates an array of threshold weights, which can be used to calculate
    threshold-weighted CRPS, based on a step function. Applies a weight of 1 when
    step_point >= threshold, and a weight of 0 otherwise. Zeros and ones in the output
    weight function can be reversed by setting `weight_upper=False`.

    Args:
        step_points (xr.DataArray): points at which the weight function changes value from 0 to 1.
        threshold_dim (str): name of the threshold dimension in the returned array weight function.
        threshold_values (str): thresholds at which to calculate weights.
        steppoints_in_thresholds (bool): include `step_points` among the `threshold_dim` values.
        steppoint_precision (float): precision at which to round step_points prior to calculating the
            weight function. Select 0 for no rounding.
        weight_upper (bool): If true, returns a weight of 1 if step_point >= threshold, and a
            weight of 0 otherwise. If false, returns a weight of 0 if step_point >= threshold,
            and a weight of 1 otherwise.

    Returns:
        (xr.DataArray): Zeros and ones with the dimensions in `step_points`
        and an additional `threshold_dim` dimension.
    """

    weight = observed_cdf(
        step_points,
        threshold_dim,
        threshold_values=threshold_values,
        include_obs_in_thresholds=steppoints_in_thresholds,
        precision=steppoint_precision,
    )

    if not weight_upper:
        weight = 1 - weight

    return weight


def crps_for_ensemble(
    fcst: XarrayLike,
    obs: XarrayLike,
    ensemble_member_dim: str,
    *,  # Force keywords arguments to be keyword-only
    method: Literal["ecdf", "fair"] = "ecdf",
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[XarrayLike] = None,
    include_components: Optional[bool] = False,
) -> XarrayLike:
    """
    Calculates the continuous ranked probability score (CRPS) given an ensemble of forecasts.
    An ensemble of forecasts can also be thought of as a random sample from the predictive
    distribution.

    Given an observation y, and ensemble member values :math:`x_i` (for :math:`1 \\leq i \\leq M`), the CRPS is
    calculated by the formula


    .. math::
        CRPS(x_i, x_j, y) = \\frac{\\sum_{i=1}^{M}(|x_i - y|)}{M} - \\frac{\\sum_{i=1}^{M}\\sum_{j=1}^{M}(|x_i - x_j|)}{2K}

    where the first sum is iterated over :math:`1 \\leq i \\leq M` and the second sum is iterated over
    :math:`1 \\leq i \\leq M` and :math:`1 \\leq j \\leq M`.

    The value of the constant K in this formula depends on the method:
        - If `method="ecdf"` then :math:`K = M ^ 2`. In this case the CRPS value returned is \
            the exact CRPS value for the empirical cumulative distribution function \
            constructed using the ensemble values.
        - If `method="fair"` then :math:`K = M(M - 1)`. In this case the CRPS value returned \
            is the approximated CRPS where the ensemble values can be interpreted as a \
            random sample from the underlying predictive distribution. This interpretation \
            stems from the formula :math:`\\text{CRPS}(F, y) = \\mathbb{E}(|X - y|) - \\frac{1}{2}\\mathbb{E}(|X - X'|)`, where X and X' \
            are independent samples of the predictive distribution F, y is the observation \
            (possibly unknown) and E denotes the expectation. This choice of K gives an \
            unbiased estimate for the second expectation.

    When the `include_components` flag is set to `True`, the CRPS components are calculated as
    

    .. math::
        CRPS(x_i, x_j, y) = O(x_i, y) + U(x_i, y) - S(x_i, x_j)

    where
        - :math:`O(x_i, y) = \\frac{\\sum_{i=1}^{M} ((x_i - y) \\mathbb{1}{\\{x_i > y\\}})}{M}` which is the \
            overforecast penalty.
        - :math:`U(x_i, y) = \\frac{\\sum_{i=1}^{M} ((y - x_i) \\mathbb{1}{\\{x_i < y\\}})}{M}` which is the \
            underforecast penalty.
        - :math:`S(x_i, x_j) = \\frac{\\sum_{i=1}^{M}\\sum_{j=1}^{M}(|x_i - x_j|)}{2K}` which is the forecast spread term.

    Note that there are several ways to decompose the CRPS and this decomposition differs from the
    one used in the :py:func:`scores_probability.crps_cdf` function.

    Args:
        fcst: Forecast data. Must have a dimension ``ensemble_member_dim``.
        obs: Observation data.
        ensemble_member_dim: the dimension that specifies the ensemble member or the sample
            from the predictive distribution.
        method: Either "ecdf" or "fair".
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores.
        include_components: If True, returns the CRPS with underforecast and overforecast
            penalties, as well as the forecast spread term (see description above).

    Returns:
        xarray object of (weighted mean) CRPS values.

    Raises:
        ValueError: when method is not one of "ecdf" or "fair".

    See also:
        :py:func:`scores.probability.crps_cdf`
        :py:func:`scores.probability.tw_crps_for_ensemble`
        :py:func:`scores.probability.tail_tw_crps_for_ensemble`

    References:
        - C. Ferro (2014), "Fair scores for ensemble forecasts", Quarterly Journal of the \
            Royal Meteorol Society, 140(683):1917-1923. https://doi.org/10.1002/qj.2270
        - T. Gneiting T and A. Raftery (2007), "Strictly proper scoring rules, prediction, \
            and estimation", Journal of the American Statistical Association, 102(477):359-378. \
            https://doi.org/10.1198/016214506000001437
        - M. Zamo and P. Naveau (2018), "Estimation of the Continuous Ranked Probability \
            Score with Limited Information and Applications to Ensemble Weather Forecasts", \
            Mathematical Geosciences 50:209-234, https://doi.org/10.1007/s11004-017-9709-7
    """
    if method not in ["ecdf", "fair"]:
        raise ValueError("`method` must be one of 'ecdf' or 'fair'")

    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims

    dims_for_mean = scores.utils.gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=ensemble_member_dim,
    )
    # Calculate forecast spread term
    fcst_spread_term = 0
    for i in range(fcst.sizes[ensemble_member_dim]):
        fcst_spread_term += abs(fcst - fcst.isel({ensemble_member_dim: i})).sum(dim=ensemble_member_dim)

    ens_count = fcst.count(ensemble_member_dim)
    if method == "ecdf":
        fcst_spread_term = fcst_spread_term / (2 * ens_count**2)
    if method == "fair":
        fcst_spread_term = fcst_spread_term / (2 * ens_count * (ens_count - 1))

    # calculate final CRPS for each forecast case
    fcst_obs_term = abs(fcst - obs).mean(dim=ensemble_member_dim)
    result = fcst_obs_term - fcst_spread_term

    if include_components:
        mask = np.logical_and(~np.isnan(fcst), ~np.isnan(obs))  # create mask so that we can preserve NaNs
        under_penalty = (obs - fcst).where(fcst < obs, 0).where(mask).mean(dim=ensemble_member_dim)
        over_penalty = (fcst - obs).where(fcst > obs, 0).where(mask).mean(dim=ensemble_member_dim)
        # Match NaNs between spread terms and other terms
        fcst_spread_term = fcst_spread_term.where(~np.isnan(fcst_obs_term))  # type: ignore  # mypy is wrong, I think
        result = xr.concat([result, under_penalty, over_penalty, fcst_spread_term], dim="component")
        result = result.assign_coords(component=["total", "underforecast_penalty", "overforecast_penalty", "spread"])

    # apply weights and take means across specified dims
    result = scores.functions.apply_weights(result, weights=weights).mean(dim=dims_for_mean)  # type: ignore

    return result  # type: ignore


def tw_crps_for_ensemble(
    fcst: XarrayLike,
    obs: XarrayLike,
    ensemble_member_dim: str,
    chaining_func: Callable[[XarrayLike], XarrayLike],
    *,  # Force keywords arguments to be keyword-only
    chainging_func_kwargs: Optional[dict[str, Any]] = {},
    method: Literal["ecdf", "fair"] = "ecdf",
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[XarrayLike] = None,
    include_components: Optional[bool] = False,
) -> xr.DataArray:
    """
    Calculates the threshold weighted continuous ranked probability score (twCRPS) given
    ensemble input using a chaining function ``chaining_func`` (see below). An ensemble of 
    forecasts can also be thought of as a random sample from the predictive distribution.

    The twCRPS is calculated by the formula


    .. math::
        \\text{twCRPS}(F, y) = \\mathbb{E}_F \\left| v(X) - v(y) \\right| - \\frac{1}{2} \\mathbb{E}_F \\left| v(X) - v(X') \\right|,

    where :math:`X` and :math:`X'` are independent samples of the predictive distribution :math:`F`,
    :math:`y` is the observation, and :math:`v` is a 'chaining function'.

    The chaining function :math:`v` is the antiderivative of the threshold weight function :math:`w`,
    which is a non-negative function that assigns a weight to each threshold value. For example, if we
    wanted to assign a threshold weight of 1 to thresholds above threshold :math:`t` and a threshold
    weight of 0 to thresholds below :math:`t`, our threshold weight function would be :math:`w(x) = \\mathbb{1}{(x > t)}`,
    where :math:`\\mathbb{1}` is the indicator function which returns a value of 1 if the condition
    is true and 0 otherwise. A chaining function would then be :math:`v(x) = \\text{max}(x, t)`.

    There are currently two methods available for calculating the twCRPS: "ecdf" and "fair". 
        - If `method="ecdf"` then the twCRPS value returned is \
            the exact twCRPS value for the empirical cumulative distribution function \
            constructed using the ensemble values.
        - If `method="fair"` then the twCRPS value returned \
            is the approximated twCRPS where the ensemble values can be interpreted as a \
            random sample from the underlying predictive distribution. See  https://doi.org/10.1002/qj.2270 \
            for more details on the fair CRPS which are relevant for the fair twCRPS.

    The ensemble representation of the empirical twCRPS is


    .. math::
        \\text{twCRPS}(F_{\\text{ens}}, y; v) = \\frac{1}{M} \\sum_{m=1}^{M} \\left| v(x_m) - v(y) \\right| -
        \\frac{1}{2M^2} \\sum_{m=1}^{M} \\sum_{j=1}^{M} \\left| v(x_m) - v(x_j) \\right|,

    where :math:`M` is the number of ensemble members.

    While the ensemble representation of the fair twCRPS is


    .. math::
        \\text{twCRPS}(F_{\\text{ens}}, y; v) = \\frac{1}{M} \\sum_{m=1}^{M} \\left| v(x_m) - v(y) \\right| -
        \\frac{1}{2M(M - 1)} \\sum_{m=1}^{M} \\sum_{j=1}^{M} \\left| v(x_m) - v(x_j) \\right|.


    Args:
        fcst: Forecast data. Must have a dimension ``ensemble_member_dim``.
        obs: Observation data.
        ensemble_member_dim: the dimension that specifies the ensemble member or the sample
            from the predictive distribution.
        chaining_func: the chaining function.
        chainging_func_kwargs: keyword arguments for the chaining function.
        method: Either "ecdf" for the empirical twCRPS or "fair" for the Fair twCRPS.
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores. Note that
            these weights are different to threshold weighting which is done by decision
            threshold.
        include_components: If True, returns the twCRPS with underforecast and overforecast
            penalties, as well as the forecast spread term. See :py:func:`scores.probability.crps_for_ensemble`
            for more details on the decomposition.

    Returns:
        xarray object of twCRPS values.

    Raises:
        ValueError: when ``method`` is not one of "ecdf" or "fair".

    Notes:
        Chaining functions can be created to vary the weights across given dimensions
        such as varying the weights by climatological values.

    References:
        - Allen, S., Ginsbourger, D., & Ziegel, J. (2023). Evaluating forecasts for high-impact \
            events using transformed kernel scores. SIAM/ASA Journal on Uncertainty \
            Quantification, 11(3), 906-940. https://doi.org/10.1137/22M1532184. 
        - Allen, S. (2024). Weighted scoringRules: Emphasizing Particular Outcomes \
            When Evaluating Probabilistic Forecasts. Journal of Statistical Software, \
            110(8), 1-26. https://doi.org/10.18637/jss.v110.i08


    See also:
        :py:func:`scores.probability.crps_for_ensemble`
        :py:func:`scores.probability.tail_tw_crps_for_ensemble`
        :py:func:`scores.probability.interval_tw_crps_for_ensemble`
        :py:func:`scores.probability.crps_cdf`


    Examples:
        Calculate the twCRPS for an ensemble of forecasts where the chaining function is 
        derived from a weight function that assigns a weight of 1 to thresholds above
        0.5 and a weight of 0 to thresholds below 0.5.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import tw_crps_for_ensemble
        >>> fcst = xr.DataArray(np.random.rand(10, 10), dims=['time', 'ensemble'])
        >>> obs = xr.DataArray(np.random.rand(10), dims=['time'])
        >>> tw_crps_for_ensemble(fcst, obs, 'ensemble', lambda x: np.maximum(x, 0.5))

    """
    obs = chaining_func(obs, **chainging_func_kwargs)
    fcst = chaining_func(fcst, **chainging_func_kwargs)

    result = crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim,
        method=method,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
        include_components=include_components,
    )
    return result


def tail_tw_crps_for_ensemble(
    fcst: XarrayLike,
    obs: XarrayLike,
    ensemble_member_dim: str,
    threshold: Union[XarrayLike, float],
    *,  # Force keywords arguments to be keyword-only
    tail: Literal["upper", "lower"] = "upper",
    method: Literal["ecdf", "fair"] = "ecdf",
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[XarrayLike] = None,
    include_components: Optional[bool] = False,
) -> XarrayLike:
    """
    Calculates the threshold weighted continuous ranked probability score (twCRPS)
    weighted for a tail of the distribution from ensemble input.

    A threshold weight of 1 is assigned for values of the tail and a threshold weight of 0 otherwise.
    The threshold value of where the tail begins is specified by the ``threshold`` argument.
    The ``tail`` argument specifies whether the tail is the upper or lower tail.
    For example, if we only care about values above 40 degrees C, we can set ``threshold=40`` and ``tail="upper"``.


    For more flexible weighting options and the relevant equations, see the
    :py:func:`scores.probability.tw_crps_for_ensemble` function.

    Args:
        fcst: Forecast data. Must have a dimension ``ensemble_member_dim``.
        obs: Observation data.
        ensemble_member_dim: the dimension that specifies the ensemble member or the sample
            from the predictive distribution.
        threshold: the threshold value for where the tail begins. It can either be a float
            for a single threshold or an xarray object if the threshold varies across
            dimensions (e.g., climatological values).
        tail: the tail of the distribution to weight. Either "upper" or "lower".
        method: Either "ecdf" or "fair". See :py:func:`scores.probability.tw_crps_for_ensemble`
            for more details.
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores. Note that
            these weights are different to threshold weighting which is done by decision
            threshold.
        include_components: If True, returns the twCRPS with underforecast and overforecast
            penalties, as well as the forecast spread term. See :py:func:`scores.probability.crps_for_ensemble`
            for more details on the decomposition.

    Returns:
        xarray object of twCRPS values that has been weighted based on the tail.

    Raises:
        ValueError: when ``tail`` is not one of "upper" or "lower".
        ValueError: when ``method`` is not one of "ecdf" or "fair".

    References:
        - Allen, S., Ginsbourger, D., & Ziegel, J. (2023). Evaluating forecasts for high-impact \
            events using transformed kernel scores. SIAM/ASA Journal on Uncertainty \
            Quantification, 11(3), 906-940. https://doi.org/10.1137/22M1532184. 
        - Allen, S. (2024). Weighted scoringRules: Emphasizing Particular Outcomes \
            When Evaluating Probabilistic Forecasts. Journal of Statistical Software, \
            110(8), 1-26. https://doi.org/10.18637/jss.v110.i08

    See also:
        :py:func:`scores.probability.tw_crps_for_ensemble`
        :py:func:`scores.probability.interval_tw_crps_for_ensemble`
        :py:func:`scores.probability.crps_for_ensemble`
        :py:func:`scores.probability.crps_cdf`

    Examples:
        Calculate the twCRPS for an ensemble where we assign a threshold weight of 1
        to thresholds above 0.5 and a threshold weight of 0 to thresholds below 0.5.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import tail_tw_crps_for_ensemble
        >>> fcst = xr.DataArray(np.random.rand(10, 10), dims=['time', 'ensemble'])
        >>> obs = xr.DataArray(np.random.rand(10), dims=['time'])
        >>> tail_tw_crps_for_ensemble(fcst, obs, 'ensemble', 0.5, tail='upper')

    """
    if tail not in ["upper", "lower"]:
        raise ValueError(f"'{tail}' is not one of 'upper' or 'lower'")
    if tail == "upper":

        def _chainingfunc(x, threshold=threshold):
            return np.maximum(x, threshold)

    else:

        def _chainingfunc(x, threshold=threshold):
            return np.minimum(x, threshold)

    result = tw_crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim,
        _chainingfunc,
        chainging_func_kwargs={"threshold": threshold},
        method=method,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
        include_components=include_components,
    )
    return result


def interval_tw_crps_for_ensemble(
    fcst: XarrayLike,
    obs: XarrayLike,
    ensemble_member_dim: str,
    lower_threshold: Union[xr.DataArray, float],
    upper_threshold: Union[xr.DataArray, float],
    *,  # Force keywords arguments to be keyword-only
    method: Literal["ecdf", "fair"] = "ecdf",
    reduce_dims: Optional[Sequence[str]] = None,
    preserve_dims: Optional[Sequence[str]] = None,
    weights: Optional[XarrayLike] = None,
    include_components: Optional[bool] = False,
) -> XarrayLike:
    """
    Calculates the threshold weighted continuous ranked probability score (twCRPS) for ensemble forecasts
    where the threshold weight is 1 on a specified interval and 0 otherwise.

    The threshold values that define the bounds of the interval are given by the
    ``lower_threshold`` and ``upper_threshold`` arguments.
    For example, if we only want to foucs on the temperatures between -10 and -20 degrees C
    where aircraft icing is most likely, we can set ``lower_threshold=-20`` and ``upper_threshold=-10``.


    For more flexible weighting options and the relevant equations, see the
    :py:func:`scores.probability.tw_crps_for_ensemble` function.

    Args:
        fcst: Forecast data. Must have a dimension ``ensemble_member_dim``.
        obs: Observation data.
        ensemble_member_dim: the dimension that specifies the ensemble member or the sample
            from the predictive distribution.
        lower_threshold: the threshold value for where the interval begins. It can either be a float
            for a single threshold or an xarray object if the threshold varies across
            dimensions (e.g., climatological values).
        upper_threshold: the threshold value for where the interval ends. It can either be a float
            for a single threshold or an xarray object if the threshold varies across
            dimensions (e.g., climatological values).
        method: Either "ecdf" or "fair". See :py:func:`scores.probability.tw_crps_for_ensemble`
            for more details.
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        weights: Weights for calculating a weighted mean of individual scores. Note that
            these weights are different to threshold weighting which is done by decision
            threshold.
        include_components: If True, returns the twCRPS with underforecast and overforecast
            penalties, as well as the forecast spread term. See :py:func:`scores.probability.crps_for_ensemble`
            for more details on the decomposition.

    Returns:
        xarray object of twCRPS values where the threshold weighting is based on an interval.

    Raises:
        ValueError: when ``lower_threshold`` is not less than ``upper_threshold``.
        ValueError: when ``method`` is not one of "ecdf" or "fair".

    References:
        - Allen, S., Ginsbourger, D., & Ziegel, J. (2023). Evaluating forecasts for high-impact \
            events using transformed kernel scores. SIAM/ASA Journal on Uncertainty \
            Quantification, 11(3), 906-940. https://doi.org/10.1137/22M1532184. 
        - Allen, S. (2024). Weighted scoringRules: Emphasizing Particular Outcomes \
            When Evaluating Probabilistic Forecasts. Journal of Statistical Software, \
            110(8), 1-26. https://doi.org/10.18637/jss.v110.i08
    See also:
        :py:func:`scores.probability.tw_crps_for_ensemble`
        :py:func:`scores.probability.tail_tw_crps_for_ensemble`
        :py:func:`scores.probability.crps_for_ensemble`
        :py:func:`scores.probability.crps_cdf`

    Examples:
        Calculate the twCRPS for an ensemble where we assign a threshold weight of 1
        to thresholds between -20 and -10 and a threshold weight of 0 to thresholds outside
        that interval.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.probability import interval_tw_crps_for_ensemble
        >>> fcst = xr.DataArray(np.random.uniform(-40, 20, size=(30, 15)), dims=['time', 'ensemble'])
        >>> obs = xr.DataArray(np.random.uniform(-40, 20, size=30), dims=['time'])
        >>> interval_tw_crps_for_ensemble(fcst, obs, 'ensemble', -20, 10)
    """
    if isinstance(lower_threshold, xr.DataArray) or isinstance(upper_threshold, xr.DataArray):
        if (lower_threshold >= upper_threshold).any().values.item():  # type: ignore  # mypy is wrong, I think
            raise ValueError("`lower_threshold` must be less than `upper_threshold`")
    elif lower_threshold >= upper_threshold:
        raise ValueError("`lower_threshold` must be less than `upper_threshold`")

    def _chaining_func(x, lower_threshold=lower_threshold, upper_threshold=upper_threshold):
        return np.minimum(np.maximum(x, lower_threshold), upper_threshold)

    result = tw_crps_for_ensemble(
        fcst,
        obs,
        ensemble_member_dim,
        _chaining_func,
        chainging_func_kwargs={"lower_threshold": lower_threshold, "upper_threshold": upper_threshold},
        method=method,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
        include_components=include_components,
    )
    return result
