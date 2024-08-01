"""
Functions for calculating the probability integral transform (PIT) and
assessing probabilistic calibration of predictive distributions.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from scores.probability.crps_impl import crps_cdf_reformat_inputs
from scores.processing.cdf.cdf_functions import (
    add_thresholds,
    check_cdf,
    check_cdf_support,
    expectedvalue_from_cdf,
    fill_cdf,
    integrate_square_piecewise_linear,
    observed_cdf,
    round_values,
    variance_from_cdf,
)
from scores.utils import dims_complement


def pit(
    fcst_cdf: xr.DataArray,
    obs: xr.DataArray,
    fcst_threshold_dim: str = "threshold",
    possible_pointmass: Optional[float] = None,
    included_pit_thresholds: Optional[Sequence[float]] = None,
    pit_precision: float = 0,
):  # pylint: disable=too-many-locals
    """
    Calculates the probability integral transforms (PITs) of an array of predictive distributions
    `fcst_cdf` and corresponding observations `obs`. Here, the PIT of a forecast-observation pair
    is interpreted as an empirical CDF (Taggart 2022, c.f., Gneiting et al 2013).
    This generalisation allows the forecast to be a CDF which may have points of discontinuity.
    Given 0 < alpha < 1, value pit_cdf(alpha) is the probability that the observation fell at or
    below the alpha-quantile of the predictive distribution, for a randomly selected
    forecast-observation pair.

    Predictive CDFs can be continuous or discontinuous. If the CDF is discontinuous at a point x,
    the CDF is said to have 'point mass at x'. This implementation of the PIT assumes that
    `fcst_cdf` is continuous everywhere, except possibly at a single threshold `possible_pointmass` of
    possible point mass, where it is assumed that  F(t) = 0 whenever t < possible_pointmass.

    For wind (m/s) and rainfall (mm) forecasts, set `possible_pointmass=0`.
    For temperature (degrees Celsius) forecasts, typically set `possible_pointmass=None`.

    Whenever there are at least two non-NaN values in `fcst_cdf` along `fcst_threshold_dim`,
    any NaNs in `fcst_cdf` will be replaced by linearly interpolating and extrapolating
    values along `fcst_threshold_dim`, then clipping to 0 and 1, prior to calculating the PIT.
    Use `jive.metrics.cdfs.propagate_nan` on `fcst_cdf` or `jive.metrics.cdfs.fill_cdf` to
    deal with NaNs prior to using `pit` if this behaviour is undesirable.

    The returned array gives PIT CDFs for every observation and every predictive CDF, even where
    the observation is not paired to a predictive CDF or vice versa. In the case that there is
    no pairing, the resulting PIT CDF consists purely of NaN values.

    Args:
        fcst_cdf: the forecast CDFs, with threshohold dimension `fcst_threshold_dim`.
        obs: the corresponding observations, not in CDF format. Would have the same shape
            as `fcst_cdf` if `fcst_threshold_dim` was collapsed.
        fcst_threshold_dim: name of the dimension indexing the threshold in `fcst_cdf`.
        possible_pointmass: forecast threshold at which point mass is possible.
            `None` if no point mass is possible.
        included_pit_thresholds: thresholds at which to report the value of the PIT CDF. If
            `None`, then the thresholds `numpy.linspace(0, 1, 201)` will be included.
        pit_precision: round PIT values to specified precision (e.g. to nearest 0.05).
            Set to 0 for no rounding.

    Returns:
        xr.DataArray of PIT CDFs, with thresholds along the 'pit_threshold' dimension.
        'pit_threshold' values are the union of `included_pit_thresholds` and (rounded)
        PIT values. Here 'PIT values' are the values of the forecast CDFs at the observations.

    Raises:
        ValueError: if `fcst_threshold_dim` is not a dimension of `fcst_cdf`.
        ValueError: if `fcst_cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if fcst_cdf[fcst_threshold_dim]` coordinates are not increasing.
        ValueError: if `fcst_threshold_dim` is a dimension of `obs`.
        ValueError: if dimensions of `obs` are not also dimensions of `fcst_cdf`.
        ValueError: if `pit_precision` is negative.

    References:
        - Robert Taggart, "Assessing calibration when predictive distributions have
          discontinuities", Bureau Research Report, 64, 2022.
          http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf
        - Tilmann Gneiting and Roopesh Ranjan, "Combining predictive distributions."
          Electron. J. Statist. 7 1747 - 1782, 2013. https://doi.org/10.1214/13-EJS823
    """

    _check_pit_args(fcst_cdf, obs, fcst_threshold_dim, pit_precision)

    # reformat fcst_cdf, obs so they have same shape and are both in CDF format
    cdf_fcst, cdf_obs, _ = crps_cdf_reformat_inputs(
        fcst_cdf,
        obs,
        fcst_threshold_dim,
        threshold_weight=None,
        additional_thresholds=None,
        fcst_fill_method="linear",
        threshold_weight_fill_method="step",  # this value doesn't matter
    )

    # get PIT, interpreted as single values rather than as CDFs
    pit_singlevalues = cdf_fcst.where(np.isclose(cdf_obs, 1)).min(fcst_threshold_dim)

    included_pit_thresholds = included_pit_thresholds if included_pit_thresholds is not None else np.linspace(0, 1, 201)

    # get all the PIT thresholds for the returned array
    final_pit_thresholds = np.concatenate(
        (
            included_pit_thresholds,
            round_values(pit_singlevalues, pit_precision).values.flatten(),
        )
    )
    final_pit_thresholds = np.sort(pd.unique(final_pit_thresholds))
    # remove NaNs
    final_pit_thresholds = final_pit_thresholds[~np.isnan(final_pit_thresholds)]

    # identify pit values where observation fell in fcst point mass
    pit_singlevalues2, obs2 = xr.broadcast(pit_singlevalues, obs)
    is_pit_pointmass = (np.isclose(obs2, possible_pointmass)) & (pit_singlevalues2 > 0)

    # calculate the PIT CDF associated with non-point mass
    # first, calculate all PIT CDFs, assuming non-pointmass
    pit_cdf_no_ptmass = _pit_no_ptmass(
        pit_singlevalues,
        final_pit_thresholds,
        pit_precision=pit_precision,
    )

    # then, nan PIT CDFs associated with point mass, via broadcasting
    dab_pit_cdf_no_ptmass, dab_is_pit_pointmass = xr.broadcast(pit_cdf_no_ptmass, is_pit_pointmass)

    dab_pit_cdf_no_ptmass = dab_pit_cdf_no_ptmass.where(~dab_is_pit_pointmass)

    # now deal with PIT CDFs associated with point mass
    pit_cdf_ptmass = _pit_ptmass(pit_singlevalues, final_pit_thresholds)
    pit_cdf_ptmass = pit_cdf_ptmass.where(dab_is_pit_pointmass)

    # now combine the two different PIT CDF cases
    pit_cdf = pit_cdf_ptmass.combine_first(pit_cdf_no_ptmass)

    return pit_cdf


def _check_pit_args(
    fcst_cdf: xr.DataArray,
    obs: xr.DataArray,
    fcst_threshold_dim: str,
    pit_precision: float,
):
    """Checks that args of `pit` are valid."""

    check_cdf(fcst_cdf, fcst_threshold_dim, None, "fcst_cdf", "fcst_threshold_dim", "dims")

    if fcst_threshold_dim in obs.dims:
        raise ValueError(f"'{fcst_threshold_dim}' is a dimension of `obs`")

    if not set(obs.dims).issubset(fcst_cdf.dims):
        raise ValueError("Dimensions of `obs` must be a subset of dimensions of `fcst_cdf`")

    if pit_precision < 0:
        raise ValueError("`pit_precision` must be nonnegative")


def _pit_ptmass(pit_singlevalues: xr.DataArray, pit_thresholds: np.array) -> xr.DataArray:
    """
    Calculates the PIT CDF for a forecast-observation pair assuming that:

        - the observation fell where there is (possible) point mass x in the forecast CDF
        - the forecast CDF satisfies fcst_cdf(t) = 0 whenever t < x.

    In this case, the PIT CDF is of the form

        `t -> min(t/fcst_cdf(x), 1)`

    where x is the point at which the forecast CDF has point mass.
    If fcst_cdf(x) = 0 then the PIT CDF is 1.

    Args:
        pit_singlevalues: the value of the forecast CDF at the observation.
            Here the observation is at the location of forecast point mass.
            Values of `pit_singlevalues` are assumed to be in the closed interval [0,1], apart from any NaN.
        pit_thresholds: 1-dimensional array containing thresholds for the returned array of PIT CDF values.
            Elements are assumed to have been sorted and without replication.

    Returns:
        An xr.DataArray of PIT CDF values with threshold dimension 'pit_threshold'.
    """

    da_pit_thresholds = xr.DataArray(
        data=pit_thresholds,
        dims=["pit_threshold"],
        coords={"pit_threshold": pit_thresholds},
    )

    _, da_pit_thresholds = xr.broadcast(pit_singlevalues, da_pit_thresholds)

    pit_cdf = (da_pit_thresholds / pit_singlevalues).clip(0, 1)

    pit_cdf = pit_cdf.where(pit_singlevalues != 0, 1)

    return pit_cdf


def _pit_no_ptmass(pit_singlevalues: xr.DataArray, pit_thresholds: np.array, pit_precision=0) -> xr.DataArray:
    """
    Calculates values of the PIT CDF assuming that the observation fell where there is
    no point mass in the forecast CDF. In this case,

        - pit_cdf(t) = 0 when t < obs
        - pit_cdf(t) = 1 when t >= obs.

    Args:
        pit_singlevalues: the value of the forecast CDF at the observation.
        pit_thresholds: 1-dimensional array of thresholds in the returned PIT CDF array
            at which PIT CDF values are reported.
        pit_precision: precision at which to round pit_singlevalues
            prior to constructing PIT CDF. If `precision=0` then no rounding is performed.

    Returns:
        PIT CDFs as an xr.DataArray with thresholds in dimension 'pit_threshold'.
    """

    pit_cdf = observed_cdf(
        pit_singlevalues,
        "pit_threshold",
        threshold_values=pit_thresholds,
        include_obs_in_thresholds=True,
        precision=pit_precision,
    )

    return pit_cdf


def pit_histogram_values(
    pit_cdf: xr.DataArray,
    pit_threshold_dim: str = "pit_threshold",
    n_bins: int = 10,
    dims: Optional[Sequence[str]] = None,
) -> xr.DataArray:
    """
    The PIT histogram is a histogram showing how many times PIT values fall in a bin.
    This function gives the relative frequency of binned PIT 'values', based on PIT CDFs.
    If MPC denotes the mean PIT CDF, then the relative frequency for a bin with interval (a, b]
    is given by MPC(b) - MPC(a).

    The unit interval [0,1] is partitioned into bins of equal length.
    Each bin is is an interval of the form (a,b], except for the first
    bin which is of the form [0,b]. If a bin endpoint is not among the `pit_threshold_dim` coordinates,
    then the value of the PIT CDF at that endpoint is estimated using linear interpolation and/or
    linear extrapolation, clipped to the interval [0,1]. Any NaNs in the input `pit_cdf`
    will likewise be filled, unless there are fewer than two non-NaN values along the `pit_threshold_dim`
    dimension.

    Args:
        pit_cdf: values of the PIT CDFs, with threshold dimension `pit_threshold_dim`.
        pit_threshold_dim: the name of the dimension in `pit_cdf` that has PIT threshold values.
        n_bins: the number of bins.
        dims: the dimensions of `pit_cdf` to be preserved in the output, apart from `pit_threshold_dim`.
            All other dimensions in `pit_cdf`, including `pit_threshold_dim`, are collapsed.

    Returns:
        xr.DataArray with relative frequency of PIT values for each bin.

    Raises:
        ValueError: if `pit_threshold_dim` is not a dimension of `pit_cdf`.
        ValueError: if coordinates of `pit_cdf[pit_threshold_dim]` are not increasing.
        ValueError: if `pit_cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if `pit_threshold_dim` is in `dims`.
        ValueError: if `dims` is not a subset of `pit_cdf.dims`.
        ValueError: if `pit_cdf` is not 1 when `pit_cdf[pit_threshold_dim] > 1`.
        ValueError: if `pit_cdf` is not 0 when `pit_cdf[pit_threshold_dim] < 0`.
        ValueError: if `n_bins` is not positive.
    """
    _check_args_with_pits(pit_cdf, pit_threshold_dim, dims)
    if n_bins < 1:
        raise ValueError("`n_bins` must be at least 1")

    bin_endpoints = np.linspace(0, 1, n_bins + 1)

    dims_to_collapse = dims_complement(pit_cdf, dims=dims)
    dims_to_collapse.remove(pit_threshold_dim)

    mean_pitcdf = pit_cdf.mean(dims_to_collapse)

    # calculate the value of the mean PIT CDF at the bin endpoints
    # linear interpolation/extrapolation is used if endpoints are not in the threshold coordinates
    mean_pitcdf = add_thresholds(mean_pitcdf, pit_threshold_dim, bin_endpoints, "linear")

    # just get mean PIT CDF values at the endpoints
    mean_pitcdf = mean_pitcdf.where(mean_pitcdf[pit_threshold_dim].isin(bin_endpoints)).dropna(
        pit_threshold_dim, how="all"
    )

    diffs = mean_pitcdf - mean_pitcdf.shift(**{pit_threshold_dim: 1})

    # frequency in each bin, indexed by the right endpoint of the bin
    result = diffs.sel(**{pit_threshold_dim: bin_endpoints[1:]})

    # need to count frequency for PIT=0 as well. Will put this in the first bin
    result = result.where(result[pit_threshold_dim] != bin_endpoints[1])
    result = result.combine_first(mean_pitcdf.sel(**{pit_threshold_dim: bin_endpoints[1:]}))

    result = result.rename({pit_threshold_dim: "right_endpoint"})
    result = result.assign_coords(
        left_endpoint=result.right_endpoint - 1 / n_bins,
        bin_centre=result.right_endpoint - 1 / (2 * n_bins),
    )

    return result


def _check_args_with_pits(pit_cdf, pit_threshold_dim, dims):
    """Checks that arguments involving PIT CDFs are valid."""
    check_cdf(pit_cdf, pit_threshold_dim, dims, "pit_cdf", "pit_threshold_dim", "dims")
    check_cdf_support(pit_cdf, pit_threshold_dim, 0, 1, "pit_cdf", "pit_threshold_dim")


def pit_scores(
    pit_cdf: xr.DataArray,
    pit_threshold_dim: str = "pit_threshold",
    dims: Optional[Sequence[str]] = None,
) -> xr.Dataset:
    """
    Given an array of PIT CDFs (probability integral transform in CDF format),
    calculates the combined PIT CDF over the specified dimensions. Returns a
    "calibration score" for the mean PIT CDF, as well as the expected value and the
    variance of a random variable distributed by each mean PIT CDF.

    The calibration score of a combined PIT CDF is a measure of the degree of probabilistic
    miscalibration of the forecast cases used to construct the mean_pit_cdf. Specifically,

        `score = integral((combined_pit_cdf(x) - x)^2, 0 <= x <= 1)`

    so that the score measures the deviation of the mean_pit_cdf from the standard uniform CDF.
    A lower score is better.

    If Y is a random variable distributed by combined_pit_cdf, then the expected value E(Y)
    measures the extent to which the forecasts had an over-forecast tendency (E(Y) < 0.5)
    or an under-forecast tendency (E(Y) > 0.5).

    If Y ~ combined_pit_cdf, then the variance Var(Y) is a measure of dispersion of the forecast
    system. If Var(Y) < 1/12 then the system is over-dispersed  (i.e. its predictive distributions
    are too wide) while if Var(Y) > 1/12 then the system is under-dispersed (i.e. its predictive
    distributions are too narrow).

    See Taggart (2022) for further information about this score (labelled there as PS_2) and its
    decomposition.


    Whenever there are at least two non-NaN values in `pit_cdf` along `pit_threshold_dim`,
    any NaNs in `pit_cdf` will be replaced by linearly interpolating and extrapolating
    values along `pit_threshold_dim`, then clipping to 0 and 1, prior to calculating output.
    Use `jive.metrics.cdfs.propagate_nan` on `pit_cdf` or `jive.metrics.cdfs.fill_cdf` to
    deal with NaNs prior to using `pit_scores` if this behaviour is undesirable.

    Args:
        pit_cdf: array of PITs in CDF form, with threshold dimension `pit_threshold_dim`.
        pit_threshold_dim: the name of the threshold dimension of `pit_cdf`.
        dims: The dimensions of `pit_cdf` to be preserved in the output, apart from `pit_threshold_dim`.
            All other dimensions in `pit_cdf`, including `pit_threshold_dim`, are collapsed.

    Returns:
        An xr.Dataset with data variables "score", "expectation" and "variance".

    Raises:
        ValueError: if `pit_threshold_dim` is not a dimension of `pit_cdf`.
        ValueError: if coordinates of `pit_cdf[pit_threshold_dim]` are not increasing.
        ValueError: if `pit_cdf` has non-NaN values outside the closed interval [0,1].
        ValueError: if `pit_threshold_dim` is in `dims`.
        ValueError: if `dims` is not contained in `pit_cdf.dims`.
        ValueError: if `pit_cdf` is not 1 when `pit_cdf[pit_threshold_dim] > 1`.
        ValueError: if `pit_cdf` is not 0 when `pit_cdf[pit_threshold_dim] < 0`.

    References:
        Robert Taggart, "Assessing calibration when predictive distributions have discontinuities",
        Bureau Research Report, 64, 2022.
        http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf
    """
    _check_args_with_pits(pit_cdf, pit_threshold_dim, dims)

    # only interested in PIT thresholds between 0 and 1 inclusive.
    pit_cdf = pit_cdf.sel({pit_threshold_dim: slice(0, 1)})

    pit_cdf = fill_cdf(pit_cdf, pit_threshold_dim, "linear", 2)

    dims_to_collapse = dims_complement(pit_cdf, dims=dims)
    dims_to_collapse.remove(pit_threshold_dim)  # this dim will be collapsed later
    pit_cdf = pit_cdf.mean(dims_to_collapse)

    # will compute integral(func(x)^2) where func(x) = mean_pit_cdf(x) - x
    std_uniform_cdf = xr.DataArray(
        data=pit_cdf[pit_threshold_dim],
        dims=[pit_threshold_dim],
        coords={pit_threshold_dim: pit_cdf[pit_threshold_dim]},
    )

    func = pit_cdf - std_uniform_cdf
    func = func.where(std_uniform_cdf >= 0).where(std_uniform_cdf <= 1)

    score = integrate_square_piecewise_linear(func, pit_threshold_dim).rename("score")

    expectation = expectedvalue_from_cdf(pit_cdf, pit_threshold_dim).rename("expectation")
    variance = variance_from_cdf(pit_cdf, pit_threshold_dim).rename("variance")

    result = xr.merge([score, expectation, variance])

    return result
