"""
Methods for the probability integral transform (PIT) class.

Reserved dimension names:
- 'uniform_endpoint', 'pit_x_value'
"""

from typing import Optional

import numpy as np
import xarray as xr

from scores.functions import apply_weights
from scores.processing.cdf import add_thresholds
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions

RESERVED_NAMES = {
    "uniform_endpoint",
    "pit_x_value",
    "x_plotting_position",
    "y_plotting_position",
    "plotting_point",
    "bin_left_endpoint",
    "bin_right_endpoint",
    "bin_centre",
}


class Pit_for_ensemble:
    """
    Given ensemble forecasts and corresponding observations, calculates the probability
    intergral transform (PIT) for the set of forecast cases. The calculated PIT can be a
    (possibly weighted) average over specified dimensions. Calculations are performed and
    interpreted as follows.

    Each forecast case (i.e., an ensemble of forecasts) is interpreted as an empirical
    cumulative distribution function (CDF) :math:`G`. Given any observation :math:`y`,
    the PIT value at :math:`G` is a uniform distribution on the closed interval
    :math:`[G(y-), G(y)]`, where :math:`G(y-)` denotes the left-hand limit of :math:`G`
    at :math:`y`. This is the most general form of PIT and handles cases where the
    observation :math:`y` coincides with a point of discontinuity of a predictive CDF
    :math:`G`. See Gneiting and Ranjan (2013) and Taggart (2022).

    The (possibly weighted) mean over a specified set of dimensions is the weighted mean
    of all the PIT values (each interpreted as a uniform distribution), with weighting
    rescaled if necessary so that the weighted mean is also a distribution.

    Attributes:
        left: values for the left-hand limit of the PIT (represented as a CDF)
        right: values for the PIT (represented as a CDF), which also equals the right-hand limit

    Methods:
        attribute_name (type): Description of the attribute.

    References:
        - Gneiting, T., & Ranjan, R. (2013). Combining predictive distributions. Electron. J. Statist. 7: 1747-1782 \
            https://doi.org/10.1214/13-EJS823
        - Taggart, R. J. (2022). Assessing calibration when predictive distributions have discontinuities. \
            Bureau Research Report 64, http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf
    """

    def __init__(
        self,
        fcst: XarrayLike,
        obs: XarrayLike,
        ens_member_dim: str,
        *,  # Force keywords arguments to be keyword-only
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
        weights: Optional[XarrayLike] = None,
    ):
        """
        Calculates the mean PIT :math:`F`, interpreted as a CDF, given the set of forecast and
        observations pairs.

        The CDF :math:`F` is completely determined by its values :math:`F(x)` and left-hand limits
        :math:`F(x-)` at a minimal set of points :math:`x` that are output as part of this
        calculation in the dimension "pit_x_value". All other values may be obtained via
        interpolation whenever :math:`0 < x < 1` or the fact that :math:`F(x) = 0` when
        :math:`x < 1` and :math:`F(x) = 1` when :math:`x > 1`.

        The outputs in the ``left`` and ``right`` attributes are sufficient to calculate
        precise statistics for the PIT for the set of forecasts and observations, as
        provided by ``Pit_for_ensemble`` methods.

        Args:
            fcst: an xarray object of ensemble forecasts, containing the dimension
                `ens_member_dim`.
            obs: an xarray object of observations.
            ens_member_dim: name of the ensemble member dimension in ``fcst``.
            reduce_dims: Optionally specify which dimensions to reduce when calculating the
                PIT CDF values, where the mean is taken over all forecast cases.
                All other dimensions will be preserved. As a special case, 'all' will allow
                all dimensions to be reduced. Only one of ``reduce_dims`` and ``preserve_dims``
                can be supplied. The default behaviour if neither are supplied is to reduce all dims.
            preserve_dims: Optionally specify which dimensions to preserve when calculating the
                PIT CDF values, where the mean is taken over all forecast cases.
                All other dimensions will be reduced. As a special case, 'all' will allow
                all dimensions to be preserved, apart from ``severity_dim`` and ``prob_threshold_dim``.
                Only one of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default
                behaviour if neither are supplied is to reduce all dims.
            weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
                by population, custom) of PIT CDF values across all forecast cases.

        Attributes:
            left: xarray object representing the mean PIT, interpreted as a CDF :math:`F`
                and evaluated as left-hand limits at the points :math:`x` in the dimension
                "pit_x_value". That is, values in the array or dataset are of the form :math:`F(x-)`.
            right: xarray object representing the mean PIT, interpreted as a CDF :math:`F`
                and evaluated as at the points :math:`x` in the dimension "pit_x_value".
                That is, values in the array or dataset are of the form :math:`F(x)`.

        Methods:
            plotting_points: generate plotting points for PIT-uniform probability plots.

        Raises:
            ValueError if dimenions of ``fcst``, ``obs`` or ``weights`` contain any of the following reserved names:
                'uniform_endpoint', 'pit_x_value', 'x_plotting_position', 'y_plotting_position', 'plotting_point'
        """
        pit_cdf = pit_cdfvalues(
            fcst,
            obs,
            ens_member_dim,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
            weights=weights,
        )
        self.left = pit_cdf["left"]
        self.right = pit_cdf["right"]

    def plotting_points(self) -> XarrayLike:
        """
        Returns the plotting points for PIT-uniform probability plots, or equivalently,
        the plotting points for the PIT CDF. The returned output is a dictionary with
        two keys "x_plotting_position" and "y_plotting_position", and values being xarray
        objects with plotting position data.

        Points on the plot are given by :math:`(x(t),y(t))`, where :math:`x(t)` is a value from
        the "x_plotting_position" output, :math:`y(t)` is a corresponding value from
        the "y_plotting_position" output, and :math:`t` is one of the coordinates from the
        "plotting_point" dimension.

        To construct PIT-uniform probability plots, plot the points :math:`(x(t),y(t))`
        for increasing :math:`y(t)` and fill the remaining gaps using linear interpolation.
        """
        return _get_plotting_points(self.left, self.right)


def _pit_values_for_ensemble(fcst: XarrayLike, obs: XarrayLike, ens_member_dim: str) -> XarrayLike:
    """
    For each forecast case in the form of an ensemble, the PIT value of the ensemble for
    the corresponding observation is a uniform distribution over the closed interval
    [lower,upper], where
        lower = (count of ensemble members strictly less than the observation) / n
        upper = (cont of ensemble members not exceeding the observation) / n
        n = size of the ensemble.

    Returns an array of [lower,upper] values in the dimension 'uniform_endpoint'.

    Args:
        fcst: array of forecast values, including dimension `ens_member_dim`
        obs: array of forecast values, excluding `ens_member_dim`
        ens_member_dim: name of the ensemble member dimension in `fcst`

    Returns:
        array of PIT values in the form [lower,upper], with dimensions
        'uniform_endpoint', all dimensions in `obs` and all dimensions in `fcst`
        excluding `ens_member_dim`.
    """
    ensemble_size = fcst.count(ens_member_dim).where(obs.notnull())
    pit_lower = (fcst < obs).sum(ens_member_dim) / ensemble_size
    pit_upper = (fcst <= obs).sum(ens_member_dim) / ensemble_size

    pit_lower = pit_lower.assign_coords(uniform_endpoint="lower").expand_dims("uniform_endpoint")
    pit_upper = pit_upper.assign_coords(uniform_endpoint="upper").expand_dims("uniform_endpoint")

    return xr.concat([pit_lower, pit_upper], "uniform_endpoint")


def _get_pit_x_values(pit_values: XarrayLike) -> xr.DataArray:
    """
    Returns a data array of consisting of exactly those x-axis values needed for
    constructing a unifornm PIT probability plot for the given array of `pit_values`.

    Args:
        pit_values: output from `_pit_values_for_ensemble`

    Returns:
        xarray data array of x-axis values, indexed by the same values on the dimension
        'pit_x_value'
    """
    if isinstance(pit_values, xr.Dataset):
        pit_values = pit_values.to_dataarray()

    x_values = np.unique(pit_values)
    x_values = np.unique(np.concatenate((np.array([0.0, 1.0]), x_values)))
    x_values = x_values[~np.isnan(x_values)]
    x_values = xr.DataArray(data=x_values, dims=["pit_x_value"], coords={"pit_x_value": x_values})
    return x_values


def _pit_cdfvalues_for_jumps(pit_values: XarrayLike, x_values: xr.DataArray) -> dict:
    """
    Gives the values F(x) where F is the CDF for a pit value,
    and x comes from `x_values`, given that the CDF jumps at x. This occurs precisely
    when upper == lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    It is assumed that `x_values` contains all the values in `pit_values`.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`
        x_values: xr.DataArray of x values, with dimension 'pit_x_value', output from
            `_get_pit_x_values`

    Returns:
        dictionary of cdf values, with keys 'left' and 'right',
        representing the left and right hand limits of the cdf values F(x).
        Each value in the dictionary is an xarray object representing limits of F(x),
        where x is represented by values in the 'pit_x_value' dimension.
    """
    lower_values = pit_values.sel(uniform_endpoint="lower").drop_vars("uniform_endpoint")
    upper_values = pit_values.sel(uniform_endpoint="upper").drop_vars("uniform_endpoint")
    # get the cases where jumps occur
    pit_jumps = upper_values.where(upper_values == lower_values)
    # pit_jumps, xs = xr.broadcast(pit_jumps, xs)
    cdf_left = xr.zeros_like(pit_jumps).where(x_values <= pit_jumps, 1).where(pit_jumps.notnull())
    cdf_right = xr.zeros_like(pit_jumps).where(x_values < pit_jumps, 1).where(pit_jumps.notnull())
    return {"left": cdf_left, "right": cdf_right}


def _pit_cdfvalues_for_unif(pit_values: XarrayLike, x_values: xr.DataArray) -> XarrayLike:
    """
    Gives the values F(x) where F is the CDF for a pit value,
    and x comes from `x_values`, given that the CDF is uniform. This occurs precisely
    when upper > lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    Left-hand and right-hand limits of F(x) are equal in this case.

    It is assumed that `x_values` containes all the values in `pit_values`.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`
        x_values: xr.DataArray of x values, with dimension 'pit_x_value', output from
            `_get_pit_x_values`

    Returns:
        An xarray object representing values of F(x), where x is represented by values
        in the 'pit_x_value' dimension.
    """
    lower_values = pit_values.sel(uniform_endpoint="lower").drop_vars("uniform_endpoint")
    upper_values = pit_values.sel(uniform_endpoint="upper").drop_vars("uniform_endpoint")
    # get the cases where the cdf is a uniform distribution over [a,b], a < b
    unif_cases = upper_values > lower_values
    # get the upper and lower values: these correspond to a, b
    pit_unif_upper = upper_values.where(unif_cases)  # the b
    pit_unif_lower = lower_values.where(unif_cases)  # the a
    # calculate the cdf of Unif(a,b) at the values in x
    pit_unif = (
        xr.zeros_like(pit_unif_upper)
        .where(x_values <= pit_unif_lower)  # 0 for x <= a
        .where(x_values < pit_unif_upper, 1)  # 1 for x >= b
        .interpolate_na("pit_x_value")  # interpolate in between
        .where(unif_cases)  # preserve nans where appropriate
    )
    return pit_unif


def _pit_cdfvalues(pit_values: XarrayLike) -> dict:
    """
    Calculates F(x) for each CDF F representing the PIT value for a forecast
    case. The x values used are all values where there is a non-linear change
    at least one F from the collection of Fs. Intermediate values of F can be recovered
    using linear interpolation.

    The values are output as a dictionary of two xarray object representing left hand
    and right hand limits of F at the point x.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`

    Returns:
        dictionary of cdf values, with keys 'left' and 'right',
        representing the left and right hand limits of the cdf values F(x).
        Each value in the dictionary is an xarray object representing limits of F(x),
        where x is represented by values in the 'pit_x_value' dimension.
    """
    # get the x values
    x_values = _get_pit_x_values(pit_values)

    # get cdf values where the pit cdf jumps
    cdf_at_jump_cases = _pit_cdfvalues_for_jumps(pit_values, x_values)
    # get the cdf values where the pit is uniform
    cdf_at_unif_cases = _pit_cdfvalues_for_unif(pit_values, x_values)
    # combine
    cdf_right = cdf_at_jump_cases["right"].combine_first(cdf_at_unif_cases)
    cdf_left = cdf_at_jump_cases["left"].combine_first(cdf_at_unif_cases)

    return {"left": cdf_left, "right": cdf_right}


def _pit_dimension_checks(fcst: XarrayLike, obs: XarrayLike, weights: Optional[XarrayLike] = None):
    """
    Checks the dimensions of inputs to `pit_cdfvalues` to ensure that reserved names
    "uniform_endpoint" and "pit_x_value" are not used.
    """
    all_dims = set(fcst.dims).union(obs.dims)
    if weights is not None:
        all_dims = all_dims.union(weights.dims)
    if len([dim for dim in RESERVED_NAMES if dim in all_dims]) > 0:
        raise ValueError(
            f'The following names are reserved and should not be among the dimensions of \
            `fcst`, `obs` or `weight`: {", ".join(RESERVED_NAMES)}'
        )


def pit_cdfvalues(
    fcst: XarrayLike,
    obs: XarrayLike,
    ens_member_dim: str,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> dict:
    """
    Each forecast case (i.e., an ensemble of forecasts) is interpreted as an empirical
    cumulative distribution function (CDF) :math:`G`. Given any observation :math:`y`,
    the probability intergral transform (PIT) value at :math:`G` is a uniform distribution
    on the closed interval :math:`[G(y-), G(y)]`, where :math:`G(y-)` denotes the left-hand
    limit of :math:`G` at :math:`y`.

    This function outputs values of PIT by representing each uniform distribution on
    :math:`[G(y-), G(y)]` as the corresponding uniform CDF :math:`F`. We call :math:`F`
    the 'PIT CDF' for the forecast-observation pair :math:`(G,y)`. Values :math:`F(x-)`
    and :math:`F(x)` are given for an optimal set of points :math:`x` satisfying
    :math:`0 <= x <= 1`. The set is optimal in the sense that the value of :math:`F`
    elswhere can be determined via linear interpolation, whilst no smaller set of values
    has this property.

    Dimensions are be reduced by taking (possibly weighted) means of the CDF values
    :math:`F(x)`. The weighted means can still be interpreted a values of CDFs, and
    intermediate values still attained via linear interpolation.

    Args:
        fcst: an xarray object of ensemble forecasts, containing the dimension
            `ens_member_dim`.
        obs: an xarray object of observations.
        ens_member_dim: name of the ensemble member dimension in ``fcst``.
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            PIT CDF values, where the mean is taken over all forecast cases.
            All other dimensions will be preserved. As a special case, 'all' will allow
            all dimensions to be reduced. Only one of ``reduce_dims`` and ``preserve_dims``
            can be supplied. The default behaviour if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating the
            PIT CDF values, where the mean is taken over all forecast cases.
            All other dimensions will be reduced. As a special case, 'all' will allow
            all dimensions to be preserved, apart from ``severity_dim`` and ``prob_threshold_dim``.
            Only one of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default
            behaviour if neither are supplied is to reduce all dims.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom) of PIT CDF values across all forecast cases.

    Returns:
        a dictionary with the following keys and values:
        - "left": an xarray object containing the left-hand limits :math:`F(x-)` of the PIT CDF values
        - "right": an xarray object containing the values :math:`F(x)` of the PIT CDF values

    Raises:
        ValueError if dimenions of ``fcst``, ``obs`` or ``weights`` contain any of the following reserved names:
                'uniform_endpoint', 'pit_x_value', 'x_plotting_position', 'y_plotting_position', 'plotting_point'
    """
    _pit_dimension_checks(fcst, obs, weights)

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

    # PIT values in [G(y-), G(y)] format
    pit_values = _pit_values_for_ensemble(fcst, obs, ens_member_dim)
    # convert to F(x-), F(x) format
    cdf = _pit_cdfvalues(pit_values)

    cdf_left = apply_weights(cdf["left"], weights=weights).mean(dim=dims_for_mean)
    cdf_right = apply_weights(cdf["right"], weights=weights).mean(dim=dims_for_mean)

    # rescale CDFs so that their max value is 1.
    # This corrects for weights that don't sum to 1.
    cdf_right_max = cdf_right.max("pit_x_value")
    cdf_right = cdf_right / cdf_right_max
    cdf_left = cdf_left / cdf_right_max

    return {"left": cdf_left, "right": cdf_right}


def _get_plotting_points(left: XarrayLike, right: XarrayLike) -> dict:
    """
    Given outputs left and right from `pit_cdfvalues`, calculates the plotting positions
    for the PIT CDF (equivalently PIT-uniform probability plot).

    Args:
        left: "left" output from `pit_cdfvalues`, namely the left-hand limits of the PIT CDF
        right: "right" output from `pit_cdfvalues`, namely values of the PIT CDF

    Returns:
        dictionary with following keys and values:
            'x_plotting_position': xarray object with the x-axis plotting positions for each
                point, indexed by 'plotting_point'
            'y_plotting_position': xarray object with the y-axis plotting positions for each
                point, indexed by 'plotting_point'
    """
    # only keep coordinates in left where (left != right) every and where left is not NaN.
    # (We can assume that if one value is NaN for any particular forecast case than all values
    # are NaN for `left` and `right` in that forecast case)
    dims_for_any = [dim for dim in left.dims if dim != "pit_x_value"]
    different = ((left != right) & ~np.isnan(left)).any(dims_for_any)
    left_reduced = left.where(different).dropna("pit_x_value", how="all")

    # combine points from left and right
    y_values = xr.concat([left_reduced, right], "pit_x_value").sortby("pit_x_value")
    if isinstance(y_values, xr.Dataset):
        x_values = xr.merge([y_values[var]["pit_x_value"].rename(var) for var in y_values.data_vars])
    else:
        x_values = y_values["pit_x_value"]

    # refactor 'pit_x_value' dimension to ensure coordinates are unique
    y_values = y_values.assign_coords(pit_x_value=range(len(y_values["pit_x_value"]))).rename(
        {"pit_x_value": "plotting_point"}
    )
    x_values = x_values.assign_coords(pit_x_value=range(len(x_values["pit_x_value"]))).rename(
        {"pit_x_value": "plotting_point"}
    )

    return {"x_plotting_position": x_values, "y_plotting_position": y_values}


def _value_at_pit_cdf(pit_left: XarrayLike, pit_right: XarrayLike, point: float) -> XarrayLike:
    """
    Given mean PIT CDF `F`, with left-hand limit values `pit_left` and right-hand limit values
    `pit_right`, finds the value `F(point)` under the assumption that `point` is not in
    `pit_left["pit_x_value"]`.

    The result is attained via linear interpolation between the known values at the
    nearest right-hand limit (less than `point`) and the nearest left-hand limit
    (greater than `point`).

    Args:
        pit_left: `left` attribute of a `Pit_for_ensemble` object
        pit_right: `right` attribute of a `Pit_for_ensemble` object
        point: a float strictly between 0 and 1

    Returns:
        xarray object containing values of the CDF `F` at `point`, with same dimensions
        as `pit_left`

    Raises:
        ValueError if `point` is in `pit_left["pit_x_value"].values`
    """
    if point in pit_left["pit_x_value"].values:
        raise ValueError('`point` must not be a value in `pit_left["pit_x_value"]`')
    upper = np.min([x for x in pit_left["pit_x_value"].values if x > point])
    lower = np.max([x for x in pit_right["pit_x_value"].values if x < point])
    # get the values at upper and lower
    value = xr.concat([pit_right.sel(pit_x_value=lower), pit_left.sel(pit_x_value=upper)], "pit_x_value")
    # value at newpoint is attained via linear interpolation
    value = add_thresholds(value, "pit_x_value", [point], "linear").sel(pit_x_value=slice(point, point))
    return value


def _construct_hist_values(cdf_at_endpoints: list[XarrayLike], bin_width: float) -> XarrayLike:
    """
    Calculates the PIT histogram values given values of the PIT CDF at the endpoint of
    every histogram bin.

    Args:
        cdf_at_endpoints: a list of xarray objects with dimension "pit_x_value",
            each giving the value of the PIT CDF at an endpoint of one of the histogram bins.
        bin_width: width of each bin in the histogram.

    Returns:
        xarray object including dim 'bin_centre' and coordinates 'bin_left_endpoint',
        'bin_right_endpoint', with values the hight of each bar in the histogram.
    """
    # cdf values at all histogram bin endpoints
    cdf_at_endpoints = xr.concat(cdf_at_endpoints, "pit_x_value").sortby("pit_x_value")

    # calculate the histogram values
    histogram_values = cdf_at_endpoints.diff("pit_x_value")

    histogram_values = histogram_values.assign_coords(
        {
            "pit_x_value": histogram_values["pit_x_value"] - bin_width / 2,
            "bin_left_endpoint": histogram_values["pit_x_value"] - bin_width,
            "bin_right_endpoint": histogram_values["pit_x_value"],
        }
    ).rename({"pit_x_value": "bin_centre"})
    return histogram_values


def _pit_hist_left(pit_left: XarrayLike, pit_right: XarrayLike, bins: int) -> XarrayLike:
    """
    Gets the pit histogram values, with left endpoints included in every bin.
    e.g. if bins=5, the bins are [0, 0.2), [0.2, 0.4), ..., [0.8, 1]

    Args:
        pit_left: `left` attribute of a `Pit_for_ensemble` object
        pit_right: `right` attribute of a `Pit_for_ensemble` object
        bins: the number of bins in the histogram

    Returns:
        xarray object including dim 'bin_centre' and coordinates 'bin_left_endpoint',
        'bin_right_endpoint', with values the hight of each bar in the histogram.
    """
    bin_width = 1 / bins
    left_endpoints = [k * bin_width for k in range(bins)]

    # import pdb

    # pdb.set_trace()
    # the value of the mean PIT CDF at 1 is 1
    # using .sel is safe because the value 1.0 is guaranteed to be in pit_x_value (see _get_pit_x_values)
    value_at_1 = pit_right.sel(pit_x_value=slice(1.0, 1.0))

    # start collecting the value of the pit CDF at each of the endpoints of the histogram bins
    cdf_at_endpoints = [value_at_1]

    # get values of the PIT CDF that are aleady available in pit
    common_points = [x for x in left_endpoints if x in pit_left["pit_x_value"]]
    if len(common_points) > 0:
        cdf_at_common_points = pit_left.sel(pit_x_value=common_points)
        cdf_at_endpoints.append(cdf_at_common_points)

    # get values of the PIT CDF that are not available in pit
    new_points = [x for x in left_endpoints if x not in pit_left["pit_x_value"]]
    for new_point in new_points:
        value = _value_at_pit_cdf(pit_left, pit_right, new_point)
        cdf_at_endpoints.append(value)

    histogram_values = _construct_hist_values(cdf_at_endpoints, bin_width)
    return histogram_values
