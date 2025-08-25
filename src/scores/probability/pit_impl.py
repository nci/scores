"""
Methods for the probability integral transform (PIT) class.

Reserved dimension names:
- 'uniform_endpoint', 'pit_x_value'

Write checks for fcst cdf: between 0, 1, threshold dim increasing, cdf increasing???

_pit_values_for_cdf - only works for array so far, not dataset
"""

import warnings
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from scores.functions import apply_weights
from scores.probability.checks import cdf_values_within_bounds, coords_increasing
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


class Pit:
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
        left: values for the left-hand limit of the PIT distribution (represented as a CDF)
        right: values for the PIT distribution (represented as a CDF), which also equals the right-hand limit

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
        special_fcst_dim: str,
        *,  # Force keywords arguments to be keyword-only
        fcst_type: str = "ensemble",
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
                `special_fcst_dim`.
            obs: an xarray object of observations.
            special_fcst_dim: name of the ensemble member dimension in ``fcst`` if ``fcst_type='ensemble'``
                or of the CDF threshold dimension in ``fcst`` if ``fcst_type='cdf'``.
            fcst_type: either "ensemble" or "cdf".
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
            plotting_points: generate plotting points for PIT-uniform probability plots
            plotting_points_parametric: generate plotting points for PIT-uniform probability plots
                in parametric format
            hist_values: generate values for the PIT histogram
            alpha_score: calculates the 'alpha score', which is the absolute area between
                the diagonal and PIT curve of the PIT-uniform probability plot


        Raises:
            ValueError if dimenions of ``fcst``, ``obs`` or ``weights`` contain any of the following reserved names:
                'uniform_endpoint', 'pit_x_value', 'x_plotting_position', 'y_plotting_position', 'plotting_point'
        """
        if fcst_type not in ["ensemble", "cdf"]:
            raise ValueError('`fcst_type` must be one of "ensemble" or "cdf"')

        if fcst_type == "ensemble":
            pit_cdf = pit_distribution_for_ens(
                fcst,
                obs,
                special_fcst_dim,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
            )
        else:
            pit_cdf = pit_distribution_for_cdf(
                fcst,
                obs,
                special_fcst_dim,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
            )
        self.left = pit_cdf["left"]
        self.right = pit_cdf["right"]

    def plotting_points(self):
        """
        Returns an xarray object with the plotting points for PIT-uniform probability plots.
        Note that coordinates in the "pit_x_value" dimension will have duplicate values.
        For a parametric approach to plotting points without duplicate coordinate values, see the
        ``plotting_points_parametric`` method.
        """
        return xr.concat([self.left, self.right], "pit_x_value").sortby("pit_x_value")

    def plotting_points_parametric(self) -> dict:
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
        return _get_plotting_points_dict(self.left, self.right)

    def hist_values(self, bins: int, right: bool = True) -> XarrayLike:
        """
        Returns an xarray object with the PIT histogram values.

        Args:
            bins: the number of bins in the histogram.
            right: If True, histogram bins always include the rightmost edge. If False,
                bins always include the leftmost edge.
        """
        if right:
            return _pit_hist_right(self.left, self.right, bins)
        return _pit_hist_left(self.left, self.right, bins)

    def alpha_score(self) -> XarrayLike:
        """
        Calculates the so-called 'alpha score', which is a measure of how close the PIT distribution
        is to the uniform distribution. If the PIT CDF is :math:`F`, then the alpha score :math:`S`
        is given by
            :math:`\\int_0^1 |F(x) - x|\\,\\text{d}x}`
        """
        return _alpha_score(self.plotting_points())

    def expected_value(self) -> XarrayLike:
        """
        Calculates the expected value of the PIT distribution.
        An expected value greater than 0.5 indicates an under-prediction tendency.
        An expected value less than 0.5 indicates an over-prediction tendency.
        """
        return _expected_value(self.plotting_points())

    def variance(self) -> XarrayLike:
        """
        Calculates the variance of the PIT distribution.
        A variance greater than 1/12 indicates predictive under-dispersion.
        A variance less than 1/12 indicates predictive over-dispersion.
        """
        return _variance(self.plotting_points())


def _pit_values_for_ens(fcst: XarrayLike, obs: XarrayLike, ens_member_dim: str) -> XarrayLike:
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


def pit_distribution_for_cdf(
    fcst: Union[dict, XarrayLike],
    obs: XarrayLike,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> dict:
    """
    Each forecast case is an array of cumulative distribution function (CDF) values :math:`G(x)`,
    which is the probability that the random variable being forecast does not exceed :math:`x`.
    Given any observation :math:`y`, the probability intergral transform (PIT) value at :math:`G`
    is a uniform distribution on the closed interval :math:`[G(y-), G(y)]`, where :math:`G(y-)`
    denotes the left-hand limit of :math:`G` at :math:`y`.

    This function outputs values of PIT by representing each uniform distribution on
    :math:`[G(y-), G(y)]` as the corresponding uniform CDF :math:`F`. We call :math:`F`
    the 'PIT distribution' for the forecast-observation pair :math:`(G,y)`. Values :math:`F(x-)`
    and :math:`F(x)` are given for an optimal set of points :math:`x` satisfying
    :math:`0 <= x <= 1`. The set is optimal in the sense that the value of :math:`F`
    elswhere can be determined via linear interpolation, whilst no smaller set of values
    has this property.

    Dimensions are be reduced by taking (possibly weighted) means of the PIT distribution
    :math:`F(x)`. The weighted means can also be interpreted a values of CDFs, and exact
    intermediate values also attained via linear interpolation.

    Args:
        fcst: either an xarray object of CDF forecast values, containing the dimension `threshold_dim`,
            or a dictionary containing the keys "left" and "right", with corresponding values
            xarray objects giving the left-hand and right-hand limits of the fcst CDF
            along the dimension `threshold_dim`.
        obs: an xarray object of observations.
        threshold_dim: name of the threshold dimension in ``fcst``, such that the probability
            of not exceeding a particular threshold is one of the corresponding values of ``fcst``.
        reduce_dims: Optionally specify which dimensions to reduce when calculating the
            PIT distribution, where the mean is taken over all forecast cases.
            All other dimensions will be preserved. As a special case, 'all' will allow
            all dimensions to be reduced. Only one of ``reduce_dims`` and ``preserve_dims``
            can be supplied. The default behaviour if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve when calculating the
            PIT distribution, where the mean is taken over all forecast cases.
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
        ValueError if, when ``fcst`` is a dictionary, ``fcst['left']`` and ``fcst['right']`` do not have the same
            shape, dimensions and coordinates.
        ValueError if values in ``fcst`` arrays or sets are less than 0 or greater than 1, ignoriong NaNs.
        ValueError if coordinates in the ``fcst`` ``threshold_dim`` dimension are not increasing.

    Warns:
        - if any values in ``obs`` lie outside the range of values in the forecast ``threshold_dim`` dimension.
        - if any forecast values have NaN
    """
    if isinstance(fcst, dict):
        fcst_left = fcst["left"]
        fcst_right = fcst["right"]
        # check that both have same shape, coords and dims
        if not xr.ones_like(fcst_left).equals(xr.ones_like(fcst_right)):
            raise ValueError("left and right must have same shape, dimensions and coordinates")
        _cdf_checks(fcst_left, threshold_dim)
        _cdf_checks(fcst_right, threshold_dim)
    else:
        _cdf_checks(fcst, threshold_dim)
        fcst_left = fcst.copy()
        fcst_right = fcst.copy()

    _pit_dimension_checks(fcst_left, obs, weights)

    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims

    dims_for_mean = gather_dimensions(
        fcst_left.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=threshold_dim,
    )

    # PIT values in [G(y-), G(y)] format
    pit_values = _pit_values_for_cdf(fcst_left, fcst_right, obs, threshold_dim)

    # convert to CDF format, take weighted means, and output dictionary
    result = _pit_values_final_processing(pit_values, weights, dims_for_mean)

    return result


def _cdf_checks(cdf: XarrayLike, threshold_dim: str):
    """
    For each CDF in `cdf_list`, checks that
        - coords in `threshold_dim` are increasing
        - cdf takes values in the unit interval [0,1]
    Does not check that the CDF is increasing. This is left to the user.

    Args:
        cdf: xarray object
        threshold_dim: name of the threshold dimension in the object

    Raises:
        ValueError if coords in `threshold_dim` not increasing
        ValueError if any values in `cdf` are outside [0, 1] (NaNs are ignored)
    """
    if not coords_increasing(cdf, threshold_dim):
        raise ValueError("coordinates along `fcst[threshold_dim]` are not strictly increasing")
    if isinstance(cdf, xr.DataArray):
        if not cdf_values_within_bounds(cdf):
            raise ValueError("`fcst` values must be between 0 and 1 inclusive.")
    else:
        for var in cdf.data_vars:
            if not cdf_values_within_bounds(cdf[var]):
                raise ValueError("`fcst` values must be between 0 and 1 inclusive.")


def _pit_values_final_processing(pit_values, weights, dims_for_mean):
    """
    Given
    """
    # convert to F(x-), F(x) format
    pit_cdf = _pit_cdfvalues(pit_values)

    pit_cdf_left = apply_weights(pit_cdf["left"], weights=weights).mean(dim=dims_for_mean)
    pit_cdf_right = apply_weights(pit_cdf["right"], weights=weights).mean(dim=dims_for_mean)

    # rescale CDFs so that their max value is 1.
    # This corrects for weights that don't sum to 1.
    cdf_right_max = pit_cdf_right.max("pit_x_value")
    pit_cdf_right = pit_cdf_right / cdf_right_max
    pit_cdf_left = pit_cdf_left / cdf_right_max

    return {"left": pit_cdf_left, "right": pit_cdf_right}


def _pit_values_for_cdf_array(
    fcst_left: xr.DataArray, fcst_right: xr.DataArray, obs: xr.DataArray, threshold_dim: str
) -> xr.DataArray:
    """
    For each forecast case in the form of an CDF F (in xr.DataArray format), the PIT value of F
    for the corresponding observation y is a uniform distribution over the closed
    interval [lower,upper], where
        lower = F(y-)
        upper = F(y)
    and F(y-) denotes the left-hand limit of F at y.

    Returns an array of [lower,upper] values in the dimension 'uniform_endpoint'.

    It is assumed that `fcst_left` and `fcst_right` have the same shape, dims and coords.

    Args:
        fcst_left: array of forecast CDF left-limit values, including dimension `threshold_dim`
        fcst_right: array of forecast CDF left-right values, including dimension `threshold_dim`.
            Assumed to have same shape and dimensions at `fcst_left`
        obs: array of forecast values, excluding `threshold_dim`
        threshold_dim: name of the threshold dimension in `fcst`

    Returns:
        array of PIT values in the form [lower,upper], with dimensions
        'uniform_endpoint', all dimensions in `obs` and all dimensions in `fcst`
        excluding `ens_member_dim`.
    """
    # check whether observations are in the threshold_dim range, if not issue a warning
    max_obs = obs.max()
    min_obs = obs.min()
    max_thld = fcst_left[threshold_dim].max()
    min_thld = fcst_left[threshold_dim].min()

    if max_obs > max_thld:
        warnings.warn(
            "Some observations were greater than the maximum `threshold_dim` value in your `fcst`. "
            "The value of the fcst CDF at these observations will be set to 1.",
            UserWarning,
        )
    if min_obs < min_thld:
        warnings.warn(
            "Some observations were less than the minimum `threshold_dim` value in your `fcst`. "
            "The value of the fcst CDF at these observations will be set to 0.",
            UserWarning,
        )

    # check whether fcst values are NaN, if so issue a warning
    if bool(fcst_left.isnull().any()) or bool(fcst_right.isnull().any()):
        warnings.warn(
            "Some forecast CDF values are NaN. In such cases, the entire forecast CDF will be treated as NaN. "
            "To avoid this, you can fill NaNs using `scores.processing.cdf.fill_cdf`.",
            UserWarning,
        )

    flatten_obs = np.unique(obs.values.flatten())
    flatten_obs = flatten_obs[~np.isnan(flatten_obs)]

    # classify the observations
    obs_in_thlds = [x for x in flatten_obs if x in fcst_right[threshold_dim]]
    obs_gt_thlds = [x for x in flatten_obs if x > max_thld]
    obs_lt_thlds = np.array([x for x in flatten_obs if x < min_thld])
    obs_between_thlds = [
        x for x in flatten_obs if x not in set(obs_in_thlds).union(set(obs_gt_thlds)).union(set(obs_lt_thlds))
    ]

    # extend the fcst CDF values so they can be evaluated at obs
    fcst_left_at_obs = []
    fcst_right_at_obs = []

    # start with the original forecast
    if len(obs_in_thlds) > 0:
        fcst_left_at_obs.append(fcst_left.sel({threshold_dim: obs_in_thlds}))
        fcst_right_at_obs.append(fcst_right.sel({threshold_dim: obs_in_thlds}))

    # calculate the interpolated values using interp1d
    # (xarray interpolate_na not suitable as it cannot handle duplicate x values)
    if len(obs_between_thlds) > 0:
        plotting_points = xr.concat([fcst_left, fcst_right], threshold_dim).sortby(threshold_dim)
        threshold_axis = plotting_points.get_axis_num(threshold_dim)
        y_values = plotting_points.values
        x_values = plotting_points[threshold_dim].values

        interpolated_values = interp1d(x_values, y_values, axis=threshold_axis, kind="linear")(obs_between_thlds)

        # turn the numpy array into an xarray array
        interpolated_coords = dict(fcst_left.coords)
        interpolated_coords[threshold_dim] = obs_between_thlds
        interpolated_values = xr.DataArray(data=interpolated_values, dims=fcst_left.dims, coords=interpolated_coords)

        fcst_left_at_obs.append(interpolated_values)
        fcst_right_at_obs.append(interpolated_values)

    # add the extrapolated values, with values of either 0 or 1
    if len(obs_gt_thlds) > 0:
        high_thld = xr.DataArray(data=obs_gt_thlds, dims=[threshold_dim], coords={threshold_dim: obs_gt_thlds})
        high_thld = xr.broadcast(high_thld, fcst_left)[0].sel({threshold_dim: obs_gt_thlds})
        high_thld = xr.ones_like(high_thld)
        fcst_left_at_obs.append(high_thld)
        fcst_right_at_obs.append(high_thld)

    if len(obs_lt_thlds) > 0:
        low_thld = xr.DataArray(data=obs_lt_thlds, dims=[threshold_dim], coords={threshold_dim: obs_lt_thlds})
        low_thld = xr.broadcast(low_thld, fcst_left)[0].sel({threshold_dim: obs_lt_thlds})
        low_thld = xr.zeros_like(low_thld)
        fcst_left_at_obs.append(low_thld)
        fcst_right_at_obs.append(low_thld)

    # combine the data
    no_nans = (fcst_right.notnull() & fcst_left.notnull()).all(threshold_dim)
    fcst_left_at_obs = xr.concat(fcst_left_at_obs, threshold_dim)
    fcst_right_at_obs = xr.concat(fcst_right_at_obs, threshold_dim)

    pit_cdf_left = (
        fcst_left_at_obs.where(fcst_left_at_obs[threshold_dim] == obs)
        .mean(threshold_dim)
        .assign_coords(uniform_endpoint="lower")
        .expand_dims("uniform_endpoint")
        .where(no_nans)
    )
    pit_cdf_right = (
        fcst_right_at_obs.where(fcst_right_at_obs[threshold_dim] == obs)
        .mean(threshold_dim)
        .assign_coords(uniform_endpoint="upper")
        .expand_dims("uniform_endpoint")
        .where(no_nans)
    )
    result = xr.concat([pit_cdf_left, pit_cdf_right], "uniform_endpoint")

    return result


def _pit_values_for_cdf(
    fcst_left: XarrayLike, fcst_right: XarrayLike, obs: XarrayLike, threshold_dim: str
) -> xr.Dataset:
    """
    Does the same as `_pit_values_for_cdf_array` but where at least one of the xarray
    inputs is a dataset.

    Args:
        fcst_left: xarray object forecast CDF left-limit values, including dimension `threshold_dim`
        fcst_right: xarray object forecast CDF right-limit values, including dimension `threshold_dim`.
            Assumed to have same shape, variabes, coords, etc as fcst_right
        obs: array of forecast values, excluding `threshold_dim`
        threshold_dim: name of the threshold dimension in `fcst`

    Returns:
        array of PIT values in the form [lower,upper], with dimensions
        'uniform_endpoint', all dimensions in `obs` and all dimensions in `fcst`
        excluding `ens_member_dim`.
    """
    if isinstance(fcst_left, xr.DataArray) and isinstance(obs, xr.DataArray):
        return _pit_values_for_cdf_array(fcst_left, fcst_right, obs, threshold_dim)
    elif isinstance(fcst_left, xr.Dataset) and isinstance(obs, xr.DataArray):
        return xr.merge(
            [
                _pit_values_for_cdf_array(fcst_left[var], fcst_right[var], obs, threshold_dim)
                for var in fcst_left.data_vars
            ]
        )
    elif isinstance(fcst_left, xr.DataArray):
        return xr.merge(
            [
                _pit_values_for_cdf_array(fcst_left, fcst_right, obs[var], threshold_dim).rename(var)
                for var in obs.data_vars
            ]
        )
    else:
        return xr.merge(
            [
                _pit_values_for_cdf_array(fcst_left[var], fcst_right[var], obs[var], threshold_dim)
                for var in obs.data_vars
            ]
        )


def _get_pit_x_values(pit_values: XarrayLike) -> xr.DataArray:
    """
    Returns a data array of consisting of exactly those x-axis values needed for
    constructing a unifornm PIT probability plot for the given array of `pit_values`.

    Args:
        pit_values: output from `_pit_values_for_ens` or `_pit_values_for_cdf`, containing
            dimension 'uniform_endpoint' with coordinates ['lower', 'upper']

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


def _pit_distribution_for_jumps(pit_values: XarrayLike, x_values: xr.DataArray) -> dict:
    """
    Gives the values F(x) where F is the CDF for a pit value,
    and x comes from `x_values`, given that the CDF jumps at x. This occurs precisely
    when upper == lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    It is assumed that `x_values` contains all the values in `pit_values`.

    Args:
        pit_values: output from `_pit_values_for_ens` or `_pit_values_for_cdf`, containing
            dimension 'uniform_endpoint' with coordinates ['lower', 'upper']
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


def _pit_distribution_for_unif(pit_values: XarrayLike, x_values: xr.DataArray) -> XarrayLike:
    """
    Gives the values F(x) where F is the CDF for the PIT of a particular forcast--observation case,
    and x comes from `x_values`, given that the CDF is uniform. This occurs precisely
    when upper > lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    Left-hand and right-hand limits of F(x) are equal in this case.

    It is assumed that `x_values` containes all the values in `pit_values`.

    Args:
        pit_values: output from `_pit_values_for_ens` or `_pit_values_for_cdf`, containing
            dimension 'uniform_endpoint' with coordinates ['lower', 'upper']
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
        pit_values: output from `_pit_values_for_ens` or `_pit_values_for_cdf`, containing
            dimension 'uniform_endpoint' with coordinates ['lower', 'upper']

    Returns:
        dictionary of cdf values, with keys 'left' and 'right',
        representing the left and right hand limits of the cdf values F(x).
        Each value in the dictionary is an xarray object representing limits of F(x),
        where x is represented by values in the 'pit_x_value' dimension.
    """
    # get the x values
    x_values = _get_pit_x_values(pit_values)

    # get cdf values where the pit cdf jumps
    cdf_at_jump_cases = _pit_distribution_for_jumps(pit_values, x_values)
    # get the cdf values where the pit is uniform
    cdf_at_unif_cases = _pit_distribution_for_unif(pit_values, x_values)
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


def pit_distribution_for_ens(
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
    pit_values = _pit_values_for_ens(fcst, obs, ens_member_dim)
    # convert to CDF format, take weighted means, and output dictionary
    result = _pit_values_final_processing(pit_values, weights, dims_for_mean)

    return result


def _get_plotting_points_dict(left: XarrayLike, right: XarrayLike) -> dict:
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


def _construct_hist_values(cdf_at_endpoints: list[XarrayLike], bins: int) -> XarrayLike:
    """
    Calculates the PIT histogram values given values of the PIT CDF at the endpoint of
    every histogram bin.

    Args:
        cdf_at_endpoints: a list of xarray objects with dimension "pit_x_value",
            each giving the value of the PIT CDF at an endpoint of one of the histogram bins.
        bins: number of bins in the histogram.

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
            "pit_x_value": histogram_values["pit_x_value"] - 1 / (2 * bins),
            "bin_left_endpoint": histogram_values["pit_x_value"] - 1 / bins,
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
    left_endpoints = np.arange(bins) / bins

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

    histogram_values = _construct_hist_values(cdf_at_endpoints, bins)
    return histogram_values


def _pit_hist_right(pit_left: XarrayLike, pit_right: XarrayLike, bins: int) -> XarrayLike:
    """
    Gets the pit histogram values, with right endpoints included in every bin.
    e.g. if bins=5, the bins are [0, 0.2], (0.2, 0.4], ..., (0.8, 1]

    Args:
        pit_left: `left` attribute of a `Pit_for_ensemble` object
        pit_right: `right` attribute of a `Pit_for_ensemble` object
        bins: the number of bins in the histogram

    Returns:
        xarray object including dim 'bin_centre' and coordinates 'bin_left_endpoint',
        'bin_right_endpoint', with values the hight of each bar in the histogram.
    """
    right_endpoints = np.arange(1, bins + 1) / bins

    # the value of the mean PIT CDF at 0 is 0, except where NaNs
    # using .sel is safe because the value 0.0 is guaranteed to be in pit_x_value (see _get_pit_x_values)
    value_at_0 = pit_left.sel(pit_x_value=slice(0.0, 0.0))

    cdf_at_endpoints = [value_at_0]

    # get values of the PIT CDF that are aleady available in pit
    common_points = [x for x in right_endpoints if x in pit_right["pit_x_value"]]
    if len(common_points) > 0:
        cdf_at_common_points = pit_right.sel(pit_x_value=common_points)
        cdf_at_endpoints.append(cdf_at_common_points)

    # get values of the PIT CDF that are not available in pit
    new_points = [x for x in right_endpoints if x not in pit_right["pit_x_value"]]
    for new_point in new_points:
        value = _value_at_pit_cdf(pit_left, pit_right, new_point)
        cdf_at_endpoints.append(value)

    histogram_values = _construct_hist_values(cdf_at_endpoints, bins)

    return histogram_values


def _alpha_score(plotting_points: XarrayLike) -> XarrayLike:
    """
    Given output of plotting ploints from `Pit_for_ensemble.plotting_points`,
    calculates the alpha score.

    Args:
        plotting_points: xarray object of plotting points, indexed by non-decreasing values
            (with duplicates) along the 'pit_x_value' dimension.

    Returns:
        xarray object with alpha score, and 'pit_x_value' dimension collapsed
    """
    return np.abs(plotting_points - plotting_points["pit_x_value"]).integrate("pit_x_value")


def _expected_value(plotting_points: XarrayLike) -> XarrayLike:
    """
    Calculates the expected value of the PIT distribution, given plotting points for the
    CDF of that distribution. Uses the well-known formula that the expected value of a
    non-negative random variable is the integral of 1 - CDF.

    Here, `plotting_points`, with linear interpolation, fully describes the CDF of the
    PIT distribution.

    Args:
        plotting_points: xarray object of plotting points, indexed by non-decreasing values
            (with duplicates) along the 'pit_x_value' dimension.

    Returns:
        xarray object with expected value, and 'pit_x_value' dimension collapsed
    """
    return (1 - plotting_points).integrate("pit_x_value")


def _variance(plotting_points: XarrayLike) -> XarrayLike:
    """
    Calculates the variance of the PIT distribution using the formula
        Var(Y) = 2 * integral(x * (1 - F(x)), x >= 0) + E(Y)^2,
    for a non-negative random variable Y, where F is the CDF of Y.

    Here, `plotting_points`, with linear interpolation, fully describes the CDF of the
    PIT distribution.

    Args:
        plotting_points: xarray object of plotting points, indexed by non-decreasing values
            (with duplicates) along the 'pit_x_value' dimension.

    Returns:
        xarray object with the variance, and 'pit_x_value' dimension collapsed
    """
    expected_value = _expected_value(plotting_points)
    integral_term = 2 * _variance_integral_term(plotting_points)
    return integral_term - expected_value**2


def _variance_integral_term(plotting_points: XarrayLike) -> XarrayLike:
    """
    Calculates
        integral(x * (1 - F(x)), x >= 0)
    where F is a CDF.

    Here, `plotting_points`, with linear interpolation, fully describes the CDF of the
    PIT distribution F. It is assumed coordinates in `plotting_points['pit_x_value']`
    come in duplicate pairs (e.g. [0, 0, 0.1, 0.1, 0.15, 0.15, ...]), to describe
    left-hand and right-hand limits at each point. This assumption holds with output
    from the `.plotting_points` method.

    Args:
        plotting_points: xarray object of plotting points, indexed by non-decreasing values
            (with duplicates) along the 'pit_x_value' dimension.

    Returns:
        xarray object with the integral values, and 'pit_x_value' dimension collapsed
    """
    # notation: F_i(t) = m_i * t + b_i whenever x[i-1] <= t <= x[i]
    # and x[i] is a value in `function_values[threshold_dim]`.

    x_values = plotting_points["pit_x_value"]
    x_shifted = plotting_points["pit_x_value"].shift(pit_x_value=1)
    # difference in x values
    diff_xs = x_values - x_shifted
    # difference in function values y_i = F(x[i])
    diff_ys = plotting_points - plotting_points.shift(pit_x_value=1)
    # gradients m
    m_values = diff_ys / diff_xs
    # intercepts b_i
    b_values = plotting_points - m_values * x_values
    # integral(t * (1 - F(t))) on the interval (x[i-1], x[i]), for each i, using calculus:
    integral_i = (1 - b_values) * (x_values**2 - x_shifted**2) / 2 - m_values * (x_values**3 - x_shifted**3) / 3

    integral = integral_i.sum("pit_x_value")
    # return NaN if NaN in function_values
    integral = integral.where(~np.isnan(plotting_points).any("pit_x_value"))

    return integral
