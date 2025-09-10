"""
Methods for the probability integral transform (PIT) classes Pit and Pit_fcst_at_obs.

The basic approach follows Taggart (2023) and Gneiting and Ranjan (2013); see
http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf

For a forecast--observation pair (F,y), where F is a CDF,
the corresponding PIT value is the uniform distribution on the closed interval
[F(y-), F(y)], where F(y-) denotes the left-hand limit.
So if F(y-)=F(y), the PIT value is the distribution concentrated at the point F(y).
PIT values for particular forecast cases can be represented in the form
    lower = F(y-), upper = F(y)
or can be described by the CDF of the uniform distribution on the interval [F(y-), F(y)].

The code here first converts inputs to the [lower, upper] representation.
Then it converts this representation to the uniform CDF representation, with CDF values
stored at a minimal number of points (the remaining values obtained exactly when required
via linear interpolation).

Once each forecast case has a PIT value in CDF representation, the (possibly weighted)
mean PIT value across all forecast cases is the weighted mean across all the CDF representations.
We call this the PIT distribution (or PIT CDF) for all the forecast cases.

The PIT distribution is always piecewise linear and right-continuous. It can be represented
using "left" and "right" values, respectively representing the left-hand limit of the PIT CDF
and the value (which equals the right-hand limit) of the CDF. All standard statistics of
PIT for the set of forecast cases, such as PIT histogram bar hieghts and alpha scores,
can be calculated exactly from this representation.

The code here is structured as follows:
1. The two classes `Pit` and `Pit_fcst_at_obs` are introduced
2. Private functions that calculate the PIT distribution are then given
3. Private functions that calculate the PIT statistics from the PIT distribution are then presented.

The tutorial is a good place to start to understand the big picture.
"""

import warnings
from typing import Hashable, Optional

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from scores.functions import apply_weights
from scores.probability.checks import cdf_values_within_bounds, coords_increasing
from scores.processing import broadcast_and_match_nan
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

################################################
# The two classes: `Pit`` and `Pit_fcst_at_obs`
################################################


class Pit:
    """
    Calculates the probability intergral transform (PIT) for a set of forecast cases and
    corresponding observations. The calculated PIT can be a (possibly weighted) average
    over specified dimensions. Calculations are performed and interpreted as follows.

    Given a predictive cumulative distribution function (CDF) :math:`G` and corresponding
    observation :math:`y`, the corresponding PIT value is a uniform distribution on the
    closed interval :math:`[G(y-), G(y)]`, where :math:`G(y-)` denotes the left-hand
    limit of :math:`G` at :math:`y`. This is the most general form of PIT and handles cases
    where the observation :math:`y` coincides with a point of discontinuity of a predictive CDF
    :math:`G`. See Gneiting and Ranjan (2013) and Taggart (2022).

    Weighted means of PIT values are simply weighted means of CDFs of those uniform distributions
    on the closed interval :math:`[G(y-), G(y)]`. The weighted mean is itself a CDF. All
    statistics related to the collection of PIT values, such as bar heights of PIT histograms, can be
    calculated from this CDF.

    Forecasts used by the ``Pit`` class can be given in two forms:

    - An ensemble for forecasts, indexed by an ensemble member dimension. In this case,
        the ensemble is interpreted as empirical cumulative distribution function (CDF)
        of the ensemble members.
    - Values of the predictive CDF, indexed by a threshold dimension. Left-hand limits
        of the predictive CDF can also be specified, allowing for CDFs with discontinuities.
        Values of the CDF between coordinates along the threshold dimension are determined via
        linear interpolation, whilst values of the CDF at points outside those coordinates
        are assigned either 0 or 1 as appropriate. Any predictive CDF with a NaN value
        will be treated as NaN in its entirety.

    Consider using the ``Pit_fcst_at_obs`` class instead of ``Pit`` when values of the predictive CDFs
    evaulated at the observations is easy to generate, e.g. when the predictive CDFs are normal
     distributions with known parameters.

    Attributes:
        left: values for the left-hand limit of the PIT distribution (represented as a CDF)
        right: values for the PIT distribution (represented as a CDF). Since CDFs are right-continuous,
            these values also equal values of the right-hand limits.

    Methods:
            plotting_points: generates plotting points for PIT-uniform probability plots.
                Indexer along the horizontal dimension will contain duplicate values.
            plotting_points_parametric: generates plotting points for PIT-uniform probability plots
                in parametric format. Indexer along the parametrization will not contain duplicate values.
            hist_values: generates values for the PIT histogram.
            alpha_score: calculates the 'alpha score', which is the absolute area between
                the diagonal and PIT graph of the PIT-uniform probability plot.
            expected_value: calculates the expected value of the PIT CDF.
            variance: calculates the variance of the PIT CDF.

    References:
        - Gneiting, T., & Ranjan, R. (2013). Combining predictive distributions. Electron. J. Statist. 7: 1747-1782 \
            https://doi.org/10.1214/13-EJS823
        - Taggart, R. J. (2022). Assessing calibration when predictive distributions have discontinuities. \
            Bureau Research Report 64, http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf

    See also:
            - :py:func:`scores.probability.Pit_fcst_at_obs`
            - :py:func:`scores.probability.rank_histogram`
    """

    def __init__(
        self,
        fcst: XarrayLike,
        obs: XarrayLike,
        special_fcst_dim: str,
        *,  # Force keywords arguments to be keyword-only
        fcst_type: str = "ensemble",
        fcst_left: Optional[XarrayLike] = None,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
        weights: Optional[XarrayLike] = None,
    ):
        """
        Calculates the mean PIT :math:`F`, interpreted as a cumulative distribution function (CDF),
        given the set of forecast and observations pairs.

        The CDF :math:`F` is completely determined by its values :math:`F(x)` and left-hand limits
        :math:`F(x-)` at a minimal set of points :math:`x`. All other values may be obtained via
        interpolation whenever :math:`0 < x < 1` or the fact that :math:`F(x) = 0` when
        :math:`x < 1` and :math:`F(x) = 1` when :math:`x > 1`.

        The values of :math:`F` at this minimal set of points is accessible via the
        ``left`` and ``right`` attributes. These values are sufficient to calculate precise
        statistics for the PIT for the set of forecasts and observations, as provided by
        ``Pit`` methods.

        Args:
            fcst: an xarray object of forecasts, containing the dimension `special_fcst_dim`.
                The values of ``fcst`` are the values of the ensemble if ``fcst_type='ensemble'``,
                or the values of the predictive CDF if ``fcst_type='cdf'``
            obs: an xarray object of observations.
            special_fcst_dim: name of the ensemble member dimension in ``fcst`` if ``fcst_type='ensemble'``
                or of the CDF threshold dimension in ``fcst`` if ``fcst_type='cdf'``.
            fcst_type: either "ensemble" or "cdf".
            fcst_left: The values of the left-hand limits of the predictive CDF. Must have the same
                shape and dimensions as ``fcst``.
                Only required when ``fcst_type='cdf'`` and the predictive CDF is discontinuous.
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
            left: xarray object representing the PIT across all forecast cases, interpreted
                as a CDF :math:`F` and evaluated as left-hand limits at the points :math:`x`
                in the dimension "pit_x_value". That is, values in the array or dataset
                are of the form :math:`F(x-)`.
            right: xarray object representing the PIT across all forecast cases, interpreted
                as a CDF :math:`F` and evaluated as at the points :math:`x` in the dimension
                "pit_x_value". That is, values in the array or dataset are of the form :math:`F(x)`.

        Methods:
            plotting_points: generates plotting points for PIT-uniform probability plots.
                Indexer along the horizontal dimension will contain duplicate values.
            plotting_points_parametric: generates plotting points for PIT-uniform probability plots
                in parametric format. Indexer along the parametrization will not contain duplicate values.
            hist_values: generates values for the PIT histogram.
            alpha_score: calculates the 'alpha score', which is the absolute area between
                the diagonal and PIT graph of the PIT-uniform probability plot.
            expected_value: calculates the expected value of the PIT CDF.
            variance: calculates the variance of the PIT CDF.

        Raises:
            - ValueError if ``fcst_type`` is not one of "ensemble" or "cdf".
            - ValueError if dimensions of ``fcst``, ``obs`` or ``weights`` contain any of the following reserved names:
                "uniform_endpoint", "pit_x_value", "x_plotting_position", "y_plotting_position", "plotting_point",
                "bin_left_endpoint", "bin_right_endpoint", "bin_centre".
            - ValueError if, when ``fcst_left`` is not ``None``, ``fcst_left`` and ``fcst`` do not have identical
                shape, dimensions and coordinates.
            - ValueError if, when ``fcst_type='cdf'``, any values of ``fcst`` are less then 0 or greater than 1.
            - ValueError if, when ``fcst_type='cdf'`` and ``fcst_left`` is not ``None``,
                any values of ``fcst_left`` are less then 0 or greater than 1.
            - ValueError if, when ``fcst_type='cdf'`` and ``fcst_left`` is not ``None``,
                any values of ``fcst_left`` are greater than ``fcst``.
            - ValueError if, when ``fcst_type='cdf'`` and ``fcst[special_fcst_dim]`` is not increasing.

        Warns:
            - UserWarning if, when ``fcst_type='cdf'``, any values of ``obs`` are less than
                the minimum of ``fcst[special_fcst_dim]`` or greater than the maximum of
                ``fcst[special_fcst_dim]``.
            - UserWarning if, when ``fcst_type='cdf'``, there are any NaN values in ``fcst`` or
                ``fcst_left``.

        See also:
            - :py:func:`scores.probability.Pit_fcst_at_obs`
            - :py:func:`scores.probability.rank_histogram`
        """
        if fcst_type not in ["ensemble", "cdf"]:
            raise ValueError('`fcst_type` must be one of "ensemble" or "cdf"')

        if fcst_type == "ensemble":
            pit_cdf = _pit_distribution_for_ens(
                fcst,
                obs,
                special_fcst_dim,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
            )
        else:
            pit_cdf = _pit_distribution_for_cdf(
                fcst,
                obs,
                special_fcst_dim,
                fcst_left=fcst_left,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
                weights=weights,
            )
        self.left = pit_cdf["left"]
        self.right = pit_cdf["right"]

    def plotting_points(self) -> XarrayLike:
        """
        Returns an xarray object with the plotting points for PIT-uniform probability plots,
        or equivalently, the plotting points for the PIT CDF. The x (horizontal) plotting
        positions are values from the "pit_x_value" index, while the
        y (vertical) plotting positions are values in the output xarray object.

        Note that coordinates in the "pit_x_value" dimension will have duplicate values.
        For a parametric approach to plotting points without duplicate coordinate values,
        see the ``plotting_points_parametric`` method.
        """
        return _get_plotting_points(self.left, self.right)

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
        return _get_plotting_points_param(self.left, self.right)

    def hist_values(self, bins: int, right: bool = True) -> XarrayLike:
        """
        Returns an xarray object with the PIT histogram values across appropriate dimensions,
        with additional coordinates "bin_left_endpoint", "bin_right_endpoint" and "bin_centre".

        Args:
            bins: the number of bins in the histogram.
            right: If True, histogram bins always include the rightmost edge. If False,
                bins always include the leftmost edge.
        """
        # calculting _pit_hist_right or _pit_hist_left does not work with chunks,
        # because dask wants to chunk over the 'pit_x_value' dimension, but
        # _pit_hist_right and _pit_hist_left merely wants to sample a small number of points
        # from the 'pit_x_value'. So compute chunks at this stage.
        if self.left.chunks is not None:
            self.left = self.left.compute()
            self.right = self.right.compute()

        if right:
            return _pit_hist_right(self.left, self.right, bins)
        return _pit_hist_left(self.left, self.right, bins)

    def alpha_score(self) -> XarrayLike:
        """
        Calculates the so-called 'alpha score', which is a measure of how close the PIT distribution
        is to the uniform distribution. If the PIT CDF is :math:`F`, then the alpha score :math:`S`
        is given by
            :math:`\\int_0^1 |F(x) - x|\\,\\text{d}x}`

        References:
            - Renard, B., Kavetski, D., Kuczera, G., Thyer, M., & Franks, S. W. (2010). \
                Understanding predictive uncertainty in hydrologic modeling: \
                The challenge of identifying input and structural errors. \
                Water Resources Research, 46(5).
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


class Pit_fcst_at_obs:
    """
    Calculates the probability intergral transform (PIT) given for a set of forecast cases
    evaluated at the corresponding observations. The calculated PIT can be a (possibly weighted) average
    over specified dimensions. Calculations are performed and interpreted as follows.

    Given a predictive cumulative distribution function (CDF) :math:`G` and corresponding
    observation :math:`y`, the corresponding PIT value is a uniform distribution on the
    closed interval :math:`[G(y-), G(y)]`, where :math:`G(y-)` denotes the left-hand
    limit of :math:`G` at :math:`y`. This is the most general form of PIT and handles cases
    where the observation :math:`y` coincides with a point of discontinuity of a predictive CDF
    :math:`G`. See Gneiting and Ranjan (2013) and Taggart (2022).

    In the ``Pit_fcst_at_obs`` implementation of PIT, the user supplies the values :math:`G(y)` and
    (optionally) :math:`G(y-)`. If the predictive distributions are in the form of an ensemble
    or of CDFs whose values which are harder to evaluate at the observations, consider using the
    ``Pit`` implementation instead.

    Weighted means of PIT values are simply weighted means of CDFs of those uniform distributions
    on the closed interval :math:`[G(y-), G(y)]`. The weighted mean is itself a CDF. All
    statistics related to the collection of PIT values, such as bar heights of PIT histograms, can be
    calculated from this CDF.

    Attributes:
        left: values for the left-hand limit of the PIT distribution (represented as a CDF)
        right: values for the PIT distribution (represented as a CDF). Since CDFs are right-continuous,
            these values also equal values of the right-hand limits.

    Methods:
            plotting_points: generates plotting points for PIT-uniform probability plots.
                Indexer along the horizontal dimension will contain duplicate values.
            plotting_points_parametric: generates plotting points for PIT-uniform probability plots
                in parametric format. Indexer along the parametrization will not contain duplicate values.
            hist_values: generates values for the PIT histogram.
            alpha_score: calculates the 'alpha score', which is the absolute area between
                the diagonal and PIT graph of the PIT-uniform probability plot.
            expected_value: calculates the expected value of the PIT CDF.
            variance: calculates the variance of the PIT CDF.

    References:
        - Gneiting, T., & Ranjan, R. (2013). Combining predictive distributions. Electron. J. Statist. 7: 1747-1782 \
            https://doi.org/10.1214/13-EJS823
        - Taggart, R. J. (2022). Assessing calibration when predictive distributions have discontinuities. \
            Bureau Research Report 64, http://www.bom.gov.au/research/publications/researchreports/BRR-064.pdf

    See also:
            - :py:func:`scores.probability.Pit`
            - :py:func:`scores.probability.rank_histogram`
    """

    def __init__(
        self,
        fcst_at_obs: XarrayLike,
        *,  # Force keywords arguments to be keyword-only
        fcst_at_obs_left: Optional[XarrayLike] = None,
        reduce_dims: Optional[FlexibleDimensionTypes] = None,
        preserve_dims: Optional[FlexibleDimensionTypes] = None,
        weights: Optional[XarrayLike] = None,
    ):
        """
        Calculates the mean PIT :math:`F`, interpreted as a cumulative distribution function (CDF),
        given the set of forecast CDFs evaluated at the corresponding observations.

        The CDF :math:`F` is completely determined by its values :math:`F(x)` and left-hand limits
        :math:`F(x-)` at a minimal set of points :math:`x`. All other values may be obtained via
        interpolation whenever :math:`0 < x < 1` or the fact that :math:`F(x) = 0` when
        :math:`x < 1` and :math:`F(x) = 1` when :math:`x > 1`.

        The values of :math:`F` at this minimal set of points is accessible via the
        ``left`` and ``right`` attributes. These values are sufficient to calculate precise
        statistics for the PIT for the set of forecasts and observations, as provided by
        ``Pit_fcst_at_obs`` methods.

        Args:
            fcst_at_obs: an xarray object of values of the forecast CDF evaluated at each
                corresponding observation.
            fcst_at_obs_left: an xarray object of left-hand limits of the forecast CDF at each
                corresponding observation. Only needs to be supplied if there are any cases
                where the left-hand limit does not equal the value at the forecast CDF.
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
            left: xarray object representing the PIT across all forecast cases, interpreted
                as a CDF :math:`F` and evaluated as left-hand limits at the points :math:`x`
                in the dimension "pit_x_value". That is, values in the array or dataset
                are of the form :math:`F(x-)`.
            right: xarray object representing the PIT across all forecast cases, interpreted
                as a CDF :math:`F` and evaluated as at the points :math:`x` in the dimension
                "pit_x_value". That is, values in the array or dataset are of the form :math:`F(x)`.

        Methods:
            plotting_points: generates plotting points for PIT-uniform probability plots.
                Indexer along the horizontal dimension will contain duplicate values.
            plotting_points_parametric: generates plotting points for PIT-uniform probability plots
                in parametric format. Indexer along the parametrization will not contain duplicate values.
            hist_values: generates values for the PIT histogram.
            alpha_score: calculates the 'alpha score', which is the absolute area between
                the diagonal and PIT graph of the PIT-uniform probability plot.
            expected_value: calculates the expected value of the PIT CDF.
            variance: calculates the variance of the PIT CDF.

        Raises:
            - ValueError if dimensions of ``fcst``, ``obs`` or ``weights`` contain any of the following reserved names:
                "uniform_endpoint", "pit_x_value", "x_plotting_position", "y_plotting_position", "plotting_point",
                "bin_left_endpoint", "bin_right_endpoint", "bin_centre",
            - ValueError if, when ``fcst_at_obs_left`` is not ``None``, ``fcst_at_obs`` and ``fcst_at_obs_left``
                do not have identical shape, dimensions and coordinates.
            - ValueError if any values of ``fcst`` are less then 0 or greater than 1.
            - ValueError if, when ``fcst_at_obs_left`` is not ``None``,
                any values of ``fcst_at_obs_left`` are less then 0 or greater than 1.
            - ValueError if, when ``fcst_at_obs_left`` is not ``None``,
                any values of ``fcst_at_obs_left`` are greater than ``fcst``.

        See also:
            - :py:func:`scores.probability.Pit_fcst_at_obs`
            - :py:func:`scores.probability.rank_histogram`lah
        """
        pit_cdf = _pit_values_for_fcst_at_obs(fcst_at_obs, fcst_at_obs_left, reduce_dims, preserve_dims, weights)
        self.left = pit_cdf["left"]
        self.right = pit_cdf["right"]

    def plotting_points(self) -> XarrayLike:
        """
        Returns an xarray object with the plotting points for PIT-uniform probability plots,
        or equivalently, the plotting points for the PIT CDF. The x (horizontal) plotting
        positions are values from the "pit_x_value" index, while the
        y (vertical) plotting positions are values in the output xarray object.

        Note that coordinates in the "pit_x_value" dimension will have duplicate values.
        For a parametric approach to plotting points without duplicate coordinate values, see the
        ``plotting_points_parametric`` method.
        """
        return _get_plotting_points(self.left, self.right)

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
        return _get_plotting_points_param(self.left, self.right)

    def hist_values(self, bins: int, right: bool = True) -> XarrayLike:
        """
        Returns an xarray object with the PIT histogram values across appropriate dimensions,
        with additional coordinates "bin_left_endpoint", "bin_right_endpoint" and "bin_centre".

        Args:
            bins: the number of bins in the histogram.
            right: If True, histogram bins always include the rightmost edge. If False,
                bins always include the leftmost edge.
        """
        # calculting _pit_hist_right or _pit_hist_left does not work with chunks,
        # because dask wants to chunk over the 'pit_x_value' dimension, but
        # _pit_hist_right and _pit_hist_left merely wants to sample a small number of points
        # from the 'pit_x_value'. So compute chunks at this stage.
        if self.left.chunks is not None:
            self.left = self.left.compute()
            self.right = self.right.compute()

        if right:
            return _pit_hist_right(self.left, self.right, bins)
        return _pit_hist_left(self.left, self.right, bins)

    def alpha_score(self) -> XarrayLike:
        """
        Calculates the so-called 'alpha score', which is a measure of how close the PIT distribution
        is to the uniform distribution. If the PIT CDF is :math:`F`, then the alpha score :math:`S`
        is given by
            :math:`\\int_0^1 |F(x) - x|\\,\\text{d}x}`

        References:
            - Renard, B., Kavetski, D., Kuczera, G., Thyer, M., & Franks, S. W. (2010). \
                Understanding predictive uncertainty in hydrologic modeling: \
                The challenge of identifying input and structural errors. \
                Water Resources Research, 46(5).
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


#####################################################
# Functions for calculating __init__ of both classes
#####################################################


def _dims_for_mean_with_checks(
    fcst: XarrayLike,
    obs: XarrayLike,
    special_fcst_dim: Optional[str],
    weights: Optional[XarrayLike],
    reduce_dims: Optional[FlexibleDimensionTypes],
    preserve_dims: Optional[FlexibleDimensionTypes],
) -> set[Hashable]:
    """
    Given inputs for Pit or Pit_fcst_at_obs, checks that XarrayLike inputs don't use
    RESERVES_NAMES and then gathers dimensions, returning the set of dimensions for
    calculating the mean.

    When applying to `Pit_fcst_at_obs` inputs, use `fcst=obs=fcst_at_obs`.
    """
    all_dims = set(fcst.dims).union(obs.dims)

    weights_dims = None
    if weights is not None:
        weights_dims = weights.dims
        all_dims = all_dims.union(weights_dims)

    if len([dim for dim in RESERVED_NAMES if dim in all_dims]) > 0:
        raise ValueError(
            f'The following names are reserved and should not be among the dimensions of \
            xarray inputs: {", ".join(RESERVED_NAMES)}'
        )

    dims_for_mean = gather_dimensions(
        fcst.dims,
        obs.dims,
        weights_dims=weights_dims,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        score_specific_fcst_dims=special_fcst_dim,
    )
    return dims_for_mean


def _right_left_checks(
    right: XarrayLike, left: Optional[XarrayLike], threshold_dim: Optional[str], right_arg_name: str, left_arg_name: str
):
    """
    Raises:
        - if `right` or `left` have values less than 0 or greater than 1
        - if `right[threshold_dim]` is not strictly increasing
        - if, when `left` is not None, `right` and `left` don't have the same shape, dims or coords
        - if, when `left` is not None, `right < left` at some point
    """
    # first check right
    if isinstance(right, xr.DataArray):
        if not cdf_values_within_bounds(right):
            raise ValueError(f"`{right_arg_name}` values must be between 0 and 1 inclusive.")
    else:
        for var in right.data_vars:
            if not cdf_values_within_bounds(right[var]):
                raise ValueError(f"`{right_arg_name}` values must be between 0 and 1 inclusive.")
    if (threshold_dim is not None) and (not coords_increasing(right, threshold_dim)):
        raise ValueError(f"coordinates along `{right_arg_name}[threshold_dim]` are not strictly increasing")

    # then check left if appropriate
    if left is not None:
        # check that both fcst and fcst_left have same shape, coords and dims
        if not xr.ones_like(right).equals(xr.ones_like(left)):
            raise ValueError(
                f"If `{left_arg_name}` is not `None`, `{right_arg_name}` and `{left_arg_name}` "
                "must have same shape, dimensions and coordinates"
            )
        if isinstance(right, xr.DataArray):
            if not cdf_values_within_bounds(left):
                raise ValueError(f"`{left_arg_name}` values must be between 0 and 1 inclusive.")
            if (left > right).any():
                raise ValueError(f"`{left_arg_name}` must not exceed `{right_arg_name}`")
        else:
            for var in left.data_vars:
                if not cdf_values_within_bounds(left[var]):
                    raise ValueError(f"`{left_arg_name}` values must be between 0 and 1 inclusive.")
                if (left[var] > right[var]).any():
                    raise ValueError(f"`{left_arg_name}` must not exceed `{right_arg_name}`")


def _pit_values_for_fcst_at_obs(
    fcst_at_obs: XarrayLike,
    fcst_at_obs_left: Optional[XarrayLike],
    reduce_dims: Optional[FlexibleDimensionTypes],
    preserve_dims: Optional[FlexibleDimensionTypes],
    weights: Optional[XarrayLike],
) -> dict:
    """
    A private function to compute `Pit_fcst_at_obs.__init__`.
    Returns `Pit_fcst_at_obs().left` and `Pit_fcst_at_obs().right` in the form of a dictionary
    with keys 'left' and 'right'.

    See docstring `Pit_fcst_at_obs.__init__` for details.
    """
    _right_left_checks(fcst_at_obs, fcst_at_obs_left, None, "fcst_at_obs", "fcst_at_obs_left")

    dims_for_mean = _dims_for_mean_with_checks(fcst_at_obs, fcst_at_obs, None, weights, reduce_dims, preserve_dims)

    if fcst_at_obs_left is None:
        fcst_at_obs_left = fcst_at_obs.copy()
    else:
        fcst_at_obs, fcst_at_obs_left = broadcast_and_match_nan(fcst_at_obs, fcst_at_obs_left)

    pit_values = xr.concat(
        [
            fcst_at_obs_left.assign_coords(uniform_endpoint="lower").expand_dims("uniform_endpoint"),
            fcst_at_obs.assign_coords(uniform_endpoint="upper").expand_dims("uniform_endpoint"),
        ],
        "uniform_endpoint",
    )

    # convert to CDF format, take weighted means, and output dictionary
    result = _pit_values_final_processing(pit_values, weights, dims_for_mean)

    return result


def _pit_values_final_processing(
    pit_values: XarrayLike, weights: Optional[XarrayLike], dims_for_mean: set[Hashable]
) -> dict:
    """
    Given PIT values in the format [lower,upper], representing the uniform distribution
    on the closed interval [lower,upper], converts to CDF format, with output 'left' giving
    the left limit of the CDF and 'right' giving the value at the CDF. Weighted means are then
    taken across all the CDFs. The output is scaled so that it is in CDF format (i.e., max value is 1)

    Args:
        pit_values: values of the PIT in [upper, lower] format, including dimension 'uniform_endpoint'
            that has two coordinates "upper" and "lower"
        weights: optional array of weights when calculating the weighted mean
        dims_for_mean: list, set etc of dimensions over which to apply the mean.

    Returns:
        dictionary of two xarray objects, with keys 'left' and 'right', containing values of the
            left-hand and right-hand limits of the mean weighted PIT in CDF format.
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


def _pit_values_for_ens(fcst: XarrayLike, obs: XarrayLike, ens_member_dim: str) -> XarrayLike:
    """
    For each forecast case in the form of an ensemble, the PIT value of the ensemble for
    the corresponding observation is a uniform distribution over the closed interval
    [lower,upper], where
        lower = (count of ensemble members strictly less than the observation) / n
        upper = (count of ensemble members not exceeding the observation) / n
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


def _pit_distribution_for_cdf(
    fcst: XarrayLike,
    obs: XarrayLike,
    threshold_dim: str,
    *,  # Force keywords arguments to be keyword-only
    fcst_left: Optional[XarrayLike] = None,
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
        fcst: xarray object of CDF forecast values, containing the dimension `threshold_dim`,
            or a dictionary containing the keys "left" and "right", with corresponding values
            xarray objects giving the left-hand and right-hand limits of the fcst CDF
            along the dimension `threshold_dim`.
        obs: an xarray object of observations.
        threshold_dim: name of the threshold dimension in ``fcst``, such that the probability
            of not exceeding a particular threshold is one of the corresponding values of ``fcst``.
        fcst_left: xarray object of forecast CDF left-handed limit values. If None, it is
            assumed that the forecast CDF is continuous.
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
    _right_left_checks(fcst, fcst_left, threshold_dim, "fcst", "fcst_left")

    dims_for_mean = _dims_for_mean_with_checks(fcst, obs, threshold_dim, weights, reduce_dims, preserve_dims)

    if fcst_left is None:
        fcst_left = fcst.copy()

    # PIT values in [G(y-), G(y)] format
    pit_values = _pit_values_for_cdf(fcst_left, fcst, obs, threshold_dim)

    # convert to CDF format, take weighted means, and output dictionary
    result = _pit_values_final_processing(pit_values, weights, dims_for_mean)

    return result


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
                _pit_values_for_cdf_array(fcst_left[var], fcst_right[var], obs, threshold_dim).rename(var)
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
                _pit_values_for_cdf_array(fcst_left[var], fcst_right[var], obs[var], threshold_dim).rename(var)
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


def _pit_distribution_for_ens(
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

    dims_for_mean = _dims_for_mean_with_checks(fcst, obs, ens_member_dim, weights, reduce_dims, preserve_dims)

    # PIT values in [G(y-), G(y)] format
    pit_values = _pit_values_for_ens(fcst, obs, ens_member_dim)
    # convert to CDF format, take weighted means, and output dictionary
    result = _pit_values_final_processing(pit_values, weights, dims_for_mean)

    return result


################################################
# Functions for Pit and Pit_fcst_at_obs methods
################################################


def _get_plotting_points(left: XarrayLike, right: XarrayLike) -> XarrayLike:
    """
    Given left- and right-hand limiting values of the PIT CDF
    (e.g. as output by `Pit().left` and `Pit().right`), concatenates these objects along
    the "pit_x_value" and sorts them along that dimension.
    The "pit_x_value" index will have duplicate values.
    """
    return xr.concat([left, right], "pit_x_value").sortby("pit_x_value")


def _get_plotting_points_param(left: XarrayLike, right: XarrayLike) -> dict:
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
    # only keep coordinates in left where (left != right) somewhere and where left is not NaN.
    dims_for_any = set(left.dims) - {"pit_x_value"}
    different = ((left != right) & left.notnull()).any(dims_for_any)
    left_reduced = left.where(different).dropna("pit_x_value", how="all")

    y_points = _get_plotting_points(left_reduced, right)

    if isinstance(y_points, xr.Dataset):
        x_points = xr.merge([y_points[var]["pit_x_value"].rename(var) for var in y_points.data_vars])
    else:
        x_points = y_points["pit_x_value"]  # .copy()
    # reindex with a plotting point index
    y_points = y_points.assign_coords(pit_x_value=np.arange(len(y_points.pit_x_value))).rename(
        {"pit_x_value": "plotting_point"}
    )
    x_points = x_points.assign_coords(pit_x_value=np.arange(len(x_points.pit_x_value))).rename(
        {"pit_x_value": "plotting_point"}
    )
    return {"x_plotting_position": x_points, "y_plotting_position": y_points}


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
    # need to find where the graph of the CDF crosses the diagonal.
    return np.abs(plotting_points - plotting_points["pit_x_value"]).integrate("pit_x_value")


def _alpha_score_array(left: xr.DataArray, right: xr.DataArray) -> xr.DataArray:
    """
    The alpha score is integrated absolute difference between the graph of the PIT CDF and the
    diagonal. This will be calculated using the trapezoidal rule. For the calculation
    to be exact, we need to include points where the PIT CDF and diagonal intersect.

    This function does this when the left and right limits of the PIT CDF are data arrays.

    Args:
        left: left-hand limit of the PIT CDF, indexed by "pit_x_value". Other dimensions possible.
        left: right-hand limit of the PIT CDF, indexed by "pit_x_value". Other dimensions possible.

    Returns:
        array of alpha scores, with "pit_x_value" dimension collapsed but other dimensions preserved.
    """
    # need to expand plotting_points to include points where the PIT CDF crosses the diagonal.
    # To do this, use interp1d since xarray via pandas interpolate cannot handle duplicate values in an index
    param_plotting_points = _get_plotting_points_param(left, right)
    intersection_points = _diagonal_intersection_points(param_plotting_points)
    plotting_points = _get_plotting_points(left, right)

    x_axis_num = plotting_points.get_axis_num("pit_x_value")
    y_values = plotting_points.values
    x_values = plotting_points["pit_x_value"].values
    # x_values_extended = np.sort(np.concatenate((x_values, intersection_points)))

    plotting_points_interpolated = interp1d(x_values, y_values, axis=x_axis_num, kind="linear")(intersection_points)

    # turn the numpy array into an xarray array
    interpolated_coords = dict(plotting_points.coords)
    interpolated_coords["pit_x_value"] = intersection_points
    plotting_points_interpolated = xr.DataArray(
        data=plotting_points_interpolated, dims=plotting_points.dims, coords=interpolated_coords
    )
    plotting_points = xr.concat([plotting_points, plotting_points_interpolated], "pit_x_value").sortby("pit_x_value")

    score = np.abs(plotting_points - plotting_points["pit_x_value"]).integrate("pit_x_value")
    return score


def _diagonal_intersection_points(param_plotting_points: dict) -> np.ndarray:
    """
    Gets the x values where the line y = x intersects with the piecewise linear graph
    y = F(x), where F is the PIT CDF. If the graph of F is discontinuous (vertical)
    at a point of intersection, or if F is coinncident with the diagonal over an open interval,
    then the corresponding points of intersection are not returned as they are not needed to
    calculate the alpha score.

    Only handles param_plotting_points in xarray data array format.

    Args:
        param_plotting_points: diction consisting of xarray data array output from
            `_get_plotting_points_param`

    Returns:
        1-dimenions numpy array of intersection points
    """
    x_pos = param_plotting_points["x_plotting_position"]
    y_pos = param_plotting_points["y_plotting_position"]
    # gradient of chord AB where A(x_pos[i-1], y_pos[i-1]), B(x_pos[i], y_pos[i])
    gradient = y_pos.diff("plotting_point") / x_pos.diff("plotting_point")
    # solution if there is a desired point of intersection, optained by solving
    # simultaneous equations for equation of line AB with line x = y
    x_solution = (y_pos - gradient * x_pos) / (1 - gradient)
    # for x_solution to be of interest, require that  x_pos[i-1] < x_solution[i] < x_pos[i]
    x_solution = x_solution.where((x_solution < x_pos) & (x_solution > x_pos.shift(plotting_point=1)))
    x_solution = np.unique(x_solution.values.flatten())
    return x_solution[~np.isnan(x_solution)]


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
