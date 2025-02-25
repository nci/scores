"""
This module contains standard methods which may be used for continuous scoring
"""

from typing import Optional, Union, cast

import numpy as np
import xarray as xr

import scores.functions
import scores.utils
from scores.continuous.nse_impl import NseScore, NseScoreBuilder
from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes, XarrayLike


def mse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    is_angular: Optional[bool] = False,
) -> XarrayLike:
    """Calculates the mean squared error from forecast and observed data.

    See "Mean squared error" section at https://www.cawcr.gov.au/projects/verification/#MSE for more information

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2

    Args:
        fcst (Union[xr.Dataset, xr.DataArray, pd.Dataframe, pd.Series]):
            Forecast or predicted variables in xarray or pandas.
        obs (Union[xr.Dataset, xr.DataArray, pd.Dataframe, pd.Series]):
            Observed variables in xarray or pandas.
        reduce_dims (Union[str, Iterable[str]): Optionally specify which
            dimensions to reduce when calculating MSE. All other dimensions
            will be preserved.
        preserve_dims (Union[str, Iterable[str]): Optionally specify which
            dimensions to preserve when calculating MSE. All other dimensions
            will be reduced. As a special case, 'all' will allow all dimensions
            to be preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the squared error at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.


    Returns:
        Union[xr.Dataset, xr.DataArray, pd.Dataframe, pd.Series]: An object containing
            a single floating point number representing the mean absolute
            error for the supplied data. All dimensions will be reduced.
            Otherwise: Returns an object representing the mean squared error,
            reduced along the relevant dimensions and weighted appropriately.
    """
    if is_angular:
        error = scores.functions.angular_difference(fcst, obs)  # type: ignore
    else:
        error = fcst - obs  # type: ignore
    squared = error * error
    squared = scores.functions.apply_weights(squared, weights=weights)  # type: ignore

    if preserve_dims or reduce_dims:
        reduce_dims = scores.utils.gather_dimensions(
            fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
        )

    if reduce_dims is not None:
        _mse = squared.mean(dim=reduce_dims)  # type: ignore
    else:
        _mse = squared.mean()

    return _mse


def rmse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    is_angular: bool = False,
) -> FlexibleArrayType:
    """Calculate the Root Mean Squared Error

    A detailed explanation is on https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math ::
        \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (\\text{forecast}_i - \\text{observed}_i)^2}

    Args:
        fcst: Forecast
            or predicted variables in xarray or pandas.
        obs: Observed
            variables in xarray or pandas.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating RMSE. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating RMSE. All other dimensions will be reduced.
            As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the absolute error at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        An object containing
            a single floating point number representing the root mean squared
            error for the supplied data. All dimensions will be reduced.
            Otherwise: Returns an object representing the root mean squared error,
            reduced along the relevant dimensions and weighted appropriately.

    """
    _mse = mse(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights, is_angular=is_angular)

    _rmse = pow(_mse, (1 / 2))

    return _rmse  # type: ignore


def mae(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[xr.DataArray] = None,
    is_angular: bool = False,
) -> FlexibleArrayType:
    """Calculates the mean absolute error from forecast and observed data.

    A detailed explanation is on https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math ::
        \\frac{1}{n} \\sum_{i=1}^n | \\text{forecast}_i - \\text{observed}_i |

    Args:
        fcst: Forecast or predicted variables in xarray or pandas.
        obs: Observed variables in xarray or pandas.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating MAE. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating MAE. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)
        is_angular: specifies whether `fcst` and `obs` are angular
            data (e.g. wind direction). If True, a different function is used
            to calculate the difference between `fcst` and `obs`, which
            accounts for circularity. Angular `fcst` and `obs` data should be in
            degrees rather than radians.

    Returns:
        By default an xarray DataArray containing
        a single floating point number representing the mean absolute error for the
        supplied data. All dimensions will be reduced.

        Alternatively, an xarray structure with dimensions preserved as appropriate
        containing the score along reduced dimensions
    """
    if is_angular:
        error = scores.functions.angular_difference(fcst, obs)  # type: ignore
    else:
        error = fcst - obs  # type: ignore
    ae = abs(error)
    ae = scores.functions.apply_weights(ae, weights=weights)  # type: ignore

    if preserve_dims is not None or reduce_dims is not None:
        reduce_dims = scores.utils.gather_dimensions(
            fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
        )

    if reduce_dims is not None:
        _ae = ae.mean(dim=reduce_dims)
    else:
        _ae = ae.mean()

    # Returns unhinted types if nonstandard types passed in, but this is useful
    return _ae  # type: ignore


def mean_error(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the mean error which is also sometimes called the additive bias.

    It is defined as

    .. math::
        \\text{mean error} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}


    See "Mean error" section at https://www.cawcr.gov.au/projects/verification/ for more information

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the mean error. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the mean error. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        An xarray object with the mean error of a forecast.

    """
    return additive_bias(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights)


def additive_bias(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the additive bias which is also sometimes called the mean error.

    It is defined as

    .. math::
        \\text{Additive bias} =\\frac{1}{N}\\sum_{i=1}^{N}(x_i - y_i)
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}


    See "Mean error" section at https://www.cawcr.gov.au/projects/verification/ for more information

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the additive bias. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the additive bias. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        An xarray object with the additive bias of a forecast.

    """
    # Note - mean error call this function
    error = fcst - obs
    score = scores.functions.apply_weights(error, weights=weights)
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    score = score.mean(dim=reduce_dims)
    return score  # type: ignore


def multiplicative_bias(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the multiplicative bias.

    Most suited for forecasts that have a lower bound at 0 such as wind speed. Will return
    a np.inf where the mean of `obs` across the dims to be reduced is 0.
    It is defined as

    .. math::
        \\text{{Multiplicative bias}} = \\frac{\\frac{1}{N}\\sum_{i=1}^{N}x_i}{\\frac{1}{N}\\sum_{i=1}^{N}y_i}
        \\text{where } x = \\text{the forecast, and } y = \\text{the observation}

    See "(Multiplicative) bias" section at https://www.cawcr.gov.au/projects/verification/ for more information

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the multiplicative bias. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the multiplicative bias. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        An xarray object with the multiplicative bias of a forecast.

    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    fcst = scores.functions.apply_weights(fcst, weights=weights)
    obs = scores.functions.apply_weights(obs, weights=weights)

    # Need to broadcast and match NaNs so that the fcst mean and obs mean are for the
    # same points
    fcst, obs = broadcast_and_match_nan(fcst, obs)  # type: ignore
    multi_bias = fcst.mean(dim=reduce_dims) / obs.mean(dim=reduce_dims)
    return multi_bias


def pbias(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    weights: Optional[XarrayLike] = None,
) -> XarrayLike:
    """
    Calculates the percent bias, which is the ratio of the additive bias to the mean observed value, multiplied by 100.

    Percent bias is used for evaluating and comparing forecast accuracy across stations or datasets with varying magnitudes.
    By expressing the error as a percentage of the observed value, it allows for standardised comparisons, enabling assessment
    of forecast performance regardless of the absolute scale of values. Like :py:func:`scores.continuous.multiplicative_bias`,
    ``pbias`` will return a ``np.inf`` where the mean of ``obs`` across the dims to be reduced is 0. It is defined as

    .. math::
        \\text{Percent bias} = 100 \\cdot \\frac{\\sum_{i=1}^{N}(x_i - y_i)}{\\sum_{i=1}^{N} y_i}

    where:
        - :math:`x_i` = the values of x in a sample (i.e. forecast values)
        - :math:`y_i` = the values of y in a sample (i.e. observed values)

    See "pbias" section at https://search.r-project.org/CRAN/refmans/hydroGOF/html/pbias.html for more information

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the percent bias. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the percent bias. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
        weights: Optionally provide an array for weighted averaging (e.g. by area, by latitude,
            by population, custom)

    Returns:
        An xarray object with the percent bias of a forecast.

    References:
        -   Sorooshian, S., Duan, Q., & Gupta, V. K. (1993). Calibration of rainfall-runoff models:
            Application of global optimization to the Sacramento Soil Moisture Accounting Model.
            Water Resources Research, 29(4), 1185-1194. https://doi.org/10.1029/92WR02617
        -   Alfieri, L., Pappenberger, F., Wetterhall, F., Haiden, T., Richardson, D., & Salamon, P. (2014).
            Evaluation of ensemble streamflow predictions in Europe. Journal of Hydrology, 517, 913-922.
            https://doi.org/10.1016/j.jhydrol.2014.06.035
        -   Dawson, C. W., Abrahart, R. J., & See, L. M. (2007). HydroTest:
            A web-based toolbox of evaluation metrics for the standardised assessment of hydrological forecasts.
            Environmental Modelling and Software, 22(7), 1034-1052.
            https://doi.org/10.1016/j.envsoft.2006.06.008
        -   Moriasi, D. N., Arnold, J. G., Van Liew, M. W., Bingner, R. L., Harmel, R. D., & Veith, T. L. (2007).
            Model evaluation guidelines for systematic quantification of accuracy in watershed simulations.
            Transactions of the ASABE, 50(3), 885-900. https://doi.org/10.13031/2013.23153




    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    fcst = scores.functions.apply_weights(fcst, weights=weights)
    obs = scores.functions.apply_weights(obs, weights=weights)

    # Need to broadcast and match NaNs so that the mean error and obs mean are for the
    # same points
    fcst, obs = broadcast_and_match_nan(fcst, obs)  # type: ignore
    error = fcst - obs

    _pbias = 100 * error.mean(dim=reduce_dims) / obs.mean(dim=reduce_dims)
    return _pbias


def kge(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    scaling_factors: Optional[Union[list[float], np.ndarray]] = None,
    include_components: Optional[bool] = False,
) -> XarrayLike:
    # pylint: disable=too-many-locals
    """
    Calculate the Kling-Gupta Efficiency (KGE) between observed and simulated (or forecast) values.

    KGE is a performance metric that decomposes the error into three components:
    correlation, variability, and bias.
    It is computed as:

    .. math::
        \\text{KGE} = 1 - \\sqrt{\\left[s_\\rho \\cdot (\\rho - 1)\\right]^2 +
        \\left[s_\\alpha \\cdot (\\alpha - 1)\\right]^2 + \\left[s_\\beta \\cdot (\\beta - 1)\\right]^2}

    .. math::
        \\alpha = \\frac{\\sigma_x}{\\sigma_y}

    .. math::
        \\beta = \\frac{\\mu_x}{\\mu_y}

    where:
        - :math:`\\rho`  = Pearson's correlation coefficient between observed and forecast values as defined in :py:func:`scores.continuous.correlation.pearsonr`
        - :math:`\\alpha` is the ratio of the standard deviations (variability ratio)
        - :math:`\\beta` is the ratio of the means (bias)
        - :math:`x` and :math:`y` are forecast and observed values, respectively
        - :math:`\\mu_x` and :math:`\\mu_y` are the means of forecast and observed values, respectively
        - :math:`\\sigma_x` and :math:`\\sigma_y` are the standard deviations of forecast and observed values, respectively
        - :math:`s_\\rho`, :math:`s_\\alpha` and :math:`s_\\beta` are the scaling factors for the correlation coefficient :math:`\\rho`,
            the variability term :math:`\\alpha` and the bias term :math:`\\beta`

    Args:
        fcst: Forecast or predicted variables.
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the KGE. All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the KGE. All other dimensions will be reduced. As a
            special case, 'all' will allow all dimensions to be preserved. In
            this case, the result will be all NaN with the same shape/dimensionality
            as the forecast because the standard deviation is zero for a single point.
        scaling_factors : A 3-element vector or list describing the weights for each term in the KGE.
            Defined by: scaling_factors = [:math:`s_\\rho`, :math:`s_\\alpha`, :math:`s_\\beta`] to apply to the correlation term :math:`\\rho`,
            the variability term :math:`\\alpha` and the bias term :math:`\\beta` respectively. Defaults to (1.0, 1.0, 1.0). (*See
            equation 10 in Gupta et al. (2009) for definitions of them*).
        include_components (bool | False): If True, the function also returns the individual terms contributing to the KGE score.

    Returns:
        The Kling-Gupta Efficiency (KGE) score as an xarray DataArray.

        If ``include_components`` is True, the function returns ``xarray.Dataset`` kge_s with the following variables:

        - `kge`: The KGE score.
        - `rho`: The Pearson correlation coefficient.
        - `alpha`: The variability ratio.
        - `beta`: The bias term.

    Notes:
        - Statistics are calculated only from values for which both observations and
          simulations are not null values.
        - This function isn't set up to take weights.
        - Currently this function is working only on xr.DataArray.
        - When preserve_dims is set to 'all', the function returns NaN,
          similar to the Pearson correlation coefficient calculation for a single data point
          because the standard deviation is zero for a single point.

    References:
        -   Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and
            NSE performance criteria: Implications for improving hydrological modeling. Journal of Hydrology, 377(1-2), 80-91.
            https://doi.org/10.1016/j.jhydrol.2009.08.003.
        -   Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not?
            Comparing Nash-Sutcliffe and Kling-Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323-4331.
            https://doi.org/10.5194/hess-23-4323-2019.


    Examples:
        >>> kge_s = kge(forecasts, obs,preserve_dims='lat')  # if data is of dimension {lat,time}, kge value is computed across the time dimension
        >>> kge_s = kge(forecasts, obs,reduce_dims="time")  # if data is of dimension {lat,time}, reduce_dims="time" is same as preserve_dims='lat'
        >>> kge_s = kge(forecasts, obs, include_components=True) # kge_s is dataset of all three components and kge value itself
        >>> kge_s_weighted = kge(forecasts, obs, scaling_factors=(0.5, 1.0, 2.0)) # with scaling factors



    """

    # Type checks as xrray.corr can only handle xr.DataArray
    if not isinstance(fcst, xr.DataArray):
        raise TypeError("kge: fcst must be an xarray.DataArray")
    if not isinstance(obs, xr.DataArray):
        raise TypeError("kge: obs must be an xarray.DataArray")
    if scaling_factors is not None:
        if isinstance(scaling_factors, (list, np.ndarray)):
            # Check if the input has exactly 3 elements
            if len(scaling_factors) != 3:
                raise ValueError("kge: scaling_factors must contain exactly 3 elements")
        else:
            raise TypeError("kge: scaling_factors must be a list of floats or a numpy array")
    else:
        scaling_factors = [1.0, 1.0, 1.0]

    s_rho, s_alpha, s_beta = scaling_factors
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    # Need to broadcast and match NaNs so that the fcst and obs are for the
    # same points
    fcst, obs = broadcast_and_match_nan(fcst, obs)
    # compute linear correlation coefficient r between fcst and obs
    rho = xr.corr(fcst, obs, reduce_dims)  # type: ignore

    # compute alpha (sigma_sim / sigma_obs)
    sigma_fcst = fcst.std(reduce_dims)
    sigma_obs = obs.std(reduce_dims)
    alpha = sigma_fcst / sigma_obs

    # compute beta (mu_sim / mu_obs)
    mu_fcst = fcst.mean(reduce_dims)
    mu_obs = obs.mean(reduce_dims)
    beta = mu_fcst / mu_obs

    # compute Euclidian distance from the ideal point in the scaled space
    ed_s = np.sqrt((s_rho * (rho - 1)) ** 2 + (s_alpha * (alpha - 1)) ** 2 + (s_beta * (beta - 1)) ** 2)
    kge_s = 1 - ed_s
    if include_components:
        # Create dataset of all components
        kge_s = xr.Dataset(
            {
                "kge": kge_s,
                "rho": rho,
                "alpha": alpha,
                "beta": beta,
            }
        )
    return kge_s  # type: ignore


def nse(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    weights: XarrayLike | None = None,
    is_angular: bool | None = False,
) -> XarrayLike:
    """
    The Nash-Sutcliffe efficiency (NSE) is primarily used in hydrology to assess the skill of model
    predictions (of e.g. "discharge").

    While NSE is often calculated over observations and model predictions in the time dimension, it
    is actually a fairly generic statistical measure that determines the relative magnitude of the
    residual variance ("noise") compared to the measured data variance ("information") (Nash and
    Sutcliffe, 1970). Incidentally, it is (inversely) related to the signal-to-noise ratio (SNR).

    The general formulation of NSE is as follows:

    .. math::

        \\text{NSE} = 1 - \\frac{\\sum_i{(O_i - S_i)^2}}{\\sum_i{(O_i - \\bar{O})^2}}

    where
        - :math:`i` is a generic "indexer" representing the set of datapoints along the dimensions being
         reduced e.g. time (:math:`t`) or xy-coordinates (:math:`(x, y)`). The latter represents
         reduction over two dimensions as an example.
        - :math:`O_i` is the observation at the :math:`\\text{i^{th}}` index.
        - :math:`S_i` is the "forecast" or model simulation at the :math:`\\text{i^{th}}` index.
        - :math:`\\bar{O}` is the mean observation of the set of indexed samples as specified by
        ``reduce_dims`` and ``preserve_dims``.

    Args:
        fcst: "Forecast" or predicted variables.
        obs: Observed variables.
        reduce_dims: dimensions to reduce when calculating the NSE. (i.e. NSE will be calculated
            using datapoints along these dimensions as samples, the other dimensions will be
            preserved).
        preserve_dims: dimensions to preserve. Mutually exclusive to ``reduce_dims``. All
            dimensions not specified here will be reduced as described in ``reduce_dims``.
            Note: ``preserve_dims="all"`` is not supported for NSE. See notes below.
        weights: Optional weighting to apply to the NSE computation. Typically weights are applied
            over `time_dim` but can vary by location as well. Weights must be non-negative and
            specified for each data point _(i.e. the user must not assume broadcasting will handle
            appropriate assignment of weights for this score)_.
        is_angular: specifies whether `fcst` and `obs` are angular data (e.g. wind direction). If
            True, a different function is used to calculate the difference between `fcst` and `obs`,
            which accounts for circularity. Angular `fcst` and `obs` data should be in degrees
            rather than radians.

    Returns:
        ``xr.Dataset`` containing the NSE score for each preserved dimension.

    Raises:
        KeyError: If no dimensions are being reduced.
        IndexError: If the dimensions being reduced only have 1 value - not sufficient for
            computation of obs variance.
        RuntimeError: If something went wrong with the underlying implementation. The user will be
            prompted to report this as a github issue.
        UserWarning: If weights are negative.
        UserWarning: If attempting to divide by 0. The computation will still succeed but produce
            `np.nan` where divide by zero would occur.
        Exception: Any other errors or warnings not otherwise listed due to calculations associated
            with utility functions such as `gather_dimensions`.

    Supplementary details:
        - Nash-Sutcliffe efficiencies range from -Inf to 1. Essentially, the closer to 1, the more
          accurate the model is.
            - NSE = 1, corresponds to a perfect match of the model to the obs.
            - NSE = 0, indicates that the model is as accurate as the mean obs.
            - -Inf < NSE < 0, indicates that the mean obs is better predictor than the model.
        - The optional ``weights`` argument can additionally be used to perform a weighted NSE
          (wNSE). Although, this is a generic set of weights, and it is the _user's responsiblility_
          to define them appropriately. Typically this is the observation itself (Hundecha, Y., &
          Bárdossy, A., 2004).
        - ``weights`` must be non-negative. Therefore, the observations must ideally also be
          non-negative (or formulated appropriately) if used as weights.
        - While ``is_angular`` is typically not used for this score, NSE is generic enough that it
          _could_ be used in wider context, and hence is kept as an option. It is defaulted to
          ``False`` as that's the typical use-case.

    .. note::

        For Hydrology in particular :math:`i = t` i.e. the time dimension. In order to keep things
        generic, this function does not explicitly require the user to provide a "time" dimension
        to apply the score. Instead, it sets a hard requirement that _at least one_ dimension is
        being reduced from either a specification of ``reduce_dims`` or ``preserve_dims``
        (mutually exclusive).

        The reason is - the observation variance cannot be non-zero if nothing is being reduced.

        If the user is particularly interested in applying the NSE computation over the "time"
        dimension, they should provide the "time" dimension as one of the elements in the
        ``reduce_dims`` argument, and ensure that the input datasets and associated variables have
        this dimension. Or otherwise, omit it from the ``preserve_dims`` argument.

        As a side-effect of the above requirement, ``preserve_dims="all"`` is not allowed and will
        naturally throw an error.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.continuous import nse
        >>>
        >>> obs_raw = np.array(
        ...     [
        ...         [[1,2,3], [4,5,6]],
        ...         [[3,2,1], [6,5,4]],
        ...         [[3,2,5], [2,2,6]],
        ...         [[5,2,3], [4,-1,4]],
        ...     ]
        ...
        ... )  # dimension lengths: x=4, y=2, t=3
        >>>
        >>> obs = xr.DataArray(obs_raw, dims=["x", "y", "t"])
        >>> fcst = obs * 1.2 + 0.1  # add some synthetic bias and variance
        >>>
        >>> # Example 1:
        >>> # reduce over t - time - should produce a 4x2 array representing the xy coordinate grid
        >>> nse(obs, fcst, reduce_dims=["t"])
        <xarray.DataArray (x: 4, y: 2)> Size: 64B
        array([[ 0.71180556, -0.28819444],
               [ 0.71180556, -0.28819444],
               [ 0.70982143,  0.85742188],
               [ 0.70982143,  0.93208333]])
        Dimensions without coordinates: x, y
        >>>
        >>> # Example 2:
        >>> # reduce over (x, y) - space - should produce a 3x1 representing a time series
        >>> nse(obs, fcst, reduce_dims=["x", "y"])
        <xarray.DataArray (t: 3)> Size: 24B
        array([0.77469136, 0.90123457, 0.74722222])
        Dimensions without coordinates: t

    References:
        1. Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models
           part I — A discussion of principles. In Journal of Hydrology (Vol. 10, Issue 3, pp. 282–
           290). Elsevier BV. doi:10.1016/0022-1694(70)90255-6
        2. Hundecha, Y., & Bárdossy, A. (2004). Modeling of the effect of land use changes on the
           runoff generation of a river basin through parameter regionalization of a watershed
           model. Journal of Hydrology, 292(1-4), 281-295. doi:10.1016/j.jhydrol.2004.01.002
    """

    # setup arguments for builder to do checks
    # fmt: off
    nse_score: NseScore = NseScoreBuilder().build(
        obs=obs,
        fcst=fcst,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
        is_angular=is_angular,
    )
    # fmt: on

    # calculate the actual score: note - delayed if using dask.
    nse_result: XarrayLike = nse_score.calculate()

    return nse_result
