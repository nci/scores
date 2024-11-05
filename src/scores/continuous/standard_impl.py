"""
This module contains standard methods which may be used for continuous scoring
"""

from typing import Optional, Union

import numpy as np
import xarray as xr

import scores.functions
import scores.utils
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
    fcst, obs = broadcast_and_match_nan(fcst, obs)
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
    fcst, obs = broadcast_and_match_nan(fcst, obs)
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
