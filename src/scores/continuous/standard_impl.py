"""
This module contains standard methods which may be used for continuous scoring
"""

import xarray as xr

import scores.functions
import scores.utils
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes


def mse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights: xr.DataArray = None,
    angular: bool = False,
):
    """Calculates the mean squared error from forecast and observed data.

    Dimensional reduction is not supported for pandas and the user should
    convert their data to xarray to formulate the call to the metric. At
    most one of reduce_dims and preserve_dims may be specified.
    Specifying both will result in an exception.

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
        angular: specifies whether `fcst` and `obs` are angular
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
    if angular:
        error = scores.functions.angular_difference(fcst, obs)
    else:
        error = fcst - obs
    squared = error * error
    squared = scores.functions.apply_weights(squared, weights)

    if preserve_dims or reduce_dims:
        reduce_dims = scores.utils.gather_dimensions(
            fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
        )

    if reduce_dims is not None:
        _mse = squared.mean(dim=reduce_dims)
    else:
        _mse = squared.mean()

    return _mse


def rmse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights: xr.DataArray = None,
    angular: bool = False,
) -> FlexibleArrayType:
    """Calculate the Root Mean Squared Error from xarray or pandas objects.

    A detailed explanation is on [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)


    Dimensional reduction is not supported for pandas and the user should
    convert their data to xarray to formulate the call to the metric.
    At most one of `reduce_dims` and `preserve_dims` may be specified.
    Specifying both will result in an exception.


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
        angular: specifies whether `fcst` and `obs` are angular
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
    _mse = mse(
        fcst=fcst, obs=obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights, angular=angular
    )

    _rmse = pow(_mse, (1 / 2))

    return _rmse


def mae(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights: xr.DataArray = None,
    angular: bool = False,
) -> FlexibleArrayType:
    """Calculates the mean absolute error from forecast and observed data.

    A detailed explanation is on [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)

    Dimensional reduction is not supported for pandas and the user should
    convert their data to xarray to formulate the call to the metric.
    At most one of reduce_dims and preserve_dims may be specified.
    Specifying both will result in an exception.

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
        angular: specifies whether `fcst` and `obs` are angular
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
    if angular:
        error = scores.functions.angular_difference(fcst, obs)
    else:
        error = fcst - obs
    ae = abs(error)
    ae = scores.functions.apply_weights(ae, weights)

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


def correlation(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
) -> FlexibleArrayType:
    """
    Calculates the Pearson's correlation coefficient between two xarray DataArrays

    Args:
        fcst: Forecast or predicted variables
        obs: Observed variables.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Pearson's correlation coefficient.
            All other dimensions will be preserved.
        preserve_dims: Optionally specify which dimensions to preserve when
            calculating the Pearson's correlation coefficient. All other dimensions will
            be reduced. As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same shape/dimensionality
            as the forecast, and the errors will be the absolute error at each
            point (i.e. single-value comparison against observed), and the
            forecast and observed dimensions must match precisely.
    """
    reduce_dims = scores.utils.gather_dimensions(
        fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )

    return xr.corr(fcst, obs, reduce_dims)
