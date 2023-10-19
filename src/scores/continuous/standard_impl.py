"""
This module contains standard methods which may be used for continuous scoring
"""

import numpy
import pandas

import scores.functions
import scores.utils
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes


def mse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights=None,
):
    """Calculates the mean squared error from forecast and observed data.

    Dimensional reduction is not supported for pandas dataframes and the user should
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
        weights: Not yet implemented. Allow weighted averaging (e.g. by
            area, by latitude, by population, custom)

    Returns:
        Union[xr.Dataset, xr.DataArray, pd.Dataframe, pd.Series]: An object containing
            a single floating point number representing the mean absolute
            error for the supplied data. All dimensions will be reduced.
            Otherwise: Returns an object representing the mean squared error,
            reduced along the relevant dimensions and weighted appropriately.
    """
    as_pandas_series = False
    both_pandas = False
    if isinstance(fcst, pandas.Series):
        fcst = fcst.to_xarray()
        as_pandas_series = True

    if isinstance(obs, pandas.Series):
        obs = obs.to_xarray()
        as_pandas_series = True
        if as_pandas_series is True:
            both_pandas = True

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

    # If two pandas inputs are provided, return as expected from pandas
    # If at least one xarray is provided, return as expected from xarray
    if both_pandas:
        _mse = _mse.to_pandas()
        if isinstance(_mse, numpy.ndarray):
            _mse = numpy.float64(_mse)

    return _mse


def rmse(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights=None,
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
      weights: Not yet implemented. Allow weighted averaging (e.g. by
          area, by latitude, by population, custom)

    Returns:
        An object containing
            a single floating point number representing the root mean squared
            error for the supplied data. All dimensions will be reduced.
            Otherwise: Returns an object representing the root mean squared error,
            reduced along the relevant dimensions and weighted appropriately.

    """
    _mse = mse(fcst=fcst, obs=obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, weights=weights)

    _rmse = pow(_mse, (1 / 2))

    return _rmse


def mae(
    fcst: FlexibleArrayType,
    obs: FlexibleArrayType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    weights=None,
) -> FlexibleArrayType:
    """Calculates the mean absolute error from forecast and observed data.

    A detailed explanation is on [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)

    Dimensional reduction is not supported for pandas dataframes and the user should
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
        weights:
            Not yet implemented. Allow weighted averaging (e.g. by area, by
            latitude, by population, custom).

    Returns:
        By default an xarray DataArray containing
        a single floating point number representing the mean absolute error for the
        supplied data. All dimensions will be reduced.

        Alternatively, an xarray structure with dimensions preserved as appropriate
        containing the score along reduced dimensions
    """
    as_pandas_series = False
    both_pandas = False
    if isinstance(fcst, pandas.Series):
        fcst = fcst.to_xarray()
        as_pandas_series = True

    if isinstance(obs, pandas.Series):
        obs = obs.to_xarray()
        as_pandas_series = True
        if as_pandas_series is True:
            both_pandas = True

    error = fcst - obs
    ae = abs(error)
    ae = scores.functions.apply_weights(ae, weights)

    if preserve_dims is not None or reduce_dims is not None:
        reduce_dims = scores.utils.gather_dimensions(fcst.dims, obs.dims, reduce_dims, preserve_dims)

    if reduce_dims is not None:
        _ae = ae.mean(dim=reduce_dims)
    else:
        _ae = ae.mean()

    # If two pandas inputs are provided, return as expected from pandas
    # If at least one xarray is provided, return as expected from xarray
    if both_pandas:
        _ae = _ae.to_pandas()
        if isinstance(_ae, numpy.ndarray):
            _ae = numpy.float64(_ae)

    return _ae
