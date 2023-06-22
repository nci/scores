"""
This module contains methods which may be used for continuous scoring
"""

import scores.utils


def mse(fcst, obs, reduce_dims=None, preserve_dims=None, weights=None):
    """

    Returns:
      - By default an xarray containing a single floating point number representing the mean absolute
        error for the supplied data. All dimensions will be reduced.
      - Otherwise: Returns an xarray representing the mean squared error, reduced along
      the relevant dimensions and weighted appropriately.

    Args:
      - fcst: Forecast or predicted variables in xarray or pandas
      - obs: Observed variables in xarray or pandas
      - reduce_dims: Optionally specify which dimensions to reduce when calculating MSE.
                     All other dimensions will be preserved.
      - preserve_dims: Optionally specify which dimensions to preserve when calculating MSE. All other
                       dimensions will be reduced. As a special case, 'all' will allow all dimensions to
                       be preserved. In this case, the result will be in the same shape/dimensionality as
                       the forecast, and the errors will be the squared error at each point (i.e. single-value
                       comparison against observed), and the forecast and observed dimensions must match
                       precisely.
      - weights: Not yet implemented. Allow weighted averaging (e.g. by area, by latitude, by population, custom)

    Notes:
      - Dimensional reduction is not supported for pandas and the user should convert their data to xarray
        to formulate the call to the metric.
      - At most one of reduce_dims and preserve_dims may be specified. Specifying both will result in an exception.
    """

    error = fcst - obs

    squared = error * error

    if weights is not None:  # pragma: no-cover
        raise NotImplementedError("Weights handling not implemented, placeholder for API spec")  # pragma: no-cover

    weights_dims = []
    if preserve_dims or reduce_dims:
        reduce_dims = scores.utils.gather_dimensions(fcst.dims, obs.dims, weights_dims, reduce_dims, preserve_dims)

    if reduce_dims is not None:
        _mse = squared.mean(dim=reduce_dims)
    else:
        _mse = squared.mean()

    return _mse


def mae(fcst, obs, reduce_dims=None, preserve_dims=None, weights=None):
    """**Needs a 1 liner function description**
    Args:
      - fcst: Forecast or predicted variables in xarray or pandas.
      - obs: Observed variables in xarray or pandas.
      - reduce_dims: Optionally specify which dimensions to reduce when
          calculating MAE. All other dimensions will be preserved.
      - preserve_dims: Optionally specify which dimensions to preserve
          when calculating MAE. All other dimensions will be reduced.
          As a special case, 'all' will allow all dimensions to be
          preserved. In this case, the result will be in the same
          shape/dimensionality as the forecast, and the errors will be
          the absolute error at each point (i.e. single-value comparison
          against observed), and the forecast and observed dimensions
          must match precisely.
      - weights: Not yet implemented. Allow weighted averaging (e.g. by
          area, by latitude, by population, custom).

    Returns:
      - By default an xarray DataArray containing a single floating
        point number representing the mean absolute error for the
        supplied data. All dimensions will be reduced.

        Alternatively, an xarray structure with dimensions preserved as
        appropriate containing the score along reduced dimensions

    Notes:
      - Dimensional reduction is not supported for pandas and the user
        should convert their data to xarray to formulate the call to the metric.
      - At most one of reduce_dims and preserve_dims may be specified.
        Specifying both will result in an exception.

    A detailed explanation is on [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
    """

    error = fcst - obs
    ae = abs(error)

    if weights is not None:  # pragma: no-cover
        raise NotImplementedError("Weights handling not implemented, placeholder for API spec")  # pragma: no-cover

    weights_dims = []
    if preserve_dims is not None or reduce_dims is not None:
        reduce_dims = scores.utils.gather_dimensions(fcst.dims, obs.dims, weights_dims, reduce_dims, preserve_dims)

    if reduce_dims is not None:
        _ae = ae.mean(dim=reduce_dims)
    else:
        _ae = ae.mean()

    return _ae
