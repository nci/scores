"""Tools for matching data for verification"""

from typing import overload

import xarray as xr

from scores.typing import XarrayLike


# Dataset input types lead to a Dataset return type
@overload
def broadcast_and_match_nan(*args: xr.Dataset) -> tuple[xr.Dataset, ...]: ...


# Dataset input types lead to a Dataset return type
@overload
def broadcast_and_match_nan(
    *args: xr.DataArray,
) -> tuple[xr.DataArray, ...]: ...


def broadcast_and_match_nan(*args: XarrayLike) -> tuple[XarrayLike, ...]:
    """
    Input xarray data objects are 'matched' - they are broadcast against each
    other (forced to have the same dimensions), and the position of nans are
    forced onto all DataArrays. This matching process is applied across all
    supplied DataArrays, as well as all DataArrays inside supplied Datasets.

    Args:
        *args: any number of xarray data objects supplied as positional arguments. See
            examples below.

    Returns:
        A tuple of data objects of the same length as the number of data objects
        supplied as input. Each returned object is the 'matched' version of the
        input.

    Raises:
        ValueError: if any input args is not an xarray data
            object.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.processing import broadcast_and_match_nan
        >>> da1 = xr.DataArray([1.0, np.nan, 3.0], dims=['x'], coords={'x': [0, 1, 2]})
        >>> ds = xr.Dataset({
        ...      'temp': xr.DataArray([[10, 20, 30], [40, 50, 60]],
        ...      dims=['model', 'x'],
        ...      coords={'model': ['ECMWF', 'GFS'], 'x': [0, 1, 2]})
        ... })
        >>> da2 = xr.DataArray([4.0, 5.0, 6.0], dims=['x'], coords={'x': [0, 1, 2]})
        >>> da1_matched, ds_matched, da2_matched = broadcast_and_match_nan(da1, ds, da2)
        >>> da1_matched
        <xarray.DataArray (x: 3, model: 2)> Size: 48B
        array([[ 1.,  1.],
               [nan, nan],
               [ 3.,  3.]])
        Coordinates:
          * x        (x) int64 24B 0 1 2
          * model    (model) <U5 40B 'ECMWF' 'GFS'
        >>> ds_matched
        <xarray.Dataset> Size: 112B
        Dimensions:  (model: 2, x: 3)
        Coordinates:
          * model    (model) <U5 40B 'ECMWF' 'GFS'
          * x        (x) int64 24B 0 1 2
        Data variables:
            temp     (model, x) float64 48B 10.0 nan 30.0 40.0 nan 60.0
        >>> da2_matched
        <xarray.DataArray (x: 3, model: 2)> Size: 48B
        array([[ 4.,  4.],
               [nan, nan],
               [ 6.,  6.]])
        Coordinates:
          * x        (x) int64 24B 0 1 2
          * model    (model) <U5 40B 'ECMWF' 'GFS'
        >>> input_tuple = (da1, ds, da2)
        >>> matched_tuple = broadcast_and_match_nan(*input_tuple)
        >>> matched_tuple[0]
        <xarray.DataArray (x: 3, model: 2)> Size: 48B
        array([[ 1.,  1.],
               [nan, nan],
               [ 3.,  3.]])
        Coordinates:
          * x        (x) int64 24B 0 1 2
          * model    (model) <U5 40B 'ECMWF' 'GFS'
        >>> matched_tuple[1]
        <xarray.Dataset> Size: 112B
        Dimensions:  (model: 2, x: 3)
        Coordinates:
          * model    (model) <U5 40B 'ECMWF' 'GFS'
          * x        (x) int64 24B 0 1 2
        Data variables:
            temp     (model, x) float64 48B 10.0 nan 30.0 40.0 nan 60.0
        >>> matched_tuple[2]
        <xarray.DataArray (x: 3, model: 2)> Size: 48B
        array([[ 4.,  4.],
               [nan, nan],
               [ 6.,  6.]])
        Coordinates:
          * x        (x) int64 24B 0 1 2
          * model    (model) <U5 40B 'ECMWF' 'GFS'
    """

    # sanitise inputs
    for i, arg in enumerate(args):
        if not isinstance(arg, (xr.Dataset, xr.DataArray)):
            raise ValueError(
                f"Argument {i} is not an xarray data object. (counting from 0, i.e. argument 0 is the first argument)"
            )

    # internal function to update the mask
    def update_mask(mask, data_array):
        """
        Perform the boolean AND operation on a mask (DataArray) and
        data_array.notnull()
        """
        return mask & data_array.notnull()

    # initialise the mask
    mask = True
    # generate the mask
    for arg in args:
        # update the mask for a DataArray
        if isinstance(arg, xr.DataArray):
            mask = update_mask(mask, arg)
        # update the mask for Datasets
        else:
            for data_var in arg.data_vars:
                mask = update_mask(mask, arg[data_var])

    # return matched data objects
    return tuple(arg.where(mask) for arg in args)
