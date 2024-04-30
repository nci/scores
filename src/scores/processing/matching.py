"""Tools for matching data for verification"""

import xarray as xr

from scores.typing import XarrayLike


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

        >>> # Matching xarray data objects
        >>> da1_matched, ds_matched, da2_matched = scores.processing.broadcast_and_match_nan(da1, ds, da2)

        >>> # Matching a tuple of xarray data objects
        >>> input_tuple = (da1, ds, da2)
        >>> matched_tuple = broadcast_and_match_nan(*input_tuple)
        >>> da1_matched = matched_tuple[0]
        >>> ds_matched = matched_tuple[1]
        >>> da2_matched = matched_tuple[2]
    """

    # sanitise inputs
    for i, arg in enumerate(args):
        if not isinstance(arg, (xr.Dataset, xr.DataArray)):
            raise ValueError(
                f"Argument {i} is not an xarray data object. (counting from 0, i.e. "
                "argument 0 is the first argument)"
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
