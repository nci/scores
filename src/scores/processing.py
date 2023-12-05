"""Tools for processing data for verification"""
import operator
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import gather_dimensions

INEQUALITY_MODES = {
    ">=": (operator.ge, -1),
    ">": (operator.gt, 1),
    "<=": (operator.le, 1),
    "<": (operator.lt, -1),
}
# '==' does not map to `operater.eq` and '!=' does not map to operator.ne on purpose.
# This is because we wish to test within a tolerance.
EQUALITY_MODES = {"==": (operator.le), "!=": (operator.gt)}


def check_binary(data: XarrayLike, name: str):
    """
    Checks that data does not have any non-NaN values out of the set {0, 1}

    Args:
        data: The data to convert to check if only contains binary values
    Raises:
        ValueError: if there are values in `fcst` and `obs` that are not in the
            set {0, 1, np.nan} and `check_args` is true.
    """
    if isinstance(data, xr.DataArray):
        unique_values = pd.unique(data.values.flatten())
    else:
        unique_values = pd.unique(data.to_array().values.flatten())
    unique_values = unique_values[~np.isnan(unique_values)]
    binary_set = {0, 1}

    if not set(unique_values).issubset(binary_set):
        raise ValueError(f"`{name}` contains values that are not in the set {{0, 1, np.nan}}")


def comparative_discretise(
    data: XarrayLike, comparison: Union[xr.DataArray, float, int], mode: str, abs_tolerance: Optional[float] = None
) -> XarrayLike:
    """
    Converts the values of `data` to 0 or 1 based on how they relate to the specified
    values in `comparison` via the `mode` operator.

    Args:
        data: The data to convert to
            discrete values.
        comparison: The values to which
            to compare `data`.
        mode: Specifies the required relation of `data` to `thresholds`
            for a value to fall in the 'event' category (i.e. assigned to 1).
            Allowed modes are:
            - '>=' values in `data` greater than or equal to the
            corresponding threshold are assigned as 1.
            - '>' values in `data` greater than the corresponding threshold
            are assigned as 1.
            - '<=' values in `data` less than or equal to the corresponding
            threshold are assigned as 1.
            - '<' values in `data` less than the corresponding threshold
            are assigned as 1.
            - '==' values in `data` equal to the corresponding threshold
            are assigned as 1
            - '!=' values in `data` not equal to the corresponding threshold
            are assigned as 1.
        abs_tolerance: If supplied, values in data that are
            within abs_tolerance of a threshold are considered to be equal to
            that threshold. This is generally used to correct for floating
            point rounding, e.g. we may want to consider 1.0000000000000002 as
            equal to 1.
    Returns:
        An xarray data object of the same type as `data`. The dimensions of the
        output are the union of all dimensions in `data` and `comparison`. The
        values of the output are either 0 or 1 or NaN, depending on the truth
        of the operation `data <mode> comparison`.
    Raises:
       ValueError: if abs_tolerance is not a non-negative float.
       ValueError: if `mode` is not valid.
       TypeError: if `comparison` is not a float, int or xarray.DataArray.
    """

    # sanitise abs_tolerance
    if abs_tolerance is None:
        abs_tolerance = 0
    elif abs_tolerance < 0:
        raise ValueError(f"value {abs_tolerance} of abs_tolerance is invalid, it must be a non-negative float")

    if isinstance(comparison, (float, int)):
        comparison = xr.DataArray(comparison)
    elif not isinstance(comparison, xr.DataArray):
        raise TypeError("comparison must be a float, int or xarray.DataArray")

    # mask to preserve NaN in data and comparison
    notnull_mask = data.notnull() * comparison.notnull()

    # do the discretisation
    if mode in INEQUALITY_MODES:
        operator_func, factor = INEQUALITY_MODES[mode]
        discrete_data = operator_func(data, comparison + (abs_tolerance * factor)).where(notnull_mask)
    elif mode in EQUALITY_MODES:
        operator_func = EQUALITY_MODES[mode]
        discrete_data = operator_func(abs(data - comparison), abs_tolerance).where(notnull_mask)
    else:
        raise ValueError(
            f"'{mode}' is not a valid mode. Available modes are: "
            f"{sorted(INEQUALITY_MODES) + sorted(EQUALITY_MODES)}"
        )
    discrete_data.attrs["discretisation_tolerance"] = abs_tolerance
    discrete_data.attrs["discretisation_mode"] = mode

    return discrete_data


def binary_discretise(
    data: XarrayLike,
    thresholds: FlexibleDimensionTypes,
    mode: str,
    abs_tolerance: Optional[float] = None,
    autosqueeze: Optional[bool] = False,
):
    """
    Converts the values of `data` to 0 or 1 for each threshold in `thresholds`
    according to the operation defined by `mode`.

    Args:
        data: The data to convert to
            discrete values.
        thresholds: Threshold(s) at which to convert the
            values of `data` to 0 or 1.
        mode: Specifies the required relation of `data` to `thresholds`
            for a value to fall in the 'event' category (i.e. assigned to 1).
            Allowed modes are:

            - '>=' values in `data` greater than or equal to the
            corresponding threshold are assigned as 1.
            - '>' values in `data` greater than the corresponding threshold
            are assigned as 1.
            - '<=' values in `data` less than or equal to the corresponding
            threshold are assigned as 1.
            - '<' values in `data` less than the corresponding threshold
            are assigned as 1.
            - '==' values in `data` equal to the corresponding threshold
            are assigned as 1
            - '!=' values in `data` not equal to the corresponding threshold
            are assigned as 1.

        abs_tolerance: If supplied, values in data that are
            within abs_tolerance of a threshold are considered to be equal to
            that threshold. This is generally used to correct for floating
            point rounding, e.g. we may want to consider 1.0000000000000002 as
            equal to 1

        autosqueeze: If True and only one threshold is
            supplied, then the dimension 'threshold' is squeezed out of the
            output. If `thresholds` is float-like, then this is forced to
            True, otherwise defaults to False.

    Returns:
        An xarray data object with the type and dimensions of `data`, plus an
        extra dimension 'threshold' if `autosqueeze` is False. The values of
        the output are either 0 or 1, depending on whether `data <mode> threshold`
        is True or not (although NaNs are preserved).

    Raises:
        ValueError: if 'threshold' is a dimension in `data`.
        ValueError: if "Values in `thresholds` are not montonic increasing"
    """
    if "threshold" in data.dims:
        raise ValueError("'threshold' must not be in the supplied data object dimensions")

    # if thresholds is 0-D, convert it to a length-1 1-D array
    # but autosqueeze=True so the 'threshold' dimension is dropped
    thresholds_np = np.array(thresholds)
    if thresholds_np.ndim == 0:
        thresholds_np = np.expand_dims(thresholds_np, 0)
        autosqueeze = True

    # sanitise thresholds
    if not (thresholds_np[1:] - thresholds_np[:-1] >= 0).all():
        raise ValueError("Values in `thresholds` are not montonic increasing")

    # make thresholds DataArray
    thresholds_da = xr.DataArray(thresholds_np, dims=["threshold"], coords={"threshold": thresholds_np})

    # do the discretisation
    discrete_data = comparative_discretise(data, thresholds_da, mode, abs_tolerance=abs_tolerance)

    # squeeze
    if autosqueeze and len(thresholds_np) == 1:
        # squeeze out the 'threshold' dimension, but keep the coordinate
        discrete_data = discrete_data.squeeze(dim="threshold")

    return discrete_data


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
        >>> da1_matched, ds_matched, da2_matched = xrtools.broadcast_and_match_nan(da1, ds, da2)

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
        elif isinstance(arg, xr.Dataset):
            for data_var in arg.data_vars:
                mask = update_mask(mask, arg[data_var])

    # return matched data objects
    return tuple(arg.where(mask) for arg in args)


def proportion_exceeding(
    data: XarrayLike,
    thresholds: Iterable,
    preserve_dims: FlexibleDimensionTypes = None,
    reduce_dims: FlexibleDimensionTypes = None,
):
    """
    Calculates the proportion of `data` equal to or exceeding `thresholds`.

    Args:
        data (xarray.Dataset or xarray.DataArray): The data from which
            to calculate the proportion exceeding `thresholds`
        thresholds (iterable): The proportion of Flip-Flop index results
            equal to or exceeding these thresholds will be calculated.
            the flip-flop index.
        dims (Optional[iterable]): Strings corresponding to the dimensions in the input
            xarray data objects that we wish to preserve in the output. All other
            dimensions in the input data objects are collapsed.

    Returns:
        An xarray data object with the type of `data` and dimensions
        `dims` + 'threshold'. The values are the proportion of `data`
        that are greater than or equal to the corresponding threshold.

    """
    return _binary_discretise_proportion(data, thresholds, ">=", preserve_dims, reduce_dims)


def _binary_discretise_proportion(
    data: XarrayLike,
    thresholds: Iterable,
    mode: str,
    preserve_dims: FlexibleDimensionTypes = None,
    reduce_dims: FlexibleDimensionTypes = None,
    abs_tolerance: Optional[bool] = None,
    autosqueeze: bool = False,
):
    """
    Returns the proportion of `data` in each category. The categories are
    defined by the relationship of data to threshold as specified by
    the operation `mode`.

    Args:
        data: The data to convert
           into 0 and 1 according the thresholds before calculating the
           proportion.
        thresholds: The proportion of Flip-Flop index results
            equal to or exceeding these thresholds will be calculated.
            the flip-flop index.
        mode: Specifies the required relation of `data` to `thresholds`
            for a value to fall in the 'event' category (i.e. assigned to 1).
            Allowed modes are:

            - '>=' values in `data` greater than or equal to the
              corresponding threshold are assigned as 1.
            - '>' values in `data` greater than the corresponding threshold
              are assigned as 1.
            - '<=' values in `data` less than or equal to the corresponding
              threshold are assigned as 1.
            - '<' values in `data` less than the corresponding threshold
              are assigned as 1.
            - '==' values in `data` equal to the corresponding threshold
              are assigned as 1
            - '!=' values in `data` not equal to the corresponding threshold
              are assigned as 1.
        reduce_dims: Dimensions to reduce.
        preserve_dims: Dimensions to preserve.
        abs_tolerance: If supplied, values in data that are
            within abs_tolerance of a threshold are considered to be equal to
            that threshold. This is generally used to correct for floating
            point rounding, e.g. we may want to consider 1.0000000000000002 as
            equal to 1.
        autosqueeze: If True and only one threshold is
            supplied, then the dimension 'threshold' is squeezed out of the
            output. If `thresholds` is float-like, then this is forced to
            True, otherwise defaults to False.

    Returns:
        An xarray data object with the type of `data`, dimension `dims` +
        'threshold'. The values of the output are the proportion of `data` that
        satisfy the relationship to `thresholds` as specified by `mode`.

    Examples:

        >>> data = xr.DataArray([0, 0.5, 0.5, 1])

        >>> _binary_discretise_proportion(data, [0, 0.5, 1], '==')
        <xarray.DataArray (threshold: 3)>
        array([ 0.25,  0.5 ,  0.25])
        Coordinates:
          * threshold  (threshold) float64 0.0 0.5 1.0
        Attributes:
            discretisation_tolerance: 0
            discretisation_mode: ==

        >>> _binary_discretise_proportion(data, [0, 0.5, 1], '>=')
        <xarray.DataArray (threshold: 3)>
        array([ 1.  ,  0.75,  0.25])
        Coordinates:
          * threshold  (threshold) float64 0.0 0.5 1.0
        Attributes:
            discretisation_tolerance: 0
            discretisation_mode: >=

    See also:
        `scores.processing.binary_discretise`

    """
    # values are 1 when (data {mode} threshold), and 0 when ~(data {mode} threshold).
    discrete_data = binary_discretise(data, thresholds, mode, abs_tolerance=abs_tolerance, autosqueeze=autosqueeze)

    # The proportion in each category
    dims = gather_dimensions(data.dims, data.dims, preserve_dims, reduce_dims)
    proportion = discrete_data.mean(dim=dims)

    # attach attributes
    proportion.attrs = discrete_data.attrs

    return proportion
