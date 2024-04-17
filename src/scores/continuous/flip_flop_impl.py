"""
This module contains functions for calculating Flip-Flop indices
"""

from collections.abc import Generator, Iterable, Sequence
from typing import Optional, Union, overload

import numpy as np
import xarray as xr

from scores.functions import angular_difference
from scores.processing import proportion_exceeding
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import DimensionError, check_dims, dims_complement


def _flip_flop_index(
    data: xr.DataArray, sampling_dim: str, *, is_angular: bool = False  # Force keywords arguments to be keyword-only
) -> xr.DataArray:
    """
    Calculates the Flip-Flop Index by collapsing the dimension specified by
    `sampling_dim`.

    Args:
        data: Data from which to draw subsets.
        sampling_dim: The name of the dimension along which to calculate
            the Flip-Flop Index.
        is_angular: specifies whether `data` is directional data (e.g. wind
            direction).

    Returns:
        A xarray.DataArray of the Flip-Flop Index with the dimensions of
        `data`, except for the `sampling_dim` dimension which is collapsed.

    See also:
        `scores.continuous.flip_flop.flip_flop_index`
    """
    # check that `sampling_dim` is in `data`.
    check_dims(data, [sampling_dim], mode="superset")
    # the maximum possible number of discrete flip_flops
    sequence_length = len(data[sampling_dim])
    max_possible_flip_flop_count = sequence_length - 2

    # calculate the range
    # skip_na=False guarantees that if there is a nan in that row,
    # it will show up as nan in the end
    if is_angular:
        # get complementary dimensions as `encompassing_sector_size` takes
        # dimensions to be preserved, not collapsed
        dims_to_preserve = dims_complement(data, dims=[sampling_dim])
        # get maximum forecast range, if > 180 then clip to 180 as this is the
        # maximum possible angular difference between two forecasts
        enc_size = encompassing_sector_size(data=data, dims=dims_to_preserve)
        range_val = np.clip(enc_size, a_min=None, a_max=180.0)
        flip_flop = angular_difference(data.shift({sampling_dim: 1}), data)
    else:
        max_val = data.max(dim=sampling_dim, skipna=False)
        min_val = data.min(dim=sampling_dim, skipna=False)
        range_val = max_val - min_val  # type: ignore
        # subtract each consecutive 'row' from eachother
        flip_flop = data.shift({sampling_dim: 1}) - data

    # take the absolute value and sum.
    # I don't do skipna=False here because .shift makes a row of nan
    flip_flop = abs(flip_flop).sum(dim=sampling_dim)
    # adjust based on the range. This is where nan will be introduced.
    flip_flop = flip_flop - range_val
    # normalise
    return flip_flop / max_possible_flip_flop_count


# If there are selections, a DataSet is always returned
@overload
def flip_flop_index(
    data: xr.DataArray,
    sampling_dim: str,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
    **selections: Iterable[int],
) -> xr.Dataset:
    ...


# If there are no selections, a DataArray is always returned
@overload
def flip_flop_index(
    data: xr.DataArray,
    sampling_dim: str,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
    **selections: None,
) -> xr.DataArray:
    ...


# Return type is more precise at runtime when it is known if selections are being used
def flip_flop_index(
    data: xr.DataArray,
    sampling_dim: str,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
    **selections: Optional[Iterable[int]],
) -> XarrayLike:
    """
    Calculates the Flip-Flop Index along the dimensions `sampling_dim`.

    Args:
        data: Data from which to draw subsets.
        sampling_dim: The name of the dimension along which to calculate
            the Flip-Flop Index.
        is_angular: specifies whether `data` is directional data (e.g. wind
            direction).
        **selections: Additional keyword arguments specify
            subsets to draw from the dimension `sampling_dim` of the supplied `data`
            before calculation of the Flip_Flop Index. e.g. days123=[1, 2, 3]

    Returns:
        If `selections` are not supplied: An xarray.DataArray, the Flip-Flop
        Index by collapsing the dimension `sampling_dim`.

        If `selections` are supplied: An xarray.Dataset. Each data variable
        is a supplied key-word argument, and corresponds to selecting the
        values specified from `sampling_dim` of `data`. The Flip-Flop Index
        is calculated for each of these selections.

    Notes:

        .. math::

            \\text{{Flip-Flop Index}} = \\frac{{1}}{{N-2}}
            \\left [
            \\left(\\sum\\limits_{{i=1}}^{{N-1}}|x_i - x_{{i+1}}|\\right)
            - \\left(\\max_{{j}}\\{{x_j\\}} - \\min_{{j}}\\{{x_j\\}}\\right)
            \\right ]

        Where :math:`N` is the number of data points, and :math:`x_i` is the
        :math:`i^{{\\text{{th}}}}` data point.

    Examples:
        >>> data = xr.DataArray([50, 20, 40, 80], coords={{'lead_day': [1, 2, 3, 4]}})

        >>> flip_flop_index(data, 'lead_day')
        <xarray.DataArray ()>
        array(15.0)
        Attributes:
            sampling_dim: lead_day

        >>> flip_flop_index(data, 'lead_day', days123=[1, 2, 3], all_days=[1, 2, 3, 4])
        <xarray.Dataset>
        Dimensions:   ()
        Coordinates:
            *empty*
        Data variables:
            days123   float64 20.0
            all_days  float64 15.0
        Attributes:
            selections: {{'days123': [1, 2, 3], 'all_days': [1, 2, 3, 4]}}
            sampling_dim: lead_day

    """

    if not selections and isinstance(data, xr.DataArray):
        result = _flip_flop_index(data, sampling_dim, is_angular=is_angular)
    else:
        result = xr.Dataset()  # type: ignore
        result.attrs["selections"] = selections
        for key, data_subset in iter_selections(data, sampling_dim, **selections):
            result[key] = _flip_flop_index(data_subset, sampling_dim, is_angular=is_angular)
    result.attrs["sampling_dim"] = sampling_dim

    return result


# DataArray input types lead to DataArray output types
@overload
def iter_selections(
    data: xr.DataArray, sampling_dim: str, **selections: Optional[Iterable[int]]
) -> Generator[tuple[str, xr.DataArray], None, None]:
    ...


# Dataset input types load to Dataset output types
@overload
def iter_selections(  # type: ignore
    data: xr.Dataset, sampling_dim: str, **selections: Optional[Iterable[int]]
) -> Generator[tuple[str, xr.Dataset], None, None]:
    ...


def iter_selections(
    data: XarrayLike, sampling_dim: str, **selections: Optional[Iterable[int]]
) -> Generator[tuple[str, XarrayLike], None, None]:
    """
    Selects subsets of data along dimension sampling_dim according to
    `selections`.

    Args:
        data: The data to sample from.
        sampling_dim: The dimension from which to sample.
        selections: Each supplied keyword corresponds to a
            selection of `data` from the dimensions `sampling_dim`. The
            key is the first element of the yielded tuple.

    Yields:
        A tuple (key, data_subset), where key is the supplied `**selections`
        keyword, and data_subset is the `data` at the values along
        `sampling_dim` specified by `**selections`.

    Raises:
        KeyError: values in selections are not in data[sampling_dim]

    Examples:
        >>> data = xr.DataArray(
        ...     [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
        ...     coords={'lead_day': [1, 2, 3, 4, 5, 6, 7]}
        ... )
        >>> for key, data_subset in iter_selections(
        ...         data, 'lead_day', days123=[1, 2, 3], all_days=[1, 2, 3, 4, 5, 6, 7]
        ... ):
        ...     print(key, ':', data_subset)
        all_days : <xarray.DataArray (lead_day: 7)>
        array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.7])
        Coordinates:
          * lead_day  (lead_day) int64 1 2 3 4 5 6 7
        days123 : <xarray.DataArray (lead_day: 3)>
        array([ 0. ,  0.1,  0.2])
        Coordinates:
          * lead_day  (lead_day) int64 1 2 3

    """
    check_dims(data, [sampling_dim], mode="superset")

    for key, values in selections.items():
        try:
            # Need copy so that attributes added in _iter_selections_with_attrs
            # don't affect the whole dataframe but just the subset
            data_subset = data.sel({sampling_dim: values}).copy(deep=False)
        except KeyError as ex:
            raise KeyError(
                f"for `selections` item {str({key: values})}, not all values found in " f"dimension '{sampling_dim}'",
            ) from ex

        yield key, data_subset


def encompassing_sector_size(
    data: xr.DataArray, dims: Sequence[str], *, skipna: bool = False  # Force keywords arguments to be keyword-only
) -> xr.DataArray:
    """
    Calculates the minimum angular distance which encompasses all data points
    within an xarray.DataArray along a specified dimension. Assumes data is in
    degrees.
    Only one dimension may be collapsed each time, so length of dims must be of
    length one less than the length of data.dims otherwise an exception will be
    raised.

    Args:
        data: direction data in degrees
        dims: Strings corresponding to the dimensions in the input
            xarray data objects that we wish to preserve in the output. All other
            dimensions in the input data objects are collapsed.
        skipna: specifies whether to ignore nans in the data. If False
            (default), will return a nan if one or more nans are present

    Returns:
        an xarray.DataArray of minimum encompassing sector sizes with
        dimensions `dims`.

    Raises:
        scores.utils.DimensionError: raised if

            - the set of data dimensions is not a proper superset of `dims`
            - dimension to be collapsed isn't 1
    """
    check_dims(data, dims, mode="proper superset")
    dims_to_collapse = dims_complement(data, dims=dims)
    if len(dims_to_collapse) != 1:
        raise DimensionError("can only collapse one dimension")
    dim_to_collapse = dims_to_collapse[0]
    axis_to_collapse = data.get_axis_num(dim_to_collapse)
    values = _encompassing_sector_size_np(
        data=data.values,
        axis_to_collapse=axis_to_collapse,
        skipna=skipna,
    )
    new_dims = [dim for dim in data.dims if dim in dims]
    coords = [data.coords[dim] for dim in new_dims]
    result = xr.DataArray(values, dims=new_dims, coords=coords)
    return result


@np.errstate(invalid="ignore")
def _encompassing_sector_size_np(
    data: np.ndarray,
    *,  # Force keywords arguments to be keyword-only
    axis_to_collapse: Union[int, tuple[int, ...]] = 0,
    skipna: bool = False,
) -> np.ndarray:
    """
    Calculates the minimum angular distance which encompasses all data points
    within an xarray.DataArray along a specified dimension. Assumes data is in
    degrees.

    Args:
        data: direction data in degrees
        axis_to_collapse: number of axis to collapse in data, the numpy.ndarray
        skipna: specifies whether to ignore nans in the data. If False
            (default), will return a nan if one or more nans are present

    Returns:
        an numpy.ndarray of minimum encompassing sector sizes
    """
    # code will be simpler, and makes broadcasting easier if we are dealing
    # with the axis=0
    data = np.moveaxis(data, axis_to_collapse, 0)
    # make data in range [0, 360)
    data = data % 360
    data = np.sort(data, axis=0)
    if skipna:
        # rotate so one angle is at zero, then we can replace Nans with zeroes
        if data.ndim == 1:
            data = (data - data[0]) % 360
        else:
            data = (data - data[0, :]) % 360
        all_nans = np.all(np.isnan(data), axis=0)
        # if all NaNs, we don't want to change, and still want end result to be
        # NaN.
        # if some NaNs but not all, then set to zero, which will just end up
        # being a duplicate value after we've rotated so at least one zero value
        data[np.isnan(data) & ~all_nans] = 0
    # make a back-shifted copy of `data`
    data_rolled = np.roll(data, shift=-1, axis=0)
    # determine absolute angular difference between all adjacent angles
    angular_diffs = np.abs(data - data_rolled)
    angular_diffs = np.where(
        # nan_to_num so doesn't complain about comparing with NaN
        np.nan_to_num(angular_diffs) > 180,
        360 - angular_diffs,
        angular_diffs,
    )
    # the max difference between adjacent angles, or its complement, is
    # equivalent to the smallest sector size which encompasses all angles in
    # `data`.
    max_args = np.argmax(angular_diffs, axis=0)
    max_indices = tuple([max_args] + list(np.indices(max_args.shape)))
    # determine the first of the two angles resulting in max difference
    first_bounding_angle = data[max_indices]
    # rotate all angles by `first_bounding_angle` (anticlockwise), and make any
    # resulting negative angles positive. This ensures that the rotated
    # `first_bounding_angle` is 0, and is therefore the smallest angle in the
    # rotated set
    rotated = (data_rolled - first_bounding_angle) % 360
    # determine the second of the two angles, now rotated, resulting in max
    # difference
    second_bound_angle_rotated = rotated[max_indices]
    max_of_rotated = np.max(rotated, axis=0)
    # if `second_bounding_angle_rotated` is the largest element, then
    # sector size is the clockwise span of 0 -> `second_bounding_angle_rotated`,
    # otherwise it's the anticlockwise span
    result = np.where(
        max_of_rotated == second_bound_angle_rotated,
        second_bound_angle_rotated,
        360 - second_bound_angle_rotated,
    )
    # if there are only one or two distinct angles, return the unique difference
    # calculated
    n_unique_angles = (angular_diffs != 0).sum(axis=0)
    result = np.where(n_unique_angles <= 2, np.max(angular_diffs, axis=0), result)
    return result


def flip_flop_index_proportion_exceeding(
    data: xr.DataArray,
    sampling_dim: str,
    thresholds: Iterable,
    *,  # Force keywords arguments to be keyword-only
    is_angular: bool = False,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    **selections: Iterable[int],
):
    """
    Calculates the Flip-Flop Index and returns the proportion exceeding
    (or equal to) each of the supplied `thresholds`.

    Args:
        data: Data from which to draw subsets.
        sampling_dim: The name of the dimension along which to calculate
        thresholds: The proportion of Flip-Flop Index results
            equal to or exceeding these thresholds will be calculated.
            the Flip-Flop Index.
        is_angular: specifies whether `data` is directional data (e.g. wind
            direction).
        reduce_dims: Dimensions to reduce.
        preserve_dims: Dimensions to preserve.
        **selections: Additional keyword arguments specify
            subsets to draw from the dimension `sampling_dim` of the supplied `data`
            before calculation of the Flip_Flop Index. e.g. days123=[1, 2, 3]
    Returns:
        If `selections` are not supplied - An xarray.DataArray with dimensions
        `dims` + 'threshold'. The DataArray is the proportion of the Flip-Flop
        Index calculated by collapsing dimension `sampling_dim` exceeding or
        equal to `thresholds`.

        If `selections` are supplied - An xarray.Dataset with dimensions `dims`
        + 'threshold'. There is a data variable for each keyword in
        `selections`, and corresponds to the Flip-Flop Index proportion
        exceeding for the subset of data specified by the keyword values.

    Examples:
        >>> data = xr.DataArray(
        ...     [[50, 20, 40, 80], [10, 50, 10, 100], [0, 30, 20, 50]],
        ...     dims=['station_number', 'lead_day'],
        ...     coords=[[10001, 10002, 10003], [1, 2, 3, 4]]
        ... )

        >>> flip_flop_index_proportion_exceeding(data, 'lead_day', [20])
        <xarray.DataArray (threshold: 1)>
        array([ 0.33333333])
        Coordinates:
          * threshold  (threshold) int64 20
        Attributes:
            sampling_dim: lead_day

        >>> flip_flop_index_proportion_exceeding(
        ...     data, 'lead_day', [20], days123=[1, 2, 3], all_days=[1, 2, 3, 4]
        ... )
        <xarray.Dataset>
        Dimensions:    (threshold: 1)
        Coordinates:
          * threshold  (threshold) int64 20
        Data variables:
            days123    (threshold) float64 0.6667
            all_days   (threshold) float64 0.3333
        Attributes:
            selections: {{'days123': [1, 2, 3], 'all_days': [1, 2, 3, 4]}}
            sampling_dim: lead_day

    See also:
        `scores.continuous.flip_flop_index`

    """
    if preserve_dims is not None and sampling_dim in list(preserve_dims):
        raise DimensionError(
            f"`sampling_dim`: '{sampling_dim}' must not be in dimensions to preserve "
            f"`preserve_dims`: {list(preserve_dims)}"
        )
    if reduce_dims is not None and sampling_dim in list(reduce_dims):
        raise DimensionError(
            f"`sampling_dim`: '{sampling_dim}' must not be in dimensions to reduce "
            f"`reduce_dims`: {list(reduce_dims)}"
        )
    # calculate the Flip-Flop Index
    flip_flop_data = flip_flop_index(data, sampling_dim, is_angular=is_angular, **selections)
    # calculate the proportion exceeding each threshold
    flip_flop_exceeding = proportion_exceeding(
        flip_flop_data, thresholds, reduce_dims=reduce_dims, preserve_dims=preserve_dims
    )
    # overwrite the attributes
    flip_flop_exceeding.attrs = flip_flop_data.attrs

    return flip_flop_exceeding
