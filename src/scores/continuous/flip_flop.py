"""
This module contains functions for calculating flip flop indices
"""

from collections.abc import Generator, Iterable, Sequence
from typing import Optional, Union, overload

import numpy as np
import xarray as xr

from scores.functions import angular_difference
from scores.typing import XarrayLike
from scores.utils import DimensionError, check_dims, dims_complement


def _flip_flop_index(data: xr.DataArray, sampling_dim: str, is_angular: bool = False) -> xr.DataArray:
    """
    Calculates the flip-flop index by collapsing the dimension specified by
    `sampling_dim`.

    Args:
        data: Data from which to draw subsets.
        sampling_dim: The name of the dimension along which to calculate
            the flip-flop index.
        is_angular: specifies whether `data` is directional data (e.g. wind
            direction).

    Returns:
        A xarray.DataArray of the flip-flop index with the dimensions of
        `data`, except for the `sampling_dim` dimension which is collapsed.

    See also:
        `scores.continuous.flip_flop.flip_flop_index`
    """
    # check that `sampling_dim` is in `data`.
    check_dims(data, [sampling_dim], mode="superset")

    # the maximum possible number of discrete flipflops
    sequence_length = len(data[sampling_dim])
    max_possible_flip_flop_count = sequence_length - 2

    # calculate the range
    # skip_na=False guarantees that if there is a nan in that row,
    # it will show up as nan in the end
    if is_angular:
        # get complementary dimensions as `encompassing_sector_size` takes
        # dimensions to be preserved, not collapsed
        dims_to_preserve = dims_complement(data, [sampling_dim])
        # get maximum forecast range, if > 180 then clip to 180 as this is the
        # maximum possible angular difference between two forecasts
        enc_size = encompassing_sector_size(data=data, dims=dims_to_preserve)
        range_val = np.clip(enc_size, a_min=None, a_max=180.0)
        flipflop = angular_difference(data.shift({sampling_dim: 1}), data)
    else:
        max_val = data.max(dim=sampling_dim, skipna=False)
        min_val = data.min(dim=sampling_dim, skipna=False)
        range_val = max_val - min_val
        # subtract each consecutive 'row' from eachother
        flipflop = data.shift({sampling_dim: 1}) - data

    # take the absolute value and sum.
    # I don't do skipna=False here because .shift makes a row of nan
    flipflop = abs(flipflop).sum(dim=sampling_dim)
    # adjust based on the range. This is where nan will be introduced.
    flipflop = flipflop - range_val
    # normalise
    return flipflop / max_possible_flip_flop_count


@overload
def flip_flop_index(
    data: xr.DataArray, sampling_dim: str, is_angular: bool = False, **selections: Iterable[int]
) -> xr.Dataset:
    ...


@overload
def flip_flop_index(
    data: xr.DataArray, sampling_dim: str, is_angular: bool = False, **selections: None
) -> xr.DataArray:
    ...


def flip_flop_index(
    data: xr.DataArray, sampling_dim: str, is_angular: bool = False, **selections: Optional[Iterable[int]]
) -> XarrayLike:
    """
    Calculates the Flip-flop Index along the dimensions `sampling_dim`.

    Args:
        data: Data from which to draw subsets.
        sampling_dim: The name of the dimension along which to calculate
            the flip-flop index.
        is_angular: specifies whether `data` is directional data (e.g. wind
            direction).
        **selections: Additional keyword arguments specify
            subsets to draw from the dimension `sampling_dim` of the supplied `data`
            before calculation of the flipflop index. e.g. days123=[1, 2, 3]

    Returns:
        If `selections` are not supplied: An xarray.DataArray, the Flip-flop
        Index by collapsing the dimension `sampling_dim`.

        If `selections` are supplied: An xarray.Dataset. Each data variable
        is a supplied key-word argument, and corresponds to selecting the
        values specified from `sampling_dim` of `data`. The Flip-flop Index
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
        result = xr.Dataset()
        result.attrs["selections"] = selections
        for key, data_subset in iter_selections(data, sampling_dim, **selections):
            result[key] = _flip_flop_index(data_subset, sampling_dim, is_angular=is_angular)
    result.attrs["sampling_dim"] = sampling_dim

    return result


@overload
def iter_selections(
    data: xr.DataArray, sampling_dim: str, **selections: Optional[Iterable[int]]
) -> Generator[tuple[str, xr.DataArray], None, None]:
    ...


@overload
def iter_selections(
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
    check_dims(data, [sampling_dim], "superset")

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


def encompassing_sector_size(data: xr.DataArray, dims: Sequence[str], skipna: bool = False) -> xr.DataArray:
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
    check_dims(data, dims, "proper superset")
    dims_to_collapse = dims_complement(data, dims)
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
    data: np.ndarray, axis_to_collapse: Union[int, tuple[int, ...]] = 0, skipna: bool = False
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
