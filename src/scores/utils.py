"""
Contains frequently-used functions of a general nature within scores
"""
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import Optional, Union

import xarray as xr

from scores.typing import FlexibleDimensionTypes, XarrayLike

WARN_ALL_DATA_CONFLICT_MSG = """
You are requesting to reduce or preserve every dimension by specifying the string 'all'.
In this case, 'all' is also a named dimension in your data, leading to an ambiguity.
In order to reduce or preserve the named data dimension, specify ['all'] as a list item
rather than relying on string interpretation. The program will continue to interpret the
string as an instruction to reduce or preserve every dimension.
"""

ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION = """
You are requesting to preserve a dimension which does not appear in your data (fcst or obs).
It is ambiguous how to proceed therefore an exception has been raised instead.
"""

ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION2 = """
You are requesting to preserve a dimension which does not appear in your data 
(fcst, obs or weights). It is ambiguous how to proceed therefore an exception has been
raised instead.
"""

ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION = """
You are requesting to reduce a dimension which does not appear in your data (fcst or obs).
It is ambiguous how to proceed therefore an exception has been raised instead.
"""

ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION2 = """
You are requesting to reduce a dimension which does not appear in your data
(fcst, obs or weights). It is ambiguous how to proceed therefore an exception has been 
raised instead.
"""

ERROR_OVERSPECIFIED_PRESERVE_REDUCE = """
You have specified both preserve_dims and reduce_dims. This method doesn't know how
to properly interpret that, therefore an exception has been raised.
"""


class DimensionError(Exception):
    """
    Custom exception used when attempting to operate over xarray DataArray or
    Dataset objects that do not have compatible dimensions.
    """


def gather_dimensions(  # pylint: disable=too-many-branches
    fcst_dims: Iterable[Hashable],
    obs_dims: Iterable[Hashable],
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
) -> set[Hashable]:
    """
    Establish which dimensions to reduce when calculating errors but before taking means.

    Note: `scores.utils.gather_dimensions` and `scores.utils.gather_dimensions2` will be
    integrated at some point in the future. `scores.utils.gather_dimensions2` offers
    more comprehensive and less restrictive dimension checking and should be preferred in
    the meantime. See `scores.probability.crps_cdf` for an example of
    `scores.utils.gather_dimensions2` usage.

    Args:
        fcst_dims: Forecast dimensions inputs
        obs_dims: Observation dimensions inputs.
        reduce_dims: Dimensions to reduce.
        preserve_dims: Dimensions to preserve.

    Returns:
        Dimensions based on optional args.
    Raises:
        ValueError: When `preserve_dims and `reduce_dims` are both specified.

    See also:
        `scores.utils.gather_dimensions2`
    """

    all_dims = set(fcst_dims).union(set(obs_dims))

    # Handle error conditions related to specified dimensions
    if preserve_dims is not None and reduce_dims is not None:
        raise ValueError(ERROR_OVERSPECIFIED_PRESERVE_REDUCE)

    # Handle error conditions related to specified dimensions
    specified = preserve_dims or reduce_dims
    if specified == "all":
        if "all" in all_dims:
            warnings.warn(WARN_ALL_DATA_CONFLICT_MSG)
    elif specified is not None:
        if isinstance(specified, str):
            specified = [specified]

        if not set(specified).issubset(all_dims):
            if preserve_dims is not None:
                raise ValueError(ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION)
            raise ValueError(ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION)

    # Handle preserve_dims case
    if preserve_dims is not None:
        if preserve_dims == "all":
            return set([])

        if isinstance(preserve_dims, str):
            preserve_dims = [preserve_dims]

        reduce_dims = set(all_dims).difference(preserve_dims)

    # Handle reduce all
    elif reduce_dims == "all":
        reduce_dims = set(all_dims)

    # Handle is reduce_dims and preserve_dims are both None
    if reduce_dims is None and preserve_dims is None:
        reduce_dims = set(all_dims)

    # Handle reduce by string
    elif isinstance(reduce_dims, str):
        reduce_dims = set([reduce_dims])

    # Turn into a set if needed
    assert reduce_dims is not None
    reduce_dims = set(reduce_dims)

    # Reduce by list is the default so no handling needed
    return reduce_dims


def gather_dimensions2(  # pylint: disable=too-many-branches
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    weights: xr.DataArray = None,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    special_fcst_dims: FlexibleDimensionTypes = None,
) -> set[Hashable]:
    """
    Performs standard dimensions checks for inputs of functions that calculate (mean) scores.
    Returns a set of the dimensions to reduce.

    Note: `scores.utils.gather_dimensions` and `scores.utils.gather_dimensions2` will be
    integrated at some point in the future. `scores.utils.gather_dimensions2` offers
    more comprehensive and less restrictive dimension checking and should be preferred in
    the meantime. See `scores.probability.crps_cdf` for an example of
    `scores.utils.gather_dimensions2` usage.

    Args:
        fcst: Forecast data
        obs: Observation data
        weights: Weights for calculating a weighted mean of scores
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        special_fcst_dims: Dimension(s) in `fcst` that are reduced to calculate individual scores.
            Must not appear as a dimension in `obs`, `weights`, `reduce_dims` or `preserve_dims`.
            e.g. the ensemble member dimension if calculating CRPS for ensembles, or the
            threshold dimension of calculating CRPS for CDFs.

    Returns:
        Set of dimensions over which to take the mean once the checks are passed.

    Raises:
        ValueError: when `preserve_dims and `reduce_dims` are both specified.
        ValueError: when `special_fcst_dims` is not a subset of `fcst.dims`.
        ValueError: when `obs.dims`, `weights.dims`, `reduce_dims` or `preserve_dims`
            contains elements from `special_fcst_dims`.
        ValueError: when `preserve_dims and `reduce_dims` contain elements not among dimensions
            of the data (`fcst`, `obs` or `weights`).

    See also:
        `scores.utils.gather_dimensions`
    """
    all_data_dims = set(fcst.dims).union(set(obs.dims))
    if weights is not None:
        all_data_dims = all_data_dims.union(set(weights.dims))

    # all_scoring_dims is the set of dims remaining after individual scores are computed.
    all_scoring_dims = all_data_dims.copy()

    # Handle error conditions related to specified dimensions
    if preserve_dims is not None and reduce_dims is not None:
        raise ValueError(ERROR_OVERSPECIFIED_PRESERVE_REDUCE)

    specified_dims = preserve_dims or reduce_dims

    if specified_dims == "all":
        if "all" in all_data_dims:
            warnings.warn(WARN_ALL_DATA_CONFLICT_MSG)
    elif specified_dims is not None:
        if isinstance(specified_dims, str):
            specified_dims = [specified_dims]

    # check that special_fcst_dims are in fcst.dims only
    if special_fcst_dims is not None:
        if isinstance(special_fcst_dims, str):
            special_fcst_dims = [special_fcst_dims]
        if not set(special_fcst_dims).issubset(set(fcst.dims)):
            raise ValueError("`special_fcst_dims` must be a subset of `fcst` dimensions")
        if len(set(obs.dims).intersection(set(special_fcst_dims))) > 0:
            raise ValueError("`obs.dims` must not contain any `special_fcst_dims`")
        if weights is not None:
            if len(set(weights.dims).intersection(set(special_fcst_dims))) > 0:
                raise ValueError("`weights.dims` must not contain any `special_fcst_dims`")
        if specified_dims is not None and specified_dims != "all":
            if len(set(specified_dims).intersection(set(special_fcst_dims))) > 0:
                raise ValueError("`reduce_dims` and `preserve_dims` must not contain any `special_fcst_dims`")
        # remove special_fcst_dims from all_scoring_dims
        all_scoring_dims = all_scoring_dims.difference(set(special_fcst_dims))

    if specified_dims is not None and specified_dims != "all":
        if not set(specified_dims).issubset(all_scoring_dims):
            if preserve_dims is not None:
                raise ValueError(ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION2)
            raise ValueError(ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION2)

    # all errors have been captured, so now return list of dims to reduce
    if specified_dims is None:
        return all_scoring_dims
    if reduce_dims is not None:
        if reduce_dims == "all":
            return all_scoring_dims
        return set(specified_dims)
    if preserve_dims == "all":
        return set([])
    return all_scoring_dims.difference(set(specified_dims))


def dims_complement(data, dims=None) -> list[str]:
    """Returns the complement of data.dims and dims

    Args:
        data: Input xarray object
        dims: an Iterable of strings corresponding to dimension names

    Returns:
        A sorted list of dimension names, the complement of data.dims and dims
    """

    if dims is None:
        dims = []

    # check that dims is in data.dims, and that dims is a of a valid form
    check_dims(data, dims, mode="superset")

    complement = set(data.dims) - set(dims)
    return sorted(list(complement))


def check_dims(xr_data: XarrayLike, expected_dims: Sequence[str], mode: Optional[str] = None):
    """
    Checks the dimensions xr_data with expected_dims, according to `mode`.

    Args:
        xr_data: if a Dataset is supplied,
            all of its data variables (DataArray objects) are checked.
        expected_dims: an Iterable of dimension names.
        mode: one of 'equal' (default), 'subset' or 'superset'.
            If 'equal', checks that the data object has the same dimensions
            as `expected_dims`.
            If 'subset', checks that the dimensions of the data object is a
            subset of `expected_dims`.
            If 'superset', checks that the dimensions of the data object is a
            superset of `expected_dims`, (i.e. contains `expected_dims`).
            If 'proper subset', checks that the dimensions of the data object is a
            subset of `expected_dims`, (i.e. is a subset, but not equal to
            `expected_dims`).
            If 'proper superset', checks that the dimensions of the data object
            is a proper superset of `expected_dims`, (i.e. contains but is not
            equal to `expected_dims`).
            If 'disjoint', checks that the dimensions of the data object shares no
            elements with `expected_dims`.

    Raises:
        scores.utils.DimensionError: the dimensions of `xr_data` does
            not pass the check as specified by `mode`.
        TypeError: `xr_data` is not an xarray data object.
        ValueError: `expected_dims` contains duplicate values.
        ValueError: `expected_dims` cannot be coerced into a set.
        ValueError: `mode` is not one of 'equal', 'subset', 'superset',
            'proper subset', 'proper superset', or 'disjoint'
    """

    if isinstance(expected_dims, str):
        raise TypeError(f"Supplied dimensions '{expected_dims}' must be an iterable of strings, not a string itself.")

    try:
        dims_set = set(expected_dims)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert supplied dims {expected_dims} into a set. Check debug log for more information."
        ) from exc

    if len(dims_set) != len(expected_dims):
        raise ValueError(f"Supplied dimensions {expected_dims} contains duplicate values.")

    if not hasattr(xr_data, "dims"):
        raise DimensionError("Supplied object has no dimensions")

    # internal functions to check a data array
    check_modes = {
        "equal": lambda da, dims_set: set(da.dims) == dims_set,
        "subset": lambda da, dims_set: set(da.dims) <= dims_set,
        "superset": lambda da, dims_set: set(da.dims) >= dims_set,
        "proper subset": lambda da, dims_set: set(da.dims) < dims_set,
        "proper superset": lambda da, dims_set: set(da.dims) > dims_set,
        "disjoint": lambda da, dims_set: len(set(da.dims) & dims_set) == 0,
    }

    if mode is None:
        mode = "equal"
    if mode not in check_modes:
        raise ValueError(f"No such mode {mode}, mode must be one of: {list(check_modes.keys())}")

    check_fn = check_modes[mode]

    # check the dims
    if not check_fn(xr_data, dims_set):
        raise DimensionError(
            f"Dimensions {list(xr_data.dims)} of data object are not {mode} to the "
            f"dimensions {sorted(list(dims_set))}."
        )

    if isinstance(xr_data, xr.Dataset):
        # every data variable must pass the dims check too!
        for data_var in xr_data.data_vars:
            if not check_fn(xr_data[data_var], dims_set):
                raise DimensionError(
                    f"Dimensions {list(xr_data[data_var].dims)} of data variable "
                    f"'{data_var}' are not {mode} to the dimensions {sorted(dims_set)}"
                )


def tmp_coord_name(xr_data: xr.DataArray, count=1) -> Union[str, list[str]]:
    """
    Generates temporary coordinate names that are not among the coordinate or dimension
    names of `xr_data`.

    Args:
        xr_data: Input xarray data array
        count: Number of unique names to generate

    Returns:
        If count = 1, a string which is the concatenation of 'new' with all coordinate and
        dimension names in the input array. (this is the default)
        If count > 1, a list of such strings, each unique from one another
    """
    all_names = ["new"] + list(xr_data.dims) + list(xr_data.coords)
    result = "".join(all_names)

    if count == 1:
        return result

    results = [str(i) + result for i in range(count)]
    return results
