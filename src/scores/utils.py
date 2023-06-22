"""
Contains frequently-used functions of a general nature within scores
"""
import xarray as xr


class DimensionError(Exception):
    """
    Custom exception used when attempting to operate over xarray DataArray or
    Dataset objects that do not have compatible dimensions.
    """


def gather_dimensions(fcst_dims, obs_dims, weights_dims=None, reduce_dims=None, preserve_dims=None):
    """
    Establish which dimensions to reduce according to the optional args
    """

    all_dims = set(fcst_dims).union(set(obs_dims))
    if weights_dims is not None:
        all_dims = all_dims.union(set(weights_dims))

    if preserve_dims is not None and reduce_dims is not None:
        msg = (
            "This method (gather_dimensions) doesn't know how to understand "
            "both preserve_dims and reduce_dims at the same time"
        )
        raise ValueError(msg)

    if preserve_dims is not None:

        if preserve_dims == "all":
            return []

        if isinstance(preserve_dims, str):
            preserve_dims = [preserve_dims]

        reduce_dims = set(all_dims).difference(preserve_dims)

    return reduce_dims


def dims_complement(data, dims=None):
    """
    Returns the complement of data.dims and dims

    Args:
        data: an xarray DataArray or Dataset
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


def check_dims(xr_data, expected_dims, mode=None):
    """
    Checks the dimensions xr_data with expected_dims, according to `mode`.

    Args:
        xr_data (xarray.DataArray or xarray.Dataset): if a Dataset is supplied,
            all of its data variables (DataArray objects) are checked.
        expected_dims (Iterable): an Iterable of dimension names.
        mode (Optional[str]): one of 'equal' (default), 'subset' or 'superset'.

            - If 'equal', checks that the data object has the same dimensions
              as `expected_dims`.
            - If 'subset', checks that the dimensions of the data object is a
              subset of `expected_dims`.
            - If 'superset', checks that the dimensions of the data object is a
              superset of `expected_dims`, (i.e. contains `expected_dims`).
            - If 'proper subset', checks that the dimensions of the data object is a
              subset of `expected_dims`, (i.e. is a subset, but not equal to
              `expected_dims`).
            - If 'proper superset', checks that the dimensions of the data object
              is a proper superset of `expected_dims`, (i.e. contains but is not
              equal to `expected_dims`).
            - If 'disjoint', checks that the dimensions of the data object shares no
              elements with `expected_dims`.

    Returns:
        None

    Raises:
        scores.utils.DimensionError: the dimensions of `xr_data` does
            not pass the check as specified by `mode`.
        TypeError: `xr_data` is not an xarray data object
        ValueError: `expected_dims` contains duplicate values
        ValueError: `expected_dims` cannot be coerced into a set
        ValueError: `mode` is not one of 'equal', 'subset',
            'superset', 'proper subset', 'proper superset', or 'disjoint'

    """

    if isinstance(expected_dims, str):
        raise TypeError(
            f"Supplied dimensions '{expected_dims}' must be an iterable of strings, " "not a string itself."
        )

    try:
        dims_set = set(expected_dims)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert supplied dims {expected_dims} into a set. " "Check debug log for more information."
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
