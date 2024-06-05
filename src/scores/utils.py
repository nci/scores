"""
Contains frequently-used functions of a general nature within scores
"""

import warnings
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Optional, TypeVar, Union

import numpy as np
import pandas as pd
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
You are requesting to preserve a dimension which does not appear in your data 
(fcst, obs or weights). It is ambiguous how to proceed therefore an exception has been
raised instead.
"""

ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION = """
You are requesting to reduce a dimension which does not appear in your data
(fcst, obs or weights). It is ambiguous how to proceed therefore an exception has been 
raised instead.
"""

ERROR_OVERSPECIFIED_PRESERVE_REDUCE = """
You have specified both preserve_dims and reduce_dims. This method doesn't know how
to properly interpret that, therefore an exception has been raised.
"""


class DimensionError(ValueError):
    """
    Custom exception used when attempting to operate over xarray DataArray or
    Dataset objects that do not have compatible dimensions.
    """


class FieldTypeError(TypeError):
    """
    Custom exception used when incompatible field types are used for a particular
    operation.
    """


class DimensionWarning(UserWarning):
    """
    Custom warning raised when dimensional arguments are ambiguous, but the
    ambiguity is implicitly resolved by the underlying algorithm. The warning
    can be nullified if the user resolves the ambiguity, by explicitly providing
    supporting arguments.

    For example, in Fractions Skill Score (FSS) the user must provide spatial
    dimensions used to compute the score. If the user also attempts to preserve
    these dimensions, e.g. due to the default behaviour of `gather_dimensions`,
    this will cause ambiguity. However, the underlying algorithm itself
    necessitates that the spatial dimensions be reduced and hence takes
    priority. This warning is raised to alert the user of this behaviour, as
    well as actions that can be taken to suppress it.
    """


class CompatibilityError(ImportError):
    """
    Custom exception used when attempting to access advanced functionality that
    require extra/optional dependencies.
    """


T = TypeVar("T")


@dataclass
class BinaryOperator(Generic[T]):
    """
    Generic datatype to represent binary operators. For specific event operators,
    refer to ``scores.categorical.contingency.EventOperator``.

    Note: This operator should not be called directly or on every computation.
        Rather, it is intended to perform validation before passing in the "real"
        operator ``self.op`` via an unwrapping call to ``BinaryOperator.get``

    Bad:

    .. code-block:: python
        x = np.rand((1000, 1000))
        threshold = 0.5

        for it in np.nditer(x):
            # validation check is done every loop - BAD
            numpy_op = NumpyThresholdOperator(np.less).get()
            binary_item = numpy_op(it, threshold)
            do_stuff(binary_item)

    Ok:

    .. code-block:: python
        x = np.rand((1000, 1000))
        threshold = 0.5

        # validation check is done one-time here - GOOD
        numpy_op = NumpyThresholdOperator(np.less).get()

        for it in np.nditer(x):
            binary_item = numpy_op(it, threshold)
            do_stuff(binary_item)  # some elementry processing function

        # EVEN BETTER:
        # basic numpy operators are already vectorized so this will work fine.
        binary_items = numpy_op(x, threshold)
        vec_do_stuff(binary_items) # vectorized version

    Key takeaway - unwrap the operator using ``.get()`` as early as possible
    """

    op: Callable[[T, T], T]
    valid_ops: Dict[str, Callable[[T, T], T]] = field(init=False)

    def __post_init__(self):
        """
        Post initialization checks go here
        """
        self._validate()

    def _validate(self):
        """
        Default operator is valid
        """
        # functions are objects so this should work in theory
        if not self.op in self.valid_ops.values():
            # intentional list comprehension, for display reasons
            raise ValueError(
                "Invalid operator specified. Allowed operators: "
                f"{[k for k in self.valid_ops.keys()]}"  # pylint: disable=unnecessary-comprehension
            )
        return self

    def get(self):
        """
        Return the underlying operator
        """
        return self.op


def left_identity_operator(x, _):
    """
    A binary operator that takes in two inputs but only returns the first.
    """
    return x


@dataclass
class NumpyThresholdOperator(BinaryOperator):
    """
    Generic numpy threshold operator to avoid function call over-head,
    for light-weight comparisons. For specific event operators, refer to
    ``scores.processing.discretise``

    Important: The input field must be the first operand and the threshold
    should be the second operand otherwise some operators may have unintended
    behaviour.
    """

    def __post_init__(self):
        self.valid_ops = {
            "numpy.greater": np.greater,
            "numpy.greater_equal": np.greater_equal,
            "numpy.less": np.less,
            "numpy.less_equal": np.less_equal,
            "scores.utils.left_identity_operator": left_identity_operator,
        }
        super().__post_init__()


def gather_dimensions(  # pylint: disable=too-many-branches
    fcst_dims: Iterable[Hashable],
    obs_dims: Iterable[Hashable],
    *,  # Force keywords arguments to be keyword-only
    weights_dims: Optional[Iterable[Hashable]] = None,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
    score_specific_fcst_dims: Optional[FlexibleDimensionTypes] = None,
) -> set[Hashable]:
    """
    Establish which dimensions to reduce when calculating errors but before taking means.

    Args:
        fcst: Forecast data
        obs: Observation data
        weights: Weights for calculating a weighted mean of scores
        reduce_dims: Dimensions to reduce. Can be "all" to reduce all dimensions.
        preserve_dims: Dimensions to preserve. Can be "all" to preserve all dimensions.
        score_specific_fcst_dims: Dimension(s) in `fcst` that are reduced to calculate individual scores.
            Must not appear as a dimension in `obs`, `weights`, `reduce_dims` or `preserve_dims`.
            e.g. the ensemble member dimension if calculating CRPS for ensembles, or the
            threshold dimension of calculating CRPS for CDFs.

    Returns:
        Set of dimensions over which to take the mean once the checks are passed.

    Raises:
        ValueError: when `preserve_dims and `reduce_dims` are both specified.
        ValueError: when `score_specific_fcst_dims` is not a subset of `fcst.dims`.
        ValueError: when `obs.dims`, `weights.dims`, `reduce_dims` or `preserve_dims`
            contains elements from `score_specific_fcst_dims`.
        ValueError: when `preserve_dims and `reduce_dims` contain elements not among dimensions
            of the data (`fcst`, `obs` or `weights`).

    """

    all_data_dims = set(fcst_dims).union(set(obs_dims))
    if weights_dims is not None:
        all_data_dims = all_data_dims.union(set(weights_dims))

    # Handle error conditions related to specified dimensions
    if preserve_dims is not None and reduce_dims is not None:
        raise ValueError(ERROR_OVERSPECIFIED_PRESERVE_REDUCE)

    # all_scoring_dims is the set of dims remaining after individual scores are computed.
    all_scoring_dims = all_data_dims.copy()

    # Handle error conditions related to specified dimensions
    specified_dims = preserve_dims or reduce_dims

    if specified_dims == "all":
        if "all" in all_data_dims:
            warnings.warn(WARN_ALL_DATA_CONFLICT_MSG, DimensionWarning)
    elif specified_dims is not None:
        if isinstance(specified_dims, str):
            specified_dims = [specified_dims]

    # Raise errors unless score_specific_fcst_dims are in fcst.dims only
    if score_specific_fcst_dims is not None:
        if isinstance(score_specific_fcst_dims, str):
            score_specific_fcst_dims = [score_specific_fcst_dims]
        if not set(score_specific_fcst_dims).issubset(set(fcst_dims)):
            raise ValueError("`score_specific_fcst_dims` must be a subset of `fcst` dimensions")
        if len(set(obs_dims).intersection(set(score_specific_fcst_dims))) > 0:
            raise ValueError("`obs.dims` must not contain any `score_specific_fcst_dims`")
        if weights_dims is not None:
            if len(set(weights_dims).intersection(set(score_specific_fcst_dims))) > 0:
                raise ValueError("`weights.dims` must not contain any `score_specific_fcst_dims`")
        if specified_dims is not None and specified_dims != "all":
            if len(set(specified_dims).intersection(set(score_specific_fcst_dims))) > 0:
                raise ValueError("`reduce_dims` and `preserve_dims` must not contain any `score_specific_fcst_dims`")

        # Finally, remove score_specific_fcst_dims from all_scoring_dims
        all_scoring_dims = all_data_dims.difference(set(score_specific_fcst_dims))

    if specified_dims is not None and specified_dims != "all":
        if not set(specified_dims).issubset(all_data_dims):
            if preserve_dims is not None:
                raise ValueError(ERROR_SPECIFIED_NONPRESENT_PRESERVE_DIMENSION)
            raise ValueError(ERROR_SPECIFIED_NONPRESENT_REDUCE_DIMENSION)

    # Turn preserve dims into reduce dims, if needed
    if preserve_dims is not None:
        if preserve_dims == "all":
            return set([])

        if isinstance(preserve_dims, str):
            preserve_dims = [preserve_dims]

        reduce_dims = set(all_scoring_dims).difference(preserve_dims)

    # Handle reduction of all dimensions by string "all""
    elif reduce_dims == "all":
        reduce_dims = set(all_scoring_dims)

    # Handle reduction of a single dim by string name
    elif isinstance(reduce_dims, str):
        reduce_dims = set([reduce_dims])

    # Handle when reduce_dims and preserve_dims are both None
    elif reduce_dims is None and preserve_dims is None:
        reduce_dims = set(all_scoring_dims)

    # Turn into a set if needed
    assert reduce_dims is not None  # nosec - this is just to modify type hinting
    reduce_dims = set(reduce_dims)

    # Reduce by list is the default so no handling needed
    return reduce_dims


def dims_complement(data, *, dims=None) -> list[str]:
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


def check_dims(xr_data: XarrayLike, expected_dims: Iterable[Hashable], *, mode: Optional[str] = None):
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

    if len(list(dims_set)) != len(list(expected_dims)):
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
            f"dimensions {sorted([str(d) for d in dims_set])}."
        )

    if isinstance(xr_data, xr.Dataset):
        # every data variable must pass the dims check too!
        for data_var in xr_data.data_vars:
            if not check_fn(xr_data[data_var], dims_set):
                raise DimensionError(
                    f"Dimensions {list(xr_data[data_var].dims)} of data variable "
                    f"'{data_var}' are not {mode} to the dimensions {sorted([str(d) for d in dims_set])}"
                )


def tmp_coord_name(xr_data: xr.DataArray, *, count=1) -> Union[str, list[str]]:
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
    result = "".join(all_names)  # type: ignore

    if count == 1:
        return result

    results = [str(i) + result for i in range(count)]
    return results


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
