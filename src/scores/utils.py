"""
Contains frequently-used functions of a general nature within scores
"""

import copy
import functools
import warnings
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from scores.typing import (
    FlexibleDimensionTypes,
    XarrayLike,
    LiftedDataset,
    XarrayTypeMarker,
    assert_xarraylike,
    assert_lifteddataset,
)

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

ERROR_INVALID_WEIGHTS = """
You have specified invalid weights. The weights (excluding NaNs) must be >= 0, with at least one
strictly positive weight.
"""

WARN_INVALID_WEIGHTS = f"""
{ERROR_INVALID_WEIGHTS}
NOTE: The score in which this check has been used has allowed this to be a warning instead of a
      error. The user is responsible for checking the integrity of the inputs and outputs.
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


class LiftedDatasetUtils:
    """
    namespace class containing utility methods for LiftedDataset.
    """

    ERROR_INVALID_LIFTED_DATASET_TYPE: str = """
    Input type is not a `LiftedDataset`. Did you attempt to pass in `xr.Dataset` or xr.DataArray`
    instead of `scores.typing.LiftedDataset`?
    """

    ERROR_INVALID_LIFTFUNC_RETTYPE: str = """
    Functions lifted by `lift_fn_ret` must return either a `xr.DataArray` or a `xr.Dataset`
    (preferrable).
    """

    ERROR_INCONSISTENT_TYPES: str = """
    The provided xarray data inputs are not of same type, they must ALL EXCLUSIVELY be ONLY
    `xr.Dataset`, otherwise, ONLY `xr.DataArray`.
    """

    WARN_EMPTYARGS_FOR_ALLSAMETYPECHECK: str = """
    No args provided for XarrayLike `all_same_type` checks. If you are using `lift_fn*` make sure the
    function you are wrapping actually uses `xr.DataArray` or `xr.Dataset`, otherwise it maybe
    clearer to just lift the output directly.
    """

    @staticmethod
    def lift_fn(fn: Callable) -> Callable:
        """
        Wrapper to maintain backward compatibility with legacy functions that use ``XarrayLike``
        arguments instead of ``LiftedDataset``.

        .. important::

            For INTERNAL use only - NOT for public API.

        .. caution::

            This wrapper should be used sparingly - it should be deprecated if the code is
            refactored to natively support ``LiftedDataset``. The end goal is to simplify
            compatibility between xarray (and potentially other) data structures.

        .. tip::

            This wrapper is preferrable for maintaining compatiblity with functions that query
            datasets to e.g. perform checks, but NOT directly operate on the underlying numeric
            values.

            For mathematical computations (which generally take numeric data and return numeric
            data), it's generally preferrable to use :py:meth:`lift_fn_ret` as it is structure
            preserving.

        .. see-also::

            :py:meth:`LiftedDatasetUtils.lift_fn_ret`
        """

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            # shallow copy is okay, since no data is actually being changed, just wrapped
            args_new = list(copy.copy(args))
            kwargs_new = copy.copy(kwargs)
            is_compat = lambda _lds: isinstance(_lds, LiftedDataset) and _lds.is_valid()  # pylint: disable=C3001
            # fixup args -> replace LiftedDataset with LiftedDataset.inner_ref()
            for i, v in enumerate(args):
                if is_compat(v):
                    args_new[i] = args[i].inner_ref()
            # fixup kwargs -> replace LiftedDataset with LiftedDataset.inner_ref()
            for k, v in kwargs.items():
                if is_compat(v):
                    kwargs_new[k] = v.inner_ref()
            return fn(*args_new, **kwargs_new)

        return _wrapper

    @classmethod
    def lift_fn_ret(cls, fn: Callable[..., XarrayLike]) -> Callable[..., LiftedDataset]:
        """
        Like ``lift_fn`` but also lifts the return type to a ``LiftedDataset``. Uses
        ``lift_fn`` under the hood.

        .. important::

            For INTERNAL use only - NOT for public API. Read the docstring for ``lift_fn``.

        .. see-also::

            :py:meth:`LiftedDatasetUtils.lift_fn`
        """
        fn_lifted = LiftedDatasetUtils.lift_fn(fn)

        @functools.wraps(fn_lifted)
        def _wrapper(*args, **kwargs) -> LiftedDataset:
            # --- check for type consistency ---
            # collect all args that are LiftedDatasets and assert that they are the same type
            lifted_ds_list = [v for v in args if isinstance(v, LiftedDataset)]
            lifted_ds_list += [v for v in kwargs.values() if isinstance(v, LiftedDataset)]
            xr_type_marker: XarrayTypeMarker = LiftedDatasetUtils.all_same_type(*lifted_ds_list)
            # invalid types if the type marker and return type don't match
            # --- run the actual function ---
            # run the actual function and store its result (reference)
            # NOTE: ``fn_lifted`` does not consume any input datasets it calls ``inner_ref`` rather
            # than ``raw``, since the inputs may need to be re-used down the track
            ret_xr: XarrayLike = fn_lifted(*args, **kwargs)
            # --- more checks ---
            if not isinstance(ret_xr, xr.Dataset) or not isinstance(ret_xr, xr.DataArray):
                raise TypeError(cls.ERROR_INVALID_LIFTFUNC_RETTYPE)
            if isinstance(ret_xr, xr.DataArray) and xr_type_marker != XarrayTypeMarker.DATAARRAY:
                raise TypeError(cls.ERROR_INVALID_LIFTFUNC_RETTYPE)
            if isinstance(ret_xr, xr.Dataset) and xr_type_marker != XarrayTypeMarker.DATASET:
                raise TypeError(cls.ERROR_INVALID_LIFTFUNC_RETTYPE)
            # --- return ---
            # if everything is okay, return the (re)lifted dataset
            return LiftedDataset(ret_xr)

        return _wrapper

    @classmethod
    def all_same_type(cls, *lds) -> XarrayTypeMarker:
        """
        Checks if the internal data types for the input :py:class:`LiftedDataset` (``lds``) have the
        same type marker (see: :py:class:`XarrayTypeMarker`).

        .. important::

            This is an internal function - not for public API.

        .. note::
            In particular, any errors thrown by this function needs to be handled by the caller. As the
            errors are mainly aimed for development and testing. Ideally, they should not be raised in
            runtime.

            However, if there are "exceptions" that need to propagate to the user, the user will have to
            handle and re-raise any errors appropriately.

            see :py:meth:`~scores.continuous.nse_impl.NseUtils.get_xr_type_marker` for an example.

        Args:
            *lds: Variadic args of type :py:class:`LiftedDataset`

        Returns:
            xarray type marker if the inputs are a subset of ``XarrayLike`` and ALL of same type.

        Raises:
            TypeError: If types are not consistent or not valid - development only
            AssertionError: For internal checks - development only
        """
        ret_marker: XarrayTypeMarker = XarrayTypeMarker.INVALID
        # need at least one argument to check - otherwise warn and exit since there's nothing to
        # compute
        if len(lds) == 0:
            warnings.warn(cls.WARN_EMPTYARGS_FOR_ALLSAMETYPECHECK, UserWarning)
            return XarrayTypeMarker.INVALID
        # define error messages - not global, as this is a internal function and these errors are more
        # relevant to a developer and in unittests.
        # check that all input types are lifted datasets
        for d in lds:
            if not isinstance(d, LiftedDataset):
                raise TypeError(cls.ERROR_INVALID_LIFTED_DATASET_TYPE)
        # do check: homogeneous xr_data types
        all_ds = all(d.xr_type_marker == XarrayTypeMarker.DATASET for d in lds)
        all_da = all(d.xr_type_marker == XarrayTypeMarker.DATAARRAY for d in lds)
        # both cannot be False (mixed types), and both cannot be True (impossible scenario)
        if all_ds == all_da:
            raise TypeError(cls.ERROR_INCONSISTENT_TYPES)
        # return marker type for dataset
        if all_ds:
            ret_marker = XarrayTypeMarker.DATASET
        # return marker type for data array
        elif all_da:
            ret_marker = XarrayTypeMarker.DATAARRAY
        # saftey check: would have raised TypeError earlier - but may not be obvious to pylint
        assert ret_marker != XarrayTypeMarker.INVALID
        return ret_marker


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


def check_weights_positive(weights: XarrayLike | None, *, raise_error=True):
    """
    This is a check that requires weights to be non-negative (NaN values are excluded from the check
    since they are used as masks). At least one of the weights must be strictly positive.

    .. note::

        It has optional support to raise a warning instead (by setting ``raise_error=False``) for
        specialized functions that may need to support negative weights (though this is not
        recommended).

    Args:
        weights: weights to check
        raise_error: raise an error, instead of warning (default: ``True``)

    .. see-also::

        This function covers most bases, regardless, for potential improvements see:
            - `#828 <GITHUB828_>`_.
            - `#829 <GITHUB829_>`_.

    Raises:
        UserWarning: if ``raise_error=False`` and weights are invalid
        ValueError: if ``raise_error=True`` and weights are invalid

    .. _GITHUB828: https://github.com/nci/scores/issues/828
    .. _GITHUB829: https://github.com/nci/scores/issues/829
    """
    # nothing to check
    if weights is None:
        return
    # safety: typehint
    assert typing.is_xrarraylike(weights)
    # ignore nans as they may be used as natural exclusion masks in weighted calculations.
    weights_masked = np.ma.array(weights, mask=np.isnan(weights))
    # however, still check that we have at least one proper number.
    checks_passed = np.any(~np.isnan(weights))
    # the rest (non-NaN) should all be non-negative ...
    checks_passed = checks_passed and np.ma.all(weights_masked >= 0)
    # ... and at least one number must be strictly positive.
    checks_passed = checks_passed and np.ma.any(weights_masked > 0)

    if not checks_passed:
        if raise_error:
            raise ValueError(ERROR_INVALID_WEIGHTS)
        else:
            warnings.warn(WARN_INVALID_WEIGHTS, UserWarning)
