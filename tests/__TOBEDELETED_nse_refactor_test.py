#!/usr/bin/env nix-shell
#! nix-shell --impure
#! nix-shell -i 'pytest -x -s __TOBEDELETED*'
#! nix-shell -p bash python3
#! nix-shell -p python312Packages.pytest
#! nix-shell -p python312Packages.pytest-cov
#! nix-shell -p python312Packages.numpy
#! nix-shell -p python312Packages.xarray
#! nix-shell -p python312Packages.ipdb

import warnings
import functools

from collections.abc import Hashable, Iterable, Callable
from typing import Any, TypeAlias, TypeGuard, Union, Unpack, cast

import numpy as np
import xarray
import pytest

# Dimension name. `xarray` recommends `str` for dimension names, however, it doesn't impose
# restrictions on generic hashables, so we also need to support hashables.
DimName: TypeAlias = Hashable

# `XarrayLike` data types should be used for all forecast, observed and weights. However, currently
# some are specified as DataArray only
XarrayLike: TypeAlias = Union[xarray.DataArray, xarray.Dataset]


# Generic collection that supports both `DimName`, a single hashable and `FlexibleDimensionTypes`.
# A iterable collection of dimension names. Useful for applying functions that can support both
# single names or collections e.g. `gather_dimensions` which currently does these checks internally.
#
# NOTE: may replace `FlexibleDimensionTypes` down the line - this has more utility functions.
DimNameCollection: TypeAlias = Union[DimName, Iterable[DimName]]


# Type error message for `DimNameCollection`
__unfmt_err_dimnamecln: str = r"""
Invalid type for dimension name collection: {_type}. `DimensionNameCollection` must be either a
`Hashable` (e.g. `str`) or an `Iterable` of `Hashable`s e.g. `list[str]`.
"""
ERROR_DIMNAMECOLLECTION: Callable[[], str] = lambda t: __unfmt_err_dimnamecln.format(_type=t)

# Warning for ambiguous `DimNameCollection`.
__unfmt_warn_dimnamecln: str = r"""
Ambiguous `DimNameCollection`. input: {_type} is `Iterable` AND `Hashable`. `Hashable` takes priority.
If you intended to provide an `Iterable`, consider using a non-hashable collection like a `list`.
"""
WARN_AMBIGUOUS_DIMNAMECOLLECTION: Callable[[], str] = lambda t: __unfmt_warn_dimnamecln.format(_type=t)

# TODO: <PACEHOLDER: insert #issue>: Implement similar functionality to `dim_name_collection`
# for other types, to ensure consistent API for typechecking for internal methods.
#
# NOTE: runtime type checking functions *should* be internal.
#
# NOTE: while the methods below apply to dimension names, they can actually apply to most generic
# iterable collections. For now this is the only one of its kind, so there's no incentive to
# introduce generic type variables yet.
def _check_dim_name_cln(t: Any) -> None:
    """
    Runtime type checking for ``DimNameCollection``.

    Raises:
        :py:class:`TypeError`: if type check fails.
    """
    if not _is_dim_name_cln(t):
        raise TypeError(ERROR_DIMNAMECOLLECTION(t))

def _is_dim_name_cln(t: Any) -> TypeGuard[DimNameCollection]:
    """
    Static type checking for ``DimNameCollection``.

    Returns:
        True if input type is valid at runtime, False otherwise.

    .. note::
        Only returns a boolean flag for static checking, use :py:func:`_check_dim_name_cln` if you
        want to raise a runtime error.
    """
    is_name: Callable[[], bool] = lambda x: isinstance(x, DimName)
    is_iterable: Callable[[], bool] = lambda x: isinstance(x, Iterable)
    are_all_names: Callable[[], bool] = lambda x: all(map(is_name, x))

    # if its iterable and is hashable (but its not a string), then its ambiguous.
    if not isinstance(t, str) and (is_name(t) and is_iterable(t)):
        warnings.warn(UserWarning(WARN_AMBIGUOUS_DIMNAMECOLLECTION(t)))

    # `is_name` takes priority otherwise tuples will never get hashed properly.
    return is_name(t) or (is_iterable(t) and are_all_names(t))


def _dim_name_cln_to_list(dim_cln: DimNameCollection) -> list[DimName]:
    """
    Standardizes ``DimNameCollection`` into a list of dimension names.
    - lifts a single ``DimName`` (e.g. str) to a list (e.g. list[str]).
    - transforms (unpacks) any iterable collection of hashes into lists

    .. note::
        Eager unpacking to list since speed & memory are normally not bottlenecks when it comes to
        dimension names.

    Returns:
        list of dimension names.

    Raises:
        TypeError if input is not a ``DimNameCollection``
    """
    # check input type is conformant
    _check_dim_name_cln(dim_cln)
    ret = None

    if isinstance(dim_cln, DimName):
        # Hashable: takes priority, lift to list
        ret = [dim_cln]
    elif not isinstance(dim_cln, list):
        # Iterable: if its not a list, transform to list.
        ret = list(dim_cln)
    else:
        # Iterable: was list to begin with - assert and pass-through
        assert isinstance(dim_cln, list)
        ret = dim_cln

    # safety: check return type is conformant and cast to expected type
    _check_dim_name_cln(ret)
    cast(ret, list[DimName])

    return ret

# >>> NEEDS TESTING
def check_weights_positive(weights: XarrayLike, context: str):
    """
    This is a semi-strict check that requires weights to be non-negative, and their (vector) norm
    to be non-zero. It alerts the user against unexpected cancellations and sign flips, from
    introduction of negative weights, as this affects how some scores are interpretted.

    Args:
        weights: xarray-like data of weights to check
        context: additional information for the warning message e.g.
            metric info and metric-specific reason for raising this warning.

    .. important::
        - This should not be in the public API/docs.
        - This check should be opt-in - i.e. only use this check if you know that the score cannot
          deal with negative weights.
        - NaN values are excluded in this check.
        - This check covers common issues, but is not a strict check (see notes below).

    .. note::
        Future work: a stricter version of this check could be introduced to support:
        - user-defined normalisation that checks that :math:`\\lVert \\text{weights} \\rVert = K`,
          where, :math:`K = 1` and l1-norm or l2-norm are common choices.
        - ``reduce_dims`` to run the check individually against each dimension being reduced, since
          they are accumulated seperately.
    """
    # ignore nans for this check
    weights_masked = np.ma.array(weights, mask=np.isnan(weights))
    # at least one weight should be strictly positive, rest must be non-negative.
    checks_passed = np.ma.all(weights_masked >= 0) and np.ma.any(weights_masked > 0)
    if not checks_passed:
        raise UserWarning(
            "Negative weights are not supported for this metric. "
            "All weights (excluding NaNs) must be >= 0, with at least one strictly positive weight."
        ).add_note(context)

def test_check_weights_positive():
    """
    TODO: complete docs

    Cases:
    - conformant weights - at least one positive, rest can be >= 0
    - any one weight negative - should raise warning
    - all NaNs - should raise warning
    - all zeros - should raise warning
    - implicitly test "context" by setting appropriate context message for the above cases
    - check various data dimensions 1-D, 2-D, 3-D
    """
    pass

# <<< NEEDS TESTING

# >>> REFACTOR
# -------------------------------------------------------------------------------------------------
# TODO
# - fix tests due to added support for hashables and other refactoring
# - move adhoc tests to tests folder
# - fixup doctest
# -------------------------------------------------------------------------------------------------
def merge_dim_names(*clns: Unpack[DimNameCollection]) -> list[DimName]:
    """
    Merge collections of dimension names. Removes any duplicates and flattens to a single list.
    Takes any number of arguments (variadic).

    A sample usecase would be to gather collections of varying types with data containing dimension
    names before feeding it into :py:func:`gather_dimensions`. This is common in functions that have
    more complicated logic to derive ``reduce_dims`` because they accept (or should be able to
    handle) polymorphic types, since ``xarray`` does it and users of ``xarray`` will expect it.

    .. important::

        While :py:mod:`xarray` recommends the use of :py:type:`str` for dimension names. It doesn't
        actually enforce it and any :py:func:`~typing.Hashable` is allowed.

    .. warning::

        Avoid dimension names that are :py:func:`~typing.Hashable` and :py:func:`~typing.Iterable`.
        :py:func:`~typing.Hashable` takes priority, in case the underlying dataset expects a hash
        that can be iterable. If the user intended to provide a collection of hashes, they are
        advised to use something mutable like a ``list`` instead - since they cannot be hashes.

    Returns:
        A merged list of unique dimension names (:py:func:`~typing.Hashable`).

    Raises:
        :py:class:`TypeError`: when *any* argument provided is *not* :py:type:`DimNameCollection`.
        :py:class:`UserWarning`: when a dimension collection type is ambiguous. See warning above.
    """
    # check all dims
    map(_check_dim_name_cln, clns)

    # typehint makes this too long for lambda.
    def _unpack_to_set(_s: set[DimName], _cln: DimNameCollection) -> set[DimName]:
        # cast to set, for unique insertion
        return _s | set(_dim_name_cln_to_list(_cln))

    dim_set: set[DimName] = functools.reduce(_unpack_to_set, clns, set())

    # cast back to list since it has broader compatibility
    return list(dim_set)

# --- TEST ---
def test__is_dim_name_cln():
    """
    TODO: complete docs
    """
    # list of strings is valid
    assert _is_dim_name_cln(["a", "b", "c"]) == True
    # string is valid
    assert _is_dim_name_cln("potato") == True
    # mixed types are valid since the requirement is "Hashable"
    assert _is_dim_name_cln(["a", "b", 4]) == True
    # tuple is valid but expect warning
    in_ = (1, 2, 3)
    with pytest.warns(UserWarning, match=r"Ambiguous.*{}".format(in_)):
        assert _is_dim_name_cln(in_) == True
    # too much nesting
    assert _is_dim_name_cln(["a", "b", ["a", "b"]]) == False

def test_merge_dim_names():
    def _expect_error(*x: Unpack[DimNameCollection], bad_entry: str=""):
        with pytest.raises(TypeError, match=r"Invalid type.*{}".format(bad_entry)):
            merge_dim_names(*x)

    def _unord_assert(x: list[str], y: list[str]):
        assert len(x) == len(y) and set(x) == set(y)

    # empty
    _unord_assert(merge_dim_names([]), [])
    # single str
    _unord_assert(merge_dim_names("potato"), ["potato"])
    # single list - tapi-o(oo)-ca-(o)
    _unord_assert(merge_dim_names(["tapi", "o", "o", "o", "ca", "o"]), ["tapi", "o", "ca"])
    # make sure no dupes
    _unord_assert(
        merge_dim_names("mushroom", ["badger", "mushroom", "mushroom"], "badger", ["badger", "badger", "mushroom"]),
        ["badger", "mushroom"]
    )
    # kitchen sink
    _unord_assert(merge_dim_names("a", ["a", "b", "c"], ["d", "b"], [], "b"), ["a", "b", "c", "d"])

    # too much depth in collection. Should only accept depth = 0 (str) or depth = 1 (list[str]).
    _expect_error(
        ["this", "is", "okay"],
        "so",
        "is",
        "this",
        ["this should", ["fail"]],
        bad_entry=["this should", ["fail"]]
    )

def test__dim_name_cln_to_list():
    """
    Tests :py:func:`scores.typing._dim_name_cln_to_list`.

    Cases:
    - single Hashable
    - list of Hashables
    - tuple: should raise warning
    - list of tuples: should not raise warning

    Condition:
    - output should be a list of Hashables

    Also partially tests :py:func:`scores.typing._check_dim_name_cln`.
    """

def test__check_dim_name_cln():
    """
    Tests :py:func:`scores.typing._check_dim_name_cln`.

    Cases:
    - input types that throw an error
    - ... a warning (e.g. ambiguous types)
    - ... don't raise anything (correct type)

    Also covers :py:func:`scores.typing._is_dim_name_collection`.
    """
    def __expect_error(_t: DimNameCollection):
        with pytest.raises(TypeError, match=r"Invalid type.*{}".format(_t)):
            _check_dim_name_cln(_t)

    def __expect_no_error_no_warn(_t: DimNameCollection):
        try:
            _check_dim_name_cln(_t)
        except Exception:
            return False
        return True

    def __expect_ambg_warning(_t: DimNameCollection):
        with pytest.warns(UserWarning, match=r"Ambiguous.*{}".format(_t)):
            _check_dim_name_cln(_t)

    # error - wrong type - dicts are not hashable (but their entries are)
    __expect_error([{ "x": 1, "y": 2 }])

    # error - nested types
    __expect_error([["potato", "tapioca"]])
    __expect_error(["a", ["b", "c", ["d"]]])

    # warning - ambiguous type - tuple
    __expect_ambg_warning((1, 2, 3, 4))

    # success - list of integers
    __expect_no_error_no_warn([1, 2, 3, 4])

    # success - mixed hashes is allowed
    __expect_no_error_no_warn(["1", 2, "3", 4])

    # success - set of strings
    __expect_no_error_no_warn(set(["a", "b", "c", "d"]))

    # success - string
    __expect_no_error_no_warn("potato")

    # sucecss - integer
    __expect_no_error_no_warn(42)

# TODO: remove - temporary testing
if False:
    test__dim_name_cln_to_list()
    print("RUN: test__dim_name_cln_to_list")
    test__check_dim_name_cln()
    print("RUN: test__check_dim_name_cln")
    test_merge_dim_names()
    print("RUN: test_merge_dim_names")
    test__is_dim_name_cln()
    print("RUN: test__is_dim_name_cln")
    test__is_dim_name_cln()

# <<< REFACTOR

