# --- FIXME: move parts of this file to appropriate places ---

import warnings
import functools

from collections.abc import Hashable, Iterable, Callable
from typing import Any, TypeAlias, TypeGuard, Union, Unpack, cast

import re
import numpy as np
import xarray
import pytest

# -------------------------------------------------------------------------------------------------
# --- FIXME: refactor into typing ---
# -------------------------------------------------------------------------------------------------
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
# Do not use this directly - use `_str_error_dimnamecollection_wrongtype` instead
ERROR_DIMNAMECOLLECTION_WRONGTYPE: str = r"""
Invalid type for dimension name collection: {_type}. `DimensionNameCollection` must be either a
`Hashable` (e.g. `str`) or an `Iterable` of `Hashable`s e.g. `list[str]`.
"""


def _str_error_dimnamecollection_wrongtype(input_type: str):
    return WARN_AMBIGUOUS_DIMNAMECOLLECTION.format(_type=input_type)


# Warning for ambiguous `DimNameCollection`.
# Do not use this directly - use `_str_warn_dimnamecollection_ambiguoustype` instead
WARN_DIMNAMECOLLECTION_AMBIGUOUSTYPE: str = r"""
Ambiguous `DimNameCollection`. input: {_type} is `Iterable` AND `Hashable`. `Hashable` takes priority.
If you intended to provide an `Iterable`, consider using a non-hashable collection like a `list`.
"""


def _str_warn_ambiguous_dimnamecollection(input_type: str):
    return WARN_DIMNAMECOLLECTION_AMBIGUOUSTYPE.format(_type=input_type)


# TODO: <PACEHOLDER: insert #issue>: Implement similar functionality to `dim_name_collection`
# for other types, to ensure consistent API for typechecking for internal methods.
#
# NOTE: runtime type checking functions *should* be internal.
#
# NOTE: while the methods below apply to dimension names, they can actually apply to most generic
# iterable collections. For now this is the only one of its kind, so there's no incentive to
# introduce generic type variables yet.
def check_dimnamecollection(t: Any) -> None:
    """
    Runtime type checking for ``DimNameCollection``.

    Raises:
        :py:class:`TypeError`: if type check fails.
    """
    if not is_dimnamecollection(t):
        msg = _str_error_dimnamecollection_wrongtype(t)
        raise TypeError(msg)


def is_dimnamecollection(t: Any) -> TypeGuard[DimNameCollection]:
    """
    Static type checking for ``DimNameCollection``.

    Returns:
        True if input type is valid at runtime, False otherwise.

    .. note::
        Only returns a boolean flag for static checking, use :py:func:`check_dimnamecollection` if you
        want to raise a runtime error.
    """
    is_name: Callable[..., bool] = lambda x: isinstance(x, DimName)
    is_iterable: Callable[..., bool] = lambda x: isinstance(x, Iterable)
    are_all_names: Callable[..., bool] = lambda x: all(map(is_name, x))

    # if its iterable and is hashable (but its not a string), then its ambiguous.
    if not isinstance(t, str) and (is_name(t) and is_iterable(t)):
        msg = _str_warn_ambiguous_dimnamecollection(t)
        warnings.warn(msg, UserWarning)

    # `is_name` takes priority otherwise tuples will never get hashed properly.
    return is_name(t) or (is_iterable(t) and are_all_names(t))


def dimnamecollection_to_list(dim_cln: DimNameCollection) -> list[DimName]:
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
    check_dimnamecollection(dim_cln)
    ret: list[DimName] | None = None

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
    check_dimnamecollection(ret)
    cast(list[DimName], ret)

    return ret


# -------------------------------------------------------------------------------------------------
# --- FIXME: refactor into utils ---
# -------------------------------------------------------------------------------------------------

def dimnames_to_list(cln: DimNameCollection) -> list[DimName]:
    """
    Similar to merge_dimnames, but for a single collection. Essentially removes duplicates and
    standardizes the collection to a list of dimension names.
    """
    return merge_dimnames(cln)


def merge_dimnames(*clns: Unpack[tuple[DimNameCollection]]) -> list[DimName]:
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
    map(check_dimnamecollection, clns)

    # typehint makes this too long for lambda.
    def _unpack_to_set(_s: set[DimName], _cln: DimNameCollection) -> set[DimName]:
        # cast to set, for unique insertion
        return _s | set(dimnamecollection_to_list(_cln))

    dim_set: set[DimName] = functools.reduce(_unpack_to_set, clns, set())

    # cast back to list since it has broader compatibility
    return list(dim_set)


def check_weights_positive(weights: XarrayLike, *, context: str):
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
    # ignore nans as they may be used as natural exclusion masks in weighted calculations.
    weights_masked = np.ma.array(weights, mask=np.isnan(weights))
    # however, still check that we have at least one proper number.
    checks_passed = np.any(~np.isnan(weights))
    # the rest (non-NaN) should all be non-negative ...
    checks_passed = checks_passed and np.ma.all(weights_masked >= 0)
    # ... and at least one number must be strictly positive.
    checks_passed = checks_passed and np.ma.any(weights_masked > 0)

    if not checks_passed:
        _warning = UserWarning(
            "Negative weights are not supported for this metric. "
            "All weights (excluding NaNs) must be >= 0, with at least one strictly positive weight."
            f"Context = '{context}'"
        )
        warnings.warn(_warning)


# -------------------------------------------------------------------------------------------------
# --- FIXME: refactor into tests ---
# -------------------------------------------------------------------------------------------------


# TODO: parameterize
def test_check_weights_positive():
    """
    Tests :py:func:`scores.utils.check_weights_positive`.

    Cases:
    - conformant weights - at least one positive, rest can be >= 0
    - any one weight negative - should raise warning
    - all NaNs - should raise warning
    - all zeros - should raise warning
    - implicitly test "context" by setting appropriate context message for the above cases
    """

    def _check_weights_positive_assert_no_warning(_w):
        threw_warning = False
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            try:
                check_weights_positive(_w, context="THIS SHOULD NOT BE RAISED!")
            except Warning:
                threw_warning = True
        assert not threw_warning

    def __expect_warning(_t: DimNameCollection, _c: str):
        with pytest.warns(UserWarning, match=r"Negative weights.*{}".format(_c)):
            check_weights_positive(_t, context=_c)

    _check_weights_positive_assert_no_warning(np.array([1.0, 0.0, 0.1]))
    # catching np.inf is not the responsibility of this function, ...
    _check_weights_positive_assert_no_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, np.inf]]))
    # ... neither is np.nan - which is ignored.
    _check_weights_positive_assert_no_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, np.nan]]))

    # --- the rest of these should throw warnings ---

    # However inputs with all nans should warn, ...
    __expect_warning(np.array([[np.nan, np.nan], [np.nan, np.nan]]), "ba-nananana!")
    # ... and negative infinity is not allowed.
    __expect_warning(np.array([[1.0, 0.0, 0.1], [0.0, 0.0, -np.inf]]), "negative infinity!")
    # In fact any negative number is should fail the check, ...
    __expect_warning(np.array([-1e-7, 0.0, 0.0, 1.0]), "negative any")
    # ... even if all of them are negative, and theoretically can be flipped to be conformant,
    # it is not the responsibility of this function to do so.
    __expect_warning(np.array([-1.0, -0.1]), "negative all")
    # also the above would have implicitly tested that "context" works as expected.


# TODO: parameterize
def test_is_dimnamecollection():
    """
    TODO: complete docs
    """
    # list of strings is valid
    assert is_dimnamecollection(["a", "b", "c"]) == True
    # string is valid
    assert is_dimnamecollection("potato") == True
    # mixed types are valid since the requirement is "Hashable"
    assert is_dimnamecollection(["a", "b", 4]) == True
    # tuple is valid but expect warning
    in_ = (1, 2, 3)
    with pytest.warns(UserWarning, match=r"Ambiguous.*{}".format(in_)):
        assert is_dimnamecollection(in_) == True
    # too much nesting
    assert is_dimnamecollection(["a", "b", ["a", "b"]]) == False


# TODO: parameterize
def test_merge_dimnames():
    """
    TODO: docstring
    """

    def _expect_error(*x: Unpack[tuple[DimNameCollection]], bad_entry: str = ""):
        with pytest.raises(TypeError, match=r"Invalid type.*{}".format(bad_entry)):
            merge_dimnames(*x)

    def _unord_assert(x: list[str], y: list[str]):
        assert len(x) == len(y) and set(x) == set(y)

    # empty
    _unord_assert(merge_dimnames([]), [])
    # single str
    _unord_assert(merge_dimnames("potato"), ["potato"])
    # single list - tapi-o(oo)-ca-(o)
    _unord_assert(merge_dimnames(["tapi", "o", "o", "o", "ca", "o"]), ["tapi", "o", "ca"])
    # make sure no dupes
    _unord_assert(
        merge_dimnames("mushroom", ["badger", "mushroom", "mushroom"], "badger", ["badger", "badger", "mushroom"]),
        ["badger", "mushroom"],
    )
    # kitchen sink
    _unord_assert(merge_dimnames("a", ["a", "b", "c"], ["d", "b"], [], "b"), ["a", "b", "c", "d"])

    # too much depth in collection. Should only accept depth = 0 (str) or depth = 1 (list[str]).
    _expect_error(
        ["this", "is", "okay"], "so", "is", "this", ["this should", ["fail"]], bad_entry=["this should", ["fail"]]
    )


# TODO: parameterize
# TODO: write tests
def test_dimnamecollection_to_list():
    """
    Tests :py:func:`scores.typing.dimnamecollection_to_list`.

    Cases:
    - single Hashable
    - list of Hashables
    - tuple: should raise warning
    - list of tuples: should not raise warning

    Condition:
    - output should be a list of Hashables

    Also partially tests :py:func:`scores.typing.check_dimnamecollection`.
    """
    assert dimnamecollection_to_list("x") == ["x"]
    # numbers are also hashable
    assert dimnamecollection_to_list(4) == [4]
    # tuples are hashable but will raise a warning due to ambiguity
    _t = ("a", 1)
    with pytest.warns(UserWarning, match=r"Ambiguous.*{}".format(re.escape(str(_t)))):
        assert dimnamecollection_to_list(_t) == [_t]
    # no effect on something that is already an iterable
    assert dimnamecollection_to_list(["x", "y", "z"]) == ["x", "y", "z"]


# TODO: parameterize
def test_check_dimnamecollection():
    """
    Tests :py:func:`scores.typing.check_dimnamecollection`.

    Cases:
    - input types that throw an error
    - ... a warning (e.g. ambiguous types)
    - ... don't raise anything (correct type)

    Also covers :py:func:`scores.typing._is_dim_name_collection`.
    """

    def _expect_error(_t: DimNameCollection):
        with pytest.raises(TypeError, match=r"Invalid type.*{}".format(re.escape(str(_t)))):
            check_dimnamecollection(_t)

    def _expect_no_error_no_warn(_t: DimNameCollection):
        try:
            check_dimnamecollection(_t)
        except Exception:
            return False
        return True

    def _expect_ambg_warning(_t: DimNameCollection):
        with pytest.warns(UserWarning, match=r"Ambiguous.*{}".format(re.escape(str(_t)))):
            check_dimnamecollection(_t)

    # error - wrong type - dicts are not hashable (but their entries are)
    _expect_error([{"x": 1, "y": 2}])

    # error - nested types
    _expect_error([["potato", "tapioca"]])
    _expect_error(["a", ["b", "c", ["d"]]])

    # warning - ambiguous type - tuple
    _expect_ambg_warning((1, 2, 3, 4))

    # success - list of integers
    _expect_no_error_no_warn([1, 2, 3, 4])

    # success - mixed hashes is allowed
    _expect_no_error_no_warn(["1", 2, "3", 4])

    # success - set of strings
    _expect_no_error_no_warn(set(["a", "b", "c", "d"]))

    # success - string
    _expect_no_error_no_warn("potato")

    # sucecss - integer
    _expect_no_error_no_warn(42)
