"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

from collections.abc import Hashable, Iterable
from typing import TypeGuard, Union

import pandas as pd
import xarray as xr

# FlexibleDimensionTypes should be used for preserve_dims and reduce_dims in
# all cases across the repository.
FlexibleDimensionTypes = Iterable[Hashable]

# Xarraylike data types should be used for all forecast, observed and weights.
# However, currently some are specified as DataArray only.
XarrayLike = Union[xr.DataArray, xr.Dataset]

# FlexibleArrayType *may* be used for various arguments across the scores
# repository; but are not establishing a standard or expectation, beyond the
# functions in which they are used.
FlexibleArrayType = Union[XarrayLike, pd.Series]


def is_flexibledimensiontypes(
    val: Iterable[Hashable] | Hashable,
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[FlexibleDimensionTypes]:
    """
    Determines if the input is a iterable of flexible dimensions. Essentially a
    ``Iterable`` of ``Hashable``s

    .. important::
        Most `reduce_dims` and `preserve_dims` arguments allow a single hash or multiple
        hashes.

        For simplicity tuples are treated as iterables by this check. (since they are both
        iterable and hashable). If you really want to use a tuple as a hash, provide a
        list of tuples instead of the tuple directly.

        In general xarray prefers the use of strings as dimension types, but doesn't
        madate it - thus loose support is maintained for an arbitrary hashable.

        However, custom hashable/iterable classes are not well supported by this check.

    .. note::

        For internal use only:

            - TypeGuards help mypy resolve complex static type checks - avoids
              the need to use "type: ignore" when types are ambiguous.

            - Often used in conjunction with, type-based control flow, asserts
              or raising TypeError at runtime.

    Returns:
        True ONLY IF ``val`` is ``Iterable`` and its elements are ``Hashable``;
        ELSE False
    """
    # A single hashable is allowed by most functions, but do not check tuples yet as we
    # treat them as Iterables.
    if not isinstance(val, tuple) and isinstance(val, Hashable):
        return True

    # safety: it should be a iterable if we reached this point
    assert isinstance(val, Iterable)

    ret = all(map(lambda _x: isinstance(_x, Hashable), val))

    return ret


def is_xarraylike(
    val: object,
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[XarrayLike]:
    """
    Determines wither ``val`` is XarrayLike.

    ``isinstance(val, XarrayLike)`` currently will fail since it is a union type

    .. note::

        For internal use only:

            - TypeGuards help mypy resolve complex static type checks - avoids
              the need to use "type: ignore" when types are ambiguous.

            - Often used in conjunction with, type-based control flow, asserts
              or raising TypeError at runtime.

    Returns:
        True ONLY IF ``val`` is  a `xr.Dataset`` or ``xr.DataArray``;
        ELSE False
    """
    return isinstance(val, (xr.Dataset, xr.DataArray))


def all_same_xarraylike(
    val: Iterable[object],
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[Iterable[XarrayLike]]:
    """
    Check whether all the elements in val are XarrayLike and of the same type

    .. note::

        For internal use only:

            - TypeGuards help mypy resolve complex static type checks - avoids
              the need to use "type: ignore" when types are ambiguous.

            - Often used in conjunction with, type-based control flow, asserts
              or raising TypeError at runtime.

    Returns:
        True ONLY IF ``val`` is ``Iterable`` and its elements are either _all_
        ``xr.Dataset`` or _all_ ``xr.DataArray`` (exclusive or - i.e. xor);
        ELSE False
    """
    assert isinstance(val, Iterable)
    all_ds = all(map(lambda _x: isinstance(_x, xr.Dataset), val))
    all_da = all(map(lambda _x: isinstance(_x, xr.DataArray), val))

    # safety: dev/testing => impossible to be both all dataarray AND all dataset
    assert not (all_ds is True and all_da is True)

    # with the above assert, "or" now behaves as a "xor";
    # i.e. only True if one is True and the other isn't.
    return all_ds or all_da
