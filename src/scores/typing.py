"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

from collections.abc import Hashable, Iterable
from typing import TypeAlias, Union

import pandas as pd
import xarray as xr

# FlexibleDimensionTypes should be used for preserve_dims and reduce_dims in
# all cases across the repository.
FlexibleDimensionTypes: TypeAlias = Iterable[Hashable]

# Xarraylike data types should be used for all forecast, observed and weights.
# However, currently some are specified as DataArray only.
XarrayLike: TypeAlias = Union[xr.DataArray, xr.Dataset]

# FlexibleArrayType *may* be used for various arguments across the scores
# repository; but are not establishing a standard or expectation, beyond the
# functions in which they are used.
FlexibleArrayType: TypeAlias = Union[XarrayLike, pd.Series]


def is_flexibledimensiontypes(
    val: Iterable[object],
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[FlexibleDimensionTypes]:
    """
    Determines if the input is a iterable of flexible dimensions. Essentially a
    ``Iterable`` of ``Hashable``s

    Returns:
        True ONLY IF ``val`` is ``Iterable`` and its elements are ``Hashable``;
        ELSE False
    """
    assert isinstance(val, Iterable)
    return all(lambda _x: isinstance(_x, Hashable), val)


def is_xarraylike(
    val: object,
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[XarrayLike]:
    """
    Determines wither ``val`` is XarrayLike.

    ``isinstance(val, XarrayLike)`` currently will fail since it is a union type

    Returns:
        True ONLY IF ``val`` is  a `xr.Dataset`` or ``xr.DataArray``;
        ELSE False
    """
    return isinstance(val, xr.Dataset) or isinstance(val, xr.DataArray)


def all_same_xarraylike(
    val: Iterable[object],
    /,  # enforce position only - typeguards can only take one arg
) -> TypeGuard[Iterable[XarrayLike]]:
    """
    Check whether all the elements in val are XarrayLike and of the same type

    Returns:
        True ONLY IF ``val`` is ``Iterable`` and its elements are either _all_
        ``xr.Dataset`` or _all_ ``xr.DataArray`` (exclusive or - i.e. xor);
        ELSE False
    """
    is_ds = lambda _x: isinstance(_x, xr.Dataset)
    is_da = lambda _x: isinstance(_x, xr.Dataarray)
    all_ds = _typecheck_iterable(is_ds, skip_none=False)
    all_da = _typecheck_iterable(is_da, skip_none=False)

    # safety: dev/testing => impossible to be both all dataarray AND all dataset
    assert not (all_ds(val) == True and all_da(val) == True)

    # with the above assert, "or" now behaves as a "xor";
    # i.e. only True if one is True and the other isn't.
    return all_ds(val) or all_da(val)
