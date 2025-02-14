"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

from collections.abc import Hashable, Iterable
from typing import Any, TypeAlias, TypeGuard, Union

import pandas as pd
import xarray

# Dimension name. `xarray` recommends `str` for dimension names, however, it doesn't imposes
# restrictions on generic hashables, so we also need to support hashables.
DimName: TypeAlias = Hashable

# `FlexibleDimensionTypes` should be used for `preserve_dims` and `reduce_dims` in all cases across
# the repository. NOTE: may be replaced by `DimNameCollection` down the track.
FlexibleDimensionTypes: TypeAlias = Iterable[DimName]

# `XarrayLike` data types should be used for all forecast, observed and weights. However, currently
# some are specified as DataArray only
XarrayLike: TypeAlias = Union[xarray.DataArray, xarray.Dataset]

# `FlexibleArrayType` *may* be used for various arguments across the scores repository, but are not
# establishing a standard or expectation beyond the function they are used in.
FlexibleArrayType: TypeAlias = Union[XarrayLike, pd.Series]

# Generic collection that supports both `DimName`, a single hashable and `FlexibleDimensionTypes`.
# A iterable collection of dimension names. Useful for applying functions that can support both
# single names or collections e.g. `gather_dimensions` which currently does these checks internally.
#
# NOTE: may replace `FlexibleDimensionTypes` down the line - this has more utility functions.
DimNameCollection: TypeAlias = Union[DimName, Iterable[DimName]]

ERROR_DIMNAMECOLLECTION: str = """
Invalid type for dimension name collection: `DimensionNameCollection` must be either a `Hashable`
(e.g. `str`) or an `Iterable` of `Hashable`s e.g. `list[str]`.
"""

WARN_AMBIGUOUS_DIMNAMECOLLECTION: Callable[[], str] =  lambda t: f"""
Ambiguous `DimNameCollection`. input: {t} is `Iterable` AND `Hashable`. `Hashable` takes priority.
If you intended to provide an `Iterable`, consider using a non-hashable collection like a `list`.
"""

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
        raise TypeError(ERROR_DIMNAMECOLLECTION)


def _is_dim_name_cln(t: Any) -> TypeGuard[DimNameCollection]:
    """
    Static type checking for ``DimNameCollection``.

    Returns:
        True if input type is valid at runtime, False otherwise.

    .. note::
        Only returns a boolean flag for static checking, use :py:func:`_check_dim_name_cln` if you
        want to raise a runtime error.
    """
    is_name = lambda x: isinstance(x, DimName)
    is_iterable = lambda x: isinstance(x, Iterable)
    are_all_names = lambda x: all(map(is_name, x))

    if is_name(t) and is_iterable(t):
        UserWarning(WARN_AMBIGUOUS_DIMNAMECOLLECTION)

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
    _check_dim_name_cln(dim_cln)
    return list(_lift_dim_name(dim_cln))


def _lift_dim_name(dim_name: DimName) -> DimNameCollection:
    """
    lifts a single ``DimName`` (e.g. str) to a collection (e.g. list[str]). For example, allows ``DimName``
    to be compatible with functions that only deal with ``DimNameCollection`` or ``FlexibleDimensionTypes``.

    Returns:
        Returns list[Hashable] if input is a DimName, otherwise does nothing.
    """
    ret : DimNameCollection = dim_name  # default - do nothing
    if isinstance(dim_name, DimName):
        ret = [dim_name]
    _check_dim_name_cln(ret)
    return ret
