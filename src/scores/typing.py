"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

import copy
from collections.abc import Hashable, Iterable
from enum import Enum
from typing import Union

import pandas as pd
import xarray as xr

# FlexibleDimensionTypes should be used for preserve_dims and reduce_dims in all
# cases across the repository
FlexibleDimensionTypes = Iterable[Hashable]

# Xarraylike data types should be used for all forecast, observed and weights
# However currently some are specified as DataArray only
XarrayLike = Union[xr.DataArray, xr.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in
FlexibleArrayType = Union[XarrayLike, pd.Series]


class XarrayTypeMarker(Enum):
    """
    xarray type marker: used to mark ``xr.Dataset`` and ``xr.DataArray`` before they are unified
    into ``LiftedDataset``

    .. important::

        For INTERNAL use only - NOT for public API.
    """

    #: invalid type
    INVALID = -1
    #: maps to ``xr.Dataset``
    DATASET = 1
    #: maps to ``xr.DataArray``
    DATAARRAY = 2


def is_xarraylike(maybe_xrlike: XarrayLike) -> bool:
    """
    Returns True if XarrayLike else False
    """
    return isinstance(maybe_xrlike, (xr.Dataset, xr.DataArray))


def assert_xarraylike(maybe_xrlike: XarrayLike):
    """
    Runtime assert for Xarraylike: For dev/testing only
    """
    err_msg: str = f" Runtime type check failed: {maybe_xrlike} != xr.Dataset or xr.DataArray"
    if not is_xarraylike(maybe_xrlike):
        raise TypeError(err_msg)


def is_lifteddataset(maybe_lds: LiftedDataset) -> bool:
    """
    Returns True if LiftedDataset else False
    """
    return isinstance(maybe_lds, LiftedDataset)


def assert_lifteddataset(maybe_lds: LiftedDataset):
    """
    Runtime assert for LiftedDataset: For dev/testing only
    """
    err_msg: str = f" Runtime type check failed: {maybe_lds} != scores.typing.LiftedDataset"
    if not is_lifteddataset(maybe_lds):
        raise TypeError(err_msg)
