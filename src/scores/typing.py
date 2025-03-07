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


class LiftedDataset:
    """
    Higher order datatype that lifts a ``xr.DataArray`` data array into a dataset, this way it is
    SUFFICIENT for functions to ONLY be compatible with datasets, even if a data array is provided
    as input.

    .. important::

        For INTERNAL use only - NOT for public API.

        In particular, any errors thrown by this function needs to be handled by the caller. As the
        errors are mainly aimed for development and testing. Ideally, they should not be raised in
        runtime; and if they must - they should be caught within the calling function and re-raised
        with a more helpful message.

    This class exists is simply an "aid" to avoid repeated logic and branching.  To dispatch to
    common utility functions, use ``.ds`` to get the underlying dataset. Or alternatively, use
    ``LiftedDataset.inner_ref()`` to get a refence to the original type (more expensive).

    Only call ``LiftedDataset.raw()`` if you want to consume (or destroy) ``LiftedDataset`` wrapper.
    I.e. only want to deal with the inner data for the rest of the execution scope.

    .. see-also::

        :py:meth:`LiftedDatasetUtils.lift_fn_ret` has a convenient wrapper to provide compatibility
        to ``LiftedDataset`` using pre-existing utility functions that only work with ``XarrayLike``.
    """

    #: dummy variable name - internal static variable
    _DUMMY_DATAARRAY_VARNAME: str = "dummyvarname"

    def __init__(self, xr_data: XarrayLike):
        """
        Converts a ``xr.DataArray`` to a ``xr.Dataset``, preserving its original type.
            - For a ``xr.Dataset``, this is essentially a shallow wrapper, with type metadata.
            - For a ``xr.DataArray``, this operation is more expensive as it wraps it inside a
              dataset first and unwrapping it requires extra logic
        """

        err_invalid_type: str = """
            Invalid type for `XarrayLike`, must be a `xr.Dataset` or a `xr.DataArray` object.
        """

        self.reset()

        if isinstance(xr_data, xr.Dataset):
            # dataset - simply wrap it
            self.ds = xr_data
            self.xr_type_marker = XarrayTypeMarker.DATASET
        elif isinstance(xr_data, xr.DataArray):
            # dataarray - lift to dataset then wrap, insert dummy name (required) if unnamed
            if xr_data.name is None or not xr_data.name:
                self.dummy_name = True
                self.ds = xr_data.to_dataset(name=LiftedDataset._DUMMY_DATAARRAY_VARNAME)
            else:
                self.dummy_name = False
                self.ds = xr_data.to_dataset()
            self.xr_type_marker = XarrayTypeMarker.DATAARRAY
        else:
            # type assert: input is not XarrayLike, raise error
            raise TypeError(err_invalid_type)

    def make_dataarray_ref(self) -> xr.DataArray:
        """
        Retrieves the underlying reference data array, removes dummy names
        """
        if not self.is_dataarray():
            raise TypeError("Cannot revert name for xr.Dataset, only xr.DataArray")

        var_names: list[Hashable] = list(self.ds.data_vars.keys())

        # safety: we should only have 1 variable if this is actually a data array
        assert len(var_names) == 1

        # retrieve array
        da: xr.DataArray = self.ds.data_vars[var_names[0]]

        # revert dummy name if it was assigned one
        if self.dummy_name:
            da.name = None

        return da

    def is_valid(self) -> bool:
        """
        return True if the LiftedDataset obj is valid
        """
        return self.xr_type_marker != XarrayTypeMarker.INVALID

    def is_dataarray(self) -> bool:
        """
        return True if the LiftedDataset obj is an xr.DataArray
        """
        return self.xr_type_marker == XarrayTypeMarker.DATAARRAY

    def is_dataset(self) -> bool:
        """
        return True if the LiftedDataset obj is an xr.Dataset
        """
        return self.xr_type_marker == XarrayTypeMarker.DATASET

    def reset(self) -> None:
        """
        Resets any members in this class, allowing them to be dereferenced and garbage collected.

        It is mainly used by ``.raw()`` to consume this structure and reproduce the original inner
        dataset or data array; and ``.__init__()`` to construct an object from this class.

        .. warning::

           This action is irreversable, and should only be called internally to this class.

        """
        self.dummy_name = False
        self.xr_type_marker = XarrayTypeMarker.INVALID
        self.ds = None

    def raw(self) -> XarrayLike:
        """
        Like ``.inner_ref()`` but consumes any references in the ``LiftedDataset`` structure,
        invalidating it.  This is usually the last operation before returning the result to the
        user.

        .. important::
            To reiterate: this is a CONSUMING operation, that is to say it will retract the
            ``LiftedDataset`` back to its original form, invalidating the wrapper structure. If you
            simply want a reference, use the ``.ds`` property or ``.inner_ref()`` to retrieve the
            underlying dataset or dataarray.
        """
        ret_xr_data: XarrayLike = self.inner_ref()
        # remove and invalidate any inner references from THIS structure and return to caller
        self.reset()
        return ret_xr_data

    def inner_ref(self) -> XarrayLike:
        """
        Retracts the inner dataset or data array contained in this lifted object.

        Uses the type marking as well as whether or not dummy names have been assigned to find a
        unique retraction such that the raw data can be recovered.

        Unlike ``.raw()`` this does not reset the overarching lifted obj, allowing it to be reused.

        .. note::
            For most intermediate operations prefer using ``.inner_ref()``, so that it doesn't
            invalidate the inner data - allowing it to be reused. However, for a finalizing
            operation, use ``.raw()`` so that the lifted obj and any stale references are
            appropriately cleaned up.
        """
        # safety: cannot call this function on an invalid initialisation
        assert self.is_valid()
        # dataarray: make ref and return
        if self.is_dataarray():
            return self.make_dataarray_ref()
        # safety assert: can only be a dataset
        assert self.is_dataset()
        return self.ds


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
