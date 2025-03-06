"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

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
            self.xr_type_marker = XarrayTypeMarker.DATASET
            self.ds = xr_data  # nothing to lift - already a dataset
        elif isinstance(xr_data, xr.DataArray):
            self.xr_type_marker = XarrayTypeMarker.DATAARRAY
            # if data array has no name give it a dummy name - required for dataset
            if xr_data.name is None or not xr_data.name:
                self.ds = xr_data.to_dataset(name=LiftedDataset._DUMMY_DATAARRAY_VARNAME)
                self.dummy_name = True
            else:
                self.ds = xr_data.to_dataset()
        else:
            # type assert: input is not XarrayLike, raise error
            raise TypeError(err_invalid_type)

    def is_valid(self) -> bool:
        """
        return True if the LiftedDataset obj is valid
        """
        ret: bool = self.xr_type_marker != XarrayTypeMarker.INVALID
        return ret

    def is_dataarray(self) -> bool:
        """
        return True if the LiftedDataset obj is an xr.DataArray
        """
        ret: bool = self.xr_type_marker == XarrayTypeMarker.DATAARRAY
        return ret

    def is_dataset(self) -> bool:
        """
        return True if the LiftedDataset obj is an xr.Dataset
        """
        ret: bool = self.xr_type_marker == XarrayTypeMarker.DATASET
        return ret

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
        # get local reference to inner xarray data, so that it doesn't get destructed.
        ret_xr_data: XarrayLike = self.inner_ref()
        # reset to remove reference to dataset in this lifted structure. Essentially invalidating
        # it. `ret_xr_data` should still exist in this scope and be unaffected.
        self.reset()
        # return reference to data to caller, this object should no longer be referring to any data
        # and can be garbage collected, i.e consumed.
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
        # get reference to dataset so that it isn't destroyed on reset.
        ds_local: xr.Dataset = self.ds
        # default to returning the whole dataset, as its non-destructive
        ret_xr_data: XarrayLike = ds_local
        # data array: remove dummy name (if assigned one) and convert it back to an array
        if self.is_dataarray():
            keys: list[Hashable] = list(ds_local.variables.keys())
            # safety: we should only have 1 variable if this is actually a data array
            assert len(keys) == 1
            da: xr.DataArray = ds_local.data_vars[keys[0]]
            # reset name back to None if dummy_name was used in-place.
            if self.dummy_name:
                da.name = None
            ret_xr_data = da
        else:
            # dataset: safety assert - should only be a dataset if we reached this point.
            assert self.is_dataset()
            # intentional repitition - for readiblity
            ret_xr_data = ds_local
        # returning the inner ds will remove any remaining references from this scope.
        return ret_xr_data


def assert_xarraylike(maybe_xrlike: XarrayLike):
    """
    Runtime assert for Xarraylike
    - To assist in type safety for development and testing purposes only
    - Should not be used as a user error as this will be obfuscated depending on compile settings
    """
    if not isinstance(maybe_xrlike, (xr.Dataset, xr.DataArray)):
        raise TypeError(
            f"""
            Runtime type check failed: {maybe_xrlike} != xr.Dataset or xr.DataArray
            """
        )


def assert_lifteddataset(maybe_lds: LiftedDataset):
    """
    Runtime assert for LiftedDataset
    - To assist in type safety for development and testing purposes only
    - Should not be used as a user error as this will be obfuscated depending on compile settings
    """
    if not isinstance(maybe_lds, LiftedDataset):
        raise TypeError(
            f"""
            Runtime type check failed: {maybe_lds} != scores.typing.LiftedDataset
            """
        )
