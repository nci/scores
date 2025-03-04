"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

from collections.abc import Hashable, Iterable
from enum import Enum
from typing import Union

import pandas as pd
import xarray

# FlexibleDimensionTypes should be used for preserve_dims and reduce_dims in all
# cases across the repository
FlexibleDimensionTypes = Iterable[Hashable]

# Xarraylike data types should be used for all forecast, observed and weights
# However currently some are specified as DataArray only
XarrayLike = Union[xarray.DataArray, xarray.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in
FlexibleArrayType = Union[XarrayLike, pd.Series]


class XarrayTypeMarker(Enum):
    """
    .. important::

        For INTERNAL use only. This is currently an EXPERIMENTAL type utility, and not to be
        used in typehints in public API. However, it can be used as part of internal functions.

    xarray type marker: used to mark ``xr.Dataset`` and ``xr.DataArray`` before they are unified
    into ``LiftedDataset``
    """

    #: invalid type
    INVALID = -1
    #: maps to ``xr.Dataset``
    DATASET = 1
    #: maps to ``xr.DataArray``
    DATAARRAY = 2


class LiftedDataset:  # pylint: disable-msg=too-few-public-methods
    """
    .. important::

        For INTERNAL use only. This is currently an EXPERIMENTAL type utility, and not to be
        used in typehints in public API. However, it can be used as part of internal functions.

    Higher order datatype that lifts a ``xr.DataArray`` data array into a dataset, this way it is
    SUFFICIENT for functions to ONLY be compatible with datasets, even if a data array is provided
    as input.

    This class exists is simply an "aid" to avoid repeated logic and branching.

    .. note::

        This is basically a wrapper with some metadata to preserve structure.

    .. caution::

        There may be cases where the functions using this class may have to iteratively perform
        operations on the underlying data arrays in the dataset, because no native implementation
        exists for ``xr.Dataset``, while it does for ``xr.DataArray``.

        Note, this is not unique to this implementation as it needs to be done anyway, since we
        are supporting both types. Regardless, do not assume ``xr.Dataset`` will "automagically"
        handle anything ``xr.DataArray`` can (though for most cases there is an equivilent method).
    """

    #: dummy variable name - internal static variable
    _DUMMY_DATAARRAY_VARNAME: str = "dummyvarname"

    def __init__(self, xr_data: XarrayLike):
        """
        Converts a ``xr.DataArray`` to a ``xr.Dataset``, preserving its original type.
        For a ``xr.Dataset``, this is just a shallow wrapper.
        """
        err_invalid_type: str = """
            Invalid type for `XarrayLike`, must be a `xr.Dataset` or a `xr.DataArray` object.
        """
        self.dummy_name: bool = False
        self.xr_type_marker: XarrayTypeMarker = XarrayTypeMarker.INVALID
        # handle dataset input
        if isinstance(xr_data, xarray.Dataset):
            self.xr_type_marker = XarrayTypeMarker.DATASET
            self.ds = xr_data  # nothing to lift - already a dataset
        # handle data array input
        elif isinstance(xr_data, xarray.DataArray):
            self.xr_type_marker = XarrayTypeMarker.DATAARRAY
            # a data array may have no name, but a dataset MUST have a name for all its variables;
            # so give it a dummy name if it doesn't have one.
            if xr_data.name is None or not xr_data.name:
                self.ds = xr_data.to_dataset(name=LiftedDataset._DUMMY_DATAARRAY_VARNAME)
                self.dummy_name = True
            # ...in this case it already had a name.
            else:
                self.ds = xr_data.to_dataset()
        # invalid: if we reached this point, then we're dealing with an illegal runtime type
        else:
            raise TypeError(err_invalid_type)

    def raw(self) -> XarrayLike:
        """
        Returns the reference to the underlying dataset or data array
        """
        # safety: `raw` can only be accessed by a constructed object, which would raised a
        # `TypeError`, if INVALID (see: ``__init__`).
        assert self.xr_type_marker != XarrayTypeMarker.INVALID
        # data array: remove dummy name (if assigned one) and convert it back to an array
        if self.xr_type_marker == XarrayTypeMarker.DATAARRAY:
            keys: list[Hashable] = list(self.ds.variables.keys())
            # safety: we should only have 1 variable if this is actually a data array
            assert len(keys) == 1
            da: xarray.DataArray = self.ds.data_vars[keys[0]]
            if self.dummy_name:
                da.name = None
            return da
        # safety: assert last remaining branch, pylint doesn't like unspecified returns so we
        # can't use `(else|if|elif) self.xr_type_marker == XarrayTypeMarker.DATASET:`
        assert self.xr_type_marker == XarrayTypeMarker.DATASET
        return self.ds


def check_lds_same_type(*lds: LiftedDataset):
    """
    Checks if the internal data types for the input :py:class:`LiftedDataset` (``lds``) have the
    same type marker (see: :py:class:`XarrayTypeMarker`).

    Args:
        *lds: Variadic args of type :py:class:`LiftedDataset`
    """
    err_invalid_type: str = """
        Input type is not a `LiftedDataset`. Did you attempt to pass in `xr.Dataset` or
        xr.DataArray` instead of `scores.typing.LiftedDataset`?
    """

    err_inconsistent_type: str = """
        The provided xarray data inputs are not of same type, they must ALL EXCLUSIVELY be ONLY
        `xr.Dataset`, otherwise, ONLY `xr.DataArray`.
    """

    for d in lds:
        if not isinstance(d, LiftedDataset):
            raise TypeError(err_invalid_type)

    all_ds = all(d.xr_type_marker == XarrayTypeMarker.DATASET for d in lds)
    all_da = all(d.xr_type_marker == XarrayTypeMarker.DATAARRAY for d in lds)

    # both cannot be False (mixed types), and both cannot be True (impossible scenario)
    if all_ds == all_da:
        raise TypeError(err_inconsistent_type)
