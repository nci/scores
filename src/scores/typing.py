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


class LiftedDataset:
    """
    Higher order datatype that lifts a ``xr.DataArray`` data array into a dataset, this way it is
    SUFFICIENT for functions to ONLY be compatible with datasets, even if a data array is provided
    as input.

    This class exists is simply an "aid" to avoid repeated logic and branching.  To dispatch to
    common utility functions, use ``.ds`` to get the underlying dataset. Only call ``.raw()`` if you
    no longer intend use the wrapped structure ``LiftedDataset`` and want to retract to the original
    xarray datatype. While, the lifted version isn't explicitly destroyed - calling ``.raw()`` will
    invalidate the lifted structure, tainting it and removing any references to the underlying
    dataset.

    For an EXPERIMENTAL auto-dispatch see: :py:meth:`LiftedDataset.lift_fn`. This automatically
    "lifts" the function itself, handling any input arguments that are ``LiftedDatasets`` as if they
    were ``XarrayLike``. This is useful to maintain compatiblity with legacy utility functions.

    .. important::

        For INTERNAL use only. This is currently an EXPERIMENTAL type utility, and not to be
        used in typehints in public API. However, it can be used as part of internal functions.

    .. note::
        In particular, any errors thrown by this function needs to be handled by the caller. As the
        errors are mainly aimed for development and testing. Ideally, they should not be raised in
        runtime.

        However, if there are "exceptions" that need to propagate to the user, the user will have to
        handle and re-raise any errors appropriately.

        see :py:meth:`~scores.continuous.nse_impl.NseUtils.get_xr_type_marker` for an example.

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
        self.reset()
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

    def is_valid(self) -> bool:
        return self.xr_type_marker != XarrayTypeMarker.INVALID

    def is_dataarray(self) -> bool:
        return self.xr_type_marker == XarrayTypeMarker.DATAARRAY

    def is_dataset(self) -> bool:
        return self.xr_type_marker == XarrayTypeMarker.DATASET

    def reset(self):
        self.dummy_name: bool = False
        self.xr_type_marker: XarrayTypeMarker = XarrayTypeMarker.INVALID
        self.ds: xarray.Dataset | None = None

    def raw(self) -> XarrayLike:
        """
        Returns the reference to the underlying dataset or data array, consuming the
        ``LiftedDataset`` structure.

        .. important::
            To reiterate: this is a CONSUMING operation, that is to say it will retract the
            ``LiftedDataset`` back to its original form, invalidating the wrapper structure. If you
            simply want a reference, use the ``ds`` property.
        """
        # safety: cannot call this function on an invalid initialisation
        assert self.is_valid()
        # get reference to dataset so that it isn't destroyed on reset.
        ds_local: xarray.Dataset = self.ds
        # default to returning the whole dataset, as its non-destructive
        ret_xr_data: XarrayLike = ds_local
        # data array: remove dummy name (if assigned one) and convert it back to an array
        if self.is_dataarray():
            keys: list[Hashable] = list(ds_local.variables.keys())
            # safety: we should only have 1 variable if this is actually a data array
            assert len(keys) == 1
            da: xarray.DataArray = ds_local.data_vars[keys[0]]
            # reset name if dummy_name was provided
            if self.dummy_name:
                da.name = None
            ret_xr_data = da
        # safety: should only be a dataset if we reached this point.
        assert self.is_dataset()
        return ret_xr_data

    # --- static methods here instead of the broader module, mainly for grouping namespace

    @staticmethod
    def as_ds(lds: LiftedDataset):
        """
        Alternative to self.ds, but callable as a static method. Useful for referring to collections
        of lifted datasets, but may have advantages when chaining operations in functional style
        programming.
            - e.g. ``map(LiftedDataset.as_ds, [lds_1, lds_2, ...])``
            - equivilent to ``[ x.ds for x in [lds_1, lds_2, ...]``
        """
        # TODO: figure out how to make this frozen. For a hint look at `xarrays` own `Frozen` class
        return lds.ds

    @staticmethod
    def lift_fn(fn: Callable) -> Callable:
        """
        Wrapper to automatically apply a function that is compatible with ``XarrayLike`` but on
        ``LiftedDataset`` instead. Since ``XarrayLike`` functions should work with ``Datasets``, by
        extension they should work for for the underlying ``LiftedDatasets.ds``

        .. important::

            CAUTION: EXPERIMENTAL

            This is an internal function - not for public API. It is mainly to maintain
            compatibility with existing utility functions that depend on ``XarrayLike`` as inputs.

        Args:
            fn: Any function, but in-particular should only be used with functions that take an
                ``XarrayLike`` argument, but made compatible with ``LiftedDataset`` instead.
        """

        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            # shallow copy is okay, since no data is actually being changed...
            args_new = args.copy()
            kwargs_new = kwargs.copy()
            is_compat = lambda _lds: isinstance(_lds, LiftedDataset) and _lds.is_valid()
            # fixup args -> replace LiftedDataset with LiftedDataset.ds
            for i, v in enumerate(args):
                if iscompat(v):
                    args_new[i] = LiftedDataset.as_ds(args[i])
            # fixup kwargs -> replace LiftedDataset with LiftedDataset.ds
            for k in kwargs.keys():
                if iscompat(kwargs[k]):
                    kwargs_new[k] = LiftedDataset.as_ds(kwargs[k])
            return fn(*args, **kwargs)

        return _wrapper

    @staticmethod
    def lift_fn_ret(fn: Callable[..., XarrayLike]) -> Callable[..., LiftedDataset]:
        """
        Like ``lift_fn`` but also lifts the return type to a ``LiftedDataset``, if possible.

        .. important::

            CAUTION: EXPERIMENTAL

            This is an internal function - not for public API. It is mainly used to maintain
            compatibility with existing utility functions that depend on ``XarrayLike`` as inputs
            and returns an ``XarrayLike`` but the caller would like to provide ``LiftedDataset`` as
            the args and also expects it as the return type

            New utility methods should just directly operate on ``LiftedDataset`` types.

            i.e. this wrapper tries to preserve isomorphism, while ``lift_fn`` is destructive and
            the caller could receive any output.

        .. see-also::

            :py:meth:`LiftedDataset.lift_fn`
        """
        fn_lifted = LiftedDataset.lift_fn(fn)

        @functools.wraps(fn_lifted)
        def _wrapper(*args, **kwargs):
            ret: XarrayLike = fn_lifted(*args, **kwargs)
            return LiftedDataset(ret)

        return _wrapper

    @staticmethod
    def check_lds_same_type(*lds: LiftedDataset) -> XarrayTypeMarker:
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
        # need at least one argument to check
        assert len(lds) > 0
        ret_marker: XarrayTypeMarker = XarrayTypeMarker.INVALID
        # define error messages - not global, as this is a internal function and these errors are more
        # relevant to a developer and in unittests.
        err_invalid_type: str = """
            Input type is not a `LiftedDataset`. Did you attempt to pass in `xr.Dataset` or
            xr.DataArray` instead of `scores.typing.LiftedDataset`?
        """
        err_inconsistent_type: str = """
            The provided xarray data inputs are not of same type, they must ALL EXCLUSIVELY be ONLY
            `xr.Dataset`, otherwise, ONLY `xr.DataArray`.
        """
        # check that all input types are lifted datasets
        for d in lds:
            if not isinstance(d, LiftedDataset):
                raise TypeError(err_invalid_type)
        # do check: homogeneous xr_data types
        all_ds = all(d.xr_type_marker == XarrayTypeMarker.DATASET for d in lds)
        all_da = all(d.xr_type_marker == XarrayTypeMarker.DATAARRAY for d in lds)
        # both cannot be False (mixed types), and both cannot be True (impossible scenario)
        if all_ds == all_da:
            raise TypeError(err_inconsistent_type)
        # return marker type for dataset
        elif all_ds:
            marker = XarrayTypeMarker.DATASET
        # return marker type for data array
        elif all_da:
            marker = XarrayTypeMarker.DATAARRAY
        # saftey check: would have raised TypeError earlier - but may not be obvious to pylint
        assert marker != XarrayTypeMarker.INVALID
        return ret_marker
