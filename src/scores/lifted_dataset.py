"""
Strucutral helper used to consolidate logic between xarray datasets and data array.

The concept is that datasets are supersets of data arrays and contain all the necessary information
to perform operations on data arrays. Therefore, instead of branching logic to handle data sets and
data arrays separately. It makes things cleaner to just deal with one type. Dataset being the
encompassing type is the natural choice.

`LiftedDataset` is a class that constructs a structural object from either a data array or a
dataset.

If a data array is provided, it will lift it to a dataset and store any metadata required to revert
it back to normal.

For a dataset, it is a transparent wrapper since lifting doesn't actually do anything (essentially
an identity operation).

Any function that utilizes datasets will thus only 
"""

import functools
from collections.abc import Hashable, Iterable
from typing import Callable

import xarray as xr

from scores.typing import (
    XarrayLike,
    XarrayTypeMarker,
    assert_lifteddataset,
    assert_xarraylike,
    is_lifteddataset,
    is_xarraylike,
)


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

        :py:meth:`LiftedDatasetUtils.lift_xrdata` has a convenient wrapper to provide compatibility
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


class LiftedDatasetUtils:
    """
    namespace class containing utility methods for LiftedDataset.
    """

    ERROR_INVALID_LIFTED_DATASET_TYPE: str = """
    Input type is not a `LiftedDataset`. Did you attempt to pass in `xr.Dataset` or xr.DataArray`
    instead of `scores.typing.LiftedDataset`?
    """

    ERROR_INVALID_LIFTFUNC_RETTYPE: str = """
    Functions lifted by `lift_args_and_ret_to_ds` must return either a `xr.DataArray` or a `xr.Dataset`
    (preferrable).
    """

    ERROR_INCONSISTENT_TYPES: str = """
    The provided xarray data inputs are not of same type, they must ALL EXCLUSIVELY be ONLY
    `xr.Dataset`, otherwise, ONLY `xr.DataArray`.
    """

    WARN_EMPTYARGS_FOR_ALLSAMETYPECHECK: str = """
    No args provided for XarrayLike `all_same_type` checks. If you are using `lift_args_to_ds*` make sure the
    function you are wrapping actually uses `xr.DataArray` or `xr.Dataset`, otherwise it maybe
    clearer to just lift the output directly.
    """

    @staticmethod
    def lift_xrlike(assert_xrdatatype_allequal: bool = False, lift_rettype: bool = False) -> Callable:
        """
        Wrapper to maintain backward compatibility with legacy functions that use ``XarrayLike``
        arguments instead of ``LiftedDataset``. Arguments to the inner ``fn`` that are not
        ``XarrayLike`` or already ``LiftedDataset``s are ignored.

        Args:

            require_same_xrdatatype: If True, asserts that all input types for `XarrayLike` args are
                the same i.e. ALL datasets or ALL dataarrays.

            lift_rettype: Whether or not to also lift the return type to `LiftedDataset`.

        # IMPORTANT:
        .. important::

            For INTERNAL use only - NOT for public API.

            Addtionally:

                1. The purpose of this wrapper is to act as a transient helper while the concept of
                   a ``LiftedDataset`` is refactored on to LEGACY functions and tested appropriately.

                2. This means any new functions or functions that can be directly applied to
                   `LiftedDataset`(s) should not rely on this wrapper

                3. if the inner function is already compatible with lifted datasets then you
                   shouldn't be using this wrapper.

                4. `inner_ref` is used instead of `raw` when applying the inner data types to the
                   inner functions as the caller may still want to retain the lifted structure.
                   (`raw` is a destructive operation)
        Usage:

        .. code-block:: python

            @lift_xrlike(assert_same_xrdatatype=True)
            def inner_computation_with_xr(x: xr.Dataset, y: int, *, z: xr.DataArray) -> bool:
                ... # some computation that returns a bool

            # because assert_same_xrdatatype=True, this will raise an error since x and z are not
            # the same type. On the other hand this:

            @lift_args_to_ds(assert_same_xrdatatype=True)
            def inner_computation_with_xr(x: xr.Dataset, y: int, *, z: xr.Dataset) -> bool:
                ... # some computation that returns a bool

            # Will work fine.

            # If we want to bypass the type check for whatever reason we could do:

            @lift_args_to_ds(assert_same_xr_data_type=False)
            def inner_computation_with_xr(x: xr.DataArray, y: int, *, z: xr.Dataset) -> bool:
                ... # some computation that returns a bool

            ds_x, int_y, da_z = ...

            # The wrapped function signature is like so:
            # args: (LiftedDataset, int), kwargs: {"z": LiftedDataset} -> ret: bool

            result = inner_computation_with_xr(LiftedDataset(ds_x), y, z=LiftedDataset(ds_z))

            # Additionally if we want to lift the return type we can specify this too:
            @lift_args_to_ds(lift_rettype=True)
            def inner_computation_with_xr(x: XarrayLike, y: int, *, z: XarrayLike) -> XarrayLike:
                ... # some computation that returns a xarraylike

            # The wrapped function signature now includes a lifted return type:
            # args: (LiftedDataset, int), kwargs: {"z": LiftedDataset} -> ret: LiftedDataset
            # note that the return type of the inner function MUST be XarrayLike for this to work.

            result = inner_computation_with_xr(LiftedDataset(ds_x), y, z=LiftedDataset(ds_z))

            # ... do stuff with result

            # extract the raw inner data type, consume the `LiftedDataset` wrapper and return
            return result.raw()  # return type = XarrayLike
                                 # i.e. original function signature is untouched
        """

        def _lift_xrdata_to_liftedds(fn: Callable) -> Callable:
            """
            lifts all xaraylike args/kwargs in `fn` to `LiftedDataset` (lds)
            """

            @functools.wraps(fn)
            def _wrapper(*args, **kwargs):
                def _update_xrlike_inputs(_args, _kwargs, _fn):
                    for i, v in enumerate(_args):
                        if is_xarraylike(v):
                            _args[i] = v
                    for k, v in _kwargs.items():
                        if is_xarraylike(v):
                            _kwargs[k] = _fn(v)

                # lift any xarraylike to lifteddataset
                _update_xrlike_inputs(args, kwargs, LiftedDataset)

                # assert same type if configured to do so
                lds_args = list(filter(is_lifteddataset, [*args] + [*kwargs.values()]))
                if assert_same_xr_data_type and len(lds_args) > 0:
                    LiftedDatasetUtils.all_same_type(*lds_args)

                # consolidate everything back to its original type before running inner function
                # since the inner function is not compatible with lifteddatasets.
                _update_xrlike_inputs(args, kwargs, lambda _x: _x.inner_ref())

                # runner inner function and lift return type if configured to do so
                ret = fn(*args_new, **kwargs_new)
                if lift_rettype:
                    assert_xarraylike(ret)  # safety: only xarraylike can be lifted
                    LiftedDataset(ret)

                return ret

            return _wrapper

        return _lift_xrdata_to_liftedds

    @classmethod
    def all_same_type(cls, *lds) -> XarrayTypeMarker:
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
        assert len(lds) > 0

        # runtime check for correct input types
        def _check_single_input(_lds: LiftedDataset):
            assert_lifteddataset(_lds)
            assert _lds.is_valid()

        # list simply consumes the iterator
        list(map(_check_single_input, lds))

        # consistency checks
        all_dataset: bool = all(map(lambda _x: _x.is_dataset(), lds))
        all_dataarray: bool = all(map(lambda _x: _x.is_dataarray(), lds))
        # safety assert: cannot be both at the same time - something wrong in the code
        assert not (all_dataset and all_dataarray)

        # return appropriate type marker
        if all_dataset:
            return XarrayTypeMarker.DATASET
        if all_dataarray:
            return XarrayTypeMarker.DATAARRAY

        # otherwise raise exception due to inconsistent types
        raise TypeError(cls.ERROR_INCONSISTENT_TYPES)
