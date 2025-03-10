"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import functools
import warnings
import weakref
from collections.abc import Hashable, Iterable
from typing import Callable, TypedDict

import numpy as np
import xarray as xr

import scores.continuous
from scores.typing import (
    FlexibleDimensionTypes,
    XarrayLike,
    all_same_xarraylike,
    is_flexibledimensiontypes,
    is_xarraylike,
)
from scores.utils import DimensionError, check_weights, gather_dimensions


def nse_score(
    *,  # Force kwargs only
    fcst: XarrayLike,
    obs: XarrayLike,
    weights: XarrayLike | None = None,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    is_angular: bool = False,
) -> XarrayLike:
    """
    Sets up the main calculation function for NSE.

    Args:
        same as the public API, but enforced to be kwarg only. Uses ``NseScore``
        class for doing the thorough checking.

        Lifts any `XarrayLike` to `xr.Datasets` so that only one type of API
        needs to be maintained, rather than also dealing with `xr.DataArray`.
        Converts it back to its original data type in the end (if it was a
        `xr.DataArray`)

    Returns:
        An ``NseScore`` object. Its ``.nse`` property can be called to run the
        computations. (Also see note below if using ``dask``.)

    Raises:
        UserWarning: If divide by 0 detected in ANY element.
                     raised by :py:meth:`NseUtils.check_nonzero_obsvar`

    .. important::

        Divide by zero is ALLOWED - to accomodate scenarios where all obs
        entries in the group being reduced is constant (0 obs variance).

        While these may cause divide by zero errors, they should not halt
        execution of computations for other valid coordinates - so a warning is
        issued instead to prompt the user to double check the data.

        It may also be that divide by zero is unavoidable - in which case we
        still want to return the correctly calculated values. To this end, this
        is how ``numpy`` resolves divide by zero:

        .. code-block::

            n / 0  =  NaN   : if n == 0  (represented by np.nan)
                   = -Inf   : if n == 0  (represented by -np.inf)

    .. note::

        Generic ``**kwargs`` are taken as input and fed directly to ``NseScore``
        to avoid errors due to repetition.  Checks are performed in ``NseScore``
        construction to assert compatiblity.

    .. note::

        if using dask, no computation should have happened until
        ``.compute(...)`` is called (similar to any other score).
    """

    # -------------------------------------------------------------------------
    # TODO: unify this into a utility function
    # -------------------------------------------------------------------------
    # check if all input xarray are same type, mixed types are not allowed
    all_same_xrlike = all_same_xarraylike([fcst, obs])
    if weights is not None:
        all_same_xrlike = all_same_xarraylike([fcst, obs, weights])
    if not all_same_xrlike:
        raise TypeError(NseUtils.ERROR_MIXED_XR_DATA_TYPES)

    # retain original xarray type: all types same so only need to check one.
    is_dataarray: bool = isinstance(fcst, xr.DataArray)
    use_dummyname: bool = False

    # we need to use dummyname appropriately when lifting an array to a
    # dataset, while operations still work with dataarrays with different name,
    # they don't with datasets - so we need to mimic this behaviour.
    if is_dataarray:
        use_dummyname = dataarray_use_dummyname(fcst, obs, weights)

    # lift inputs to dataset
    ds_fcst = NseUtils.lift_to_ds(fcst, use_dummyname)
    ds_obs = NseUtils.lift_to_ds(obs, use_dummyname)
    ds_weights = NseUtils.lift_to_ds(weights, use_dummyname)
    # -------------------------------------------------------------------------

    # initialize score - implicitly also runs several checks.
    nse_score: NseScore = NseScore(
        fcst=ds_fcst,
        obs=ds_obs,
        weights=ds_weights,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        is_angular=is_angular,
    )

    # -------------------------------------------------------------------------
    # TODO: unify this into a utility function
    # -------------------------------------------------------------------------
    res_ds: xr.Dataset = nse_score.nse
    # revert name and type was originally a dataarray
    if is_dataarray:
        res_da: xr.DataArray = res.to_dataarray()

        if use_dummyname:
            res_da.name = None

        # safety: assert final return type
        assert isinstance(res_da, xr.DataArray)
        return res_da

    # safety: assert final return type
    assert isinstance(res_ds, xr.Dataset)
    return res_ds
    # -------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class NseScore:
    """
    Helper structure used to perform checks on data during initialization.

    Freezes data post initialization.
    (this is a safety feature that only stops shallow assignment, it cannot
    prevent mutable properties from being changed.)
    """

    #: forecast/prediction/simulation/model
    fcst: xr.Dataset
    #: observations
    obs: xr.Dataset
    #: weighting
    weights: xr.Dataset | None = None
    #: dimensions to reduce
    reduce_dims: FlexibleDimensionTypes | None = None
    #: dimensions to preserve (mutually exlcusive to reduce_dims)
    preserved_dims: FlexibleDimensionTypes | None = None
    #: whether the input data is angular, used implicitly by `mse`
    is_angular: bool = False

    def __post_init__(self):
        """
        Run checks before creating final object
            - runtime type assertions
            - check weights
            - dimension checks
        """
        # safety: dev/testing: runtime assertions
        assert isinstance(fcst, xr.Dataset)
        assert isinstance(obs, xr.Dataset)
        assert is_flexibledimensiontypes(preserve_dims)
        assert is_flexibledimensiontypes(reduce_dims)

        if weights is not None:
            assert isinstance(weights, xr.Dataset)
            check_weights(self.weights)

        self.reduce_dims = NseUtils.check_and_gather_dimensions(
            fcst=fcst,
            obs=obs,
            weights=weights,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

    @property
    def nse(self) -> xr.Dataset:
        """
        Actual ``nse`` computation happens here.

        Divide by zero error is ignored as it will be already raised by call to
        ``obs_variance`` computation.
        """
        with np.errstate(divide="ignore"):
            ret_nse: xr.Dataset = 1.0 - (self.fcst_error / self.obs_variance)
            return ret_nse

    @property
    def fcst_error(self) -> XarrayLike:
        """
        Calculates fcst error, essentially : ``scores.mse(fcst, obs)``
        """
        return self._mse_callback(self.fcst, self.obs)

    @property
    def obs_variance(self) -> XarrayLike:
        """
        Calculates obs variance, essentially: ``scores.mse(obs_mean, obs)``
        """
        # compute mean along reduce dims - this does not need to be weighted
        obs_mean: XarrayLike = self.obs.mean(dim=self.reduce_dims)
        # using `mse` since its compatible with weights and angular data
        obs_var: XarrayLike = self._mse_callback(obs_mean, self.obs)
        # check for 0 obs variance: divide by zero condition
        NseUtils.check_nonzero_obsvar(obs_var)

        return obs_var

    @property
    def _mse_callback(self):
        return functools.partial(
            scores.continuous.mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )


class NseUtils:
    """
    Helper class with static methods for use in NSE computations only. Also
    contains error msg strings specific to NSE.

    .. note::

        These utility functions are internal to this module and hence are not as
        strict with type asserts, since the underlying utility functions
        themselves are responsible for creating some of these asserts.

        The only constraint it assumes is that any ``XarrayLike`` is transformed
        (lifted) to a xarray dataset; and all methods in this namespace is
        compatible only with datasets. Hence, ``NseUtils.lift_to_ds`` should be
        used prior using methods in this class.
    """

    ERROR_NO_DIMS_TO_REDUCE: str = """
    NSE needs at least one dimension to be reduced. Check that `preserve_dims`
    is not preserving all dimensions, OR check that `reduce_dims` is specified
    with at least one dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = """
    NSE needs at least one dimension that is being reduced where its length > 1.
    Check that the input provided have at least 2 or more data points along the
    dimensions being reduced. It is not possible to calculate a non-zero obs
    variance from 1 point.
    """

    ERROR_MIXED_XR_DATA_TYPES: str = """
    Triggered during NSE calculations` check `fcst`, `obs` and `weights` are of
    the same type `xr.Dataset` OR `xr.DataArray` EXCLUSIVELY; NOT a mix of the
    two types.
    """

    WARN_ZERO_OBS_VARIANCE: str = """
    Possible divide by zero: at least one element in the reduced obs variance
    array is 0. Any divide by zero entries will be filled in as `np.nan` if the
    forecast error is also 0, otherwise it will be `-np.inf`. This is so that
    any other valid entries are still computed and returned. The user should
    still verify that zero obs variance is expected for the given input data.
    """

    _DATAARRAY_TEMPORARY_NAME: str = "__NONAME"

    # TODO: this needs refactoring:
    @staticmethod
    def dataarray_use_dummyname(*da) -> bool:
        """
        Rules for NOT using dummy name:
           - all ``da`` have the same name AND
           - their name is not ``None`` or empty

        Alternatively,

        Rules for using a dummy name:
            - any ``da`` name is ``None`` or empty OR
            - any one name is different from the others

        """
        # safety: dev/testing: at least one variadic arg needed
        assert len(da) > 0

        fn_has_noname: Callable = lambda _da: _da.name is None or not _da.name
        any_noname: bool = any(map(fn_has_noname, da))
        any_diffname: bool = False
        refname: Hashable = da[0].name

        for _x in da:
            if _x != refname:
                any_diffname = True

        # same as: `all_samename` and `all_hasname`
        return any_diffname or any_noname

    # TODO: this needs refactoring:
    @staticmethod
    def lift_to_ds(xr_data: XarrayLike, use_dummyname: bool = True) -> xr.Dataset:
        """
        Consolidates ``XarrayLike`` - a union type - to a xarray dataset, in
        order to simplify API calls. Other utility functions assume that they
        are dealing with datasets.

        Returns:
            A tuple containing the lifted dataset and whether or not a dummy
            name was assigned during the lift.
        """
        # safety: dev/testing
        assert is_xarraylike(xr_data)

        if isinstance(xr_data, xr.Dataarray):
            if use_dummyname:
                ret_ds = xr_data.to_dataset(name=NseUtils._DATAARRAY_TEMPORARY_NAME)
            else:
                ret_ds = xr_data.to_dataset()

        # safety: dev/testing: can only be a dataset if we reached here
        assert isinstance(ret_ds, xr.Dataset)

        return (ret_ds, dummy_name)

    @staticmethod
    def check_and_gather_dimensions(
        fcst: xr.Dataset,
        obs: xr.Dataset,
        weights: xr.Dataset | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> FlexibleDimensionTypes:
        """
        Checks the dimension compatibilty of the various input arguments.
        """
        weights_dims = None if weights is None else weights.ds.dims

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=fcst.dims,
            obs_dims=obs.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        merged_sizes: dict[Hashable, int] = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension - required to
        # calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def merge_sizes(*ds) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key)
        in the given ``xr.Dataset`` object(s).

        Args:
            *ds: Variadic argument of each of type ``xr.Dataset``
        """

        def _merge_single(
            acc_sizes: dict[Hashable, int],
            curr_ds: xr.Dataset | None,
        ) -> dict[Hashable, int]:
            """
            merges ``sizes`` attribute from each dataset. ``sizes`` is a
            mapping: dimension name -> length. In theory this should also be
            compatible with data arrays.
            """
            if curr_ds is None:
                return acc_sizes
            merged_sizes = acc_sizes | curr_ds.sizes
            return merged_sizes

        return functools.reduce(_merge_single, ds, {})

    @staticmethod
    def check_nonzero_obsvar(obs_var: xr.Dataset):
        """
        Warns if at least one obs variance term is zero in any variable in the dataset.
        """
        # Array of True/False singletons for each variable in dataset
        ds_datavar_has_allzeros: xr.Dataset = (ref_ds == 0).any()
        # Combine and check if ANY element in the ENTIRE dataset is zero
        any_elem_is_zero = any(ds_datavar_has_allzeros.values())
        # At least one zero elem => warn
        if any_elem_is_zero:
            warnings.warn(NseUtils.WARN_ZERO_OBS_VARIANCE)

    @staticmethod
    def check_gathered_dims(
        gathered_dims: FlexibleDimensionTypes,
        dim_sizes: dict[Hashable, int],
    ):
        """
        Checks that gathered dimensions has at least one entry (key or hash) AND at least one of the
        gathered dimensions has more than 1 data point.

        (i.e. there exists some ``key`` in ``gathered_dims`` where ``dim_sizes[key] > 1``.)

        Args:
            gathered_dims: set of dimension names.
            dim_sizes: dictionary containing a map between dimension names and their lengths.

        .. note::

            An empty set is normally allowed in scores, however, NSE in particular requires at
            least one dimension being reduced to have more than one data point in order to compute a
            non-zero obs variance.

        .. see-also::

            For more info on the concept of ``sizes`` see :py:meth:`xarray.Dataset.sizes` or
            :py:meth:`xarray.DataArray.sizes`
        """
        if len(list(gathered_dims)) == 0:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)

        dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

        if not dim_has_more_than_one_obs:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_WITH_MULTIPLE_OBS)
