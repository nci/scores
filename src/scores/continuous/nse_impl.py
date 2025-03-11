"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import functools
import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from typing import NamedTuple

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
    *,  # force kwargs only
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
        Same as the public API, but enforced to be kwarg only. Uses ``NseScore`` class
        for doing the thorough checking.

        Lifts any ``XarrayLike`` to ``xr.Datasets`` so that only one type of API needs to
        be maintained, rather than also dealing with ``xr.DataArray``.  Converts it back
        to its original data type in the end (if it was a ``xr.DataArray``)

    Returns:
        An ``NseScore`` object. Its ``.nse`` property can be called to run the
        computations. (Also see note below if using ``dask``.)

    Raises:
        See public API docstring

    .. important::

        Divide by zero is ALLOWED - to accomodate scenarios where all obs entries in the
        group being reduced is constant (0 obs variance).

        While these may cause divide by zero errors, they should not halt execution of
        computations for other valid coordinates - so a warning is issued instead to
        prompt the user to double check the data.

        It may also be that divide by zero is unavoidable - in which case we still want
        to return the correctly calculated values. To this end, this is how ``numpy``
        resolves divide by zero:

        .. code-block::

            n / 0  =  NaN   : if n == 0  (represented by np.nan)
                   = -Inf   : if n == 0  (represented by -np.inf)

    .. note::

        if using dask, no computation should have happened until ``.compute(...)`` is
        called (similar to any other score).
    """
    # lift inputs to a meta dataset dataset see: NseUtils.make_metadataset and
    # NseTypes.MetaDataset = tuple(tuple[xr.Dataset], is_dataarray, is_dummyname)
    metads: NseMetaDataset = NseUtils.make_metadataset(fcst, obs, weights)

    # initialize score - implicitly also runs several checks.
    ret_score: NseScore = NseScore(
        fcst=metads.datasets.fcst,
        obs=metads.datasets.obs,
        weights=metads.datasets.weights,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        is_angular=is_angular,
    )

    result: xr.Dataset = ret_score.nse

    if metads.is_dataarray:
        # demote if originally a data array, must have only one key if this is the case
        da_keys = list(result.data_vars.keys())
        if len(da_keys) != 1:
            raise RuntimeError(NseUtils.ERROR_DATAARRAY_NOT_MAPPED_TO_SINGLE_KEY)
        result = result.data_vars[da_keys[0]]
        # undo dummy name
        if metads.is_dummyname:
            result.name = None

    # safety: assert xarraylike before returning
    assert is_xarraylike(result)

    return result


@dataclass(kw_only=True)
class NseScore:
    """
    Helper structure used to perform checks on data during initialization.
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
    preserve_dims: FlexibleDimensionTypes | None = None
    #: whether the input data is angular, used implicitly by `mse`
    is_angular: bool = False

    def __post_init__(self):
        """
        Run checks before creating final object
            - runtime type assertions
            - check weights
            - dimension checks
        """
        # safety: dev/testing: runtime assertions on arguments
        assert (
            isinstance(self.fcst, xr.Dataset)
            and isinstance(self.obs, xr.Dataset)
            # These are optionals (i.e. skipped if None):
            and (self.weights is None or isinstance(self.weights, xr.Dataset))
            and (self.reduce_dims is None or is_flexibledimensiontypes(self.reduce_dims))
            and (self.preserve_dims is None or is_flexibledimensiontypes(self.preserve_dims))
            and isinstance(self.is_angular, bool)
        )

        # check weights if its provided
        if self.weights is not None:
            check_weights(self.weights)

        # do any dimension checks and store the final reduced (gathered) dimensions
        self.reduce_dims = NseUtils.check_and_gather_dimensions(
            fcst=self.fcst,
            obs=self.obs,
            weights=self.weights,
            reduce_dims=self.reduce_dims,
            preserve_dims=self.preserve_dims,
        )

    # The following methods are defined as "properties", since a NseScore object should
    # not be mutable so any computation should also only act as a "getter"

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


class NseDatasets(NamedTuple):
    """
    Namespace for storing dataset information for NSE
    """

    fcst: xr.Dataset
    obs: xr.Dataset
    weights: xr.Dataset | None


class NseMetaDataset(NamedTuple):
    """
    Namespace that wraps NseDatasets with metadata about the underlying datasets
    particularly:
        is_dataarray: whether the underlying datasets were originally data arrays
        is_dummyname: whether the underlying dataarrays were provided dummy names
        datasets: the NseDatasets being wrapped
    """

    datasets: NseDatasets
    is_dataarray: bool
    is_dummyname: bool


class NseUtils:
    """
    Helper class with static methods for use in NSE computations only. Also contains
    error msg strings specific to NSE.

    .. important::

        To simplify checks, all checks are assumed to be compatible with ``xr.Dataset``
        only. The helper method ``NseUtils.lift_to_ds`` can be used to transform the
        input xarraylike into a dataset, returning any metadata associated with the
        original datatype along with the promoted dataset.

        The utility module is also not very strict with runtime type assertions (as this
        would increase code bloat - and potential circular logic) and assumes that this
        is performed at a higher level at the earliest point of the call chain before
        computing the sore.
    """

    ERROR_DATAARRAY_NOT_MAPPED_TO_SINGLE_KEY: str = """
    The underlying data array type should only be represented by a single key in its
    dataset form. Either no keys or multiple keys detected. This is NOT EXPECTED.

    Please raise an issue on github.com/nci/scores citing this error.
    """

    ERROR_NO_DIMS_TO_REDUCE: str = """
    NSE needs at least one dimension to be reduced. Check that `preserve_dims` is not
    preserving all dimensions, OR check that `reduce_dims` is specified with at least one
    dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = """
    NSE needs at least one dimension that is being reduced where its length > 1.  Check
    that the input provided have at least 2 or more data points along the dimensions
    being reduced. It is not possible to calculate a non-zero obs variance from 1 point.
    """

    ERROR_MIXED_XR_DATA_TYPES: str = """
    Triggered during NSE calculations` check `fcst`, `obs` and `weights` are of the same
    type `xr.Dataset` OR `xr.DataArray` EXCLUSIVELY; NOT a mix of the two types.
    """

    WARN_ZERO_OBS_VARIANCE: str = """
    Possible divide by zero: at least one element in the reduced obs variance array is 0.
    Any divide by zero entries will be filled in as `np.nan` if the forecast error is
    also 0, otherwise it will be `-np.inf`. This is so that any other valid entries are
    still computed and returned. The user should still verify that zero obs variance is
    expected for the given input data.
    """

    #: datasets require a name - used when promoting a dataarray without a name.
    #: NOTE: it is FAIRLY COMMON that a dataarray DOES NOT have a name.
    _DATAARRAY_TEMPORARY_NAME: str = "__NONAME"

    @staticmethod
    def check_all_same_type(*xrlike):
        """
        Checks that all XarrayLike inputs are of the same type, i.e. either ALL datasets
        OR (exclusive OR i.e. xor) ALL dataarrays.

        Raises:
            TypeError: If mixed types are provided or if input isn't XarrayLike.
        """
        assert len(xrlike) > 0

        xrlike_remove_none = [_x for _x in xrlike if _x is not None]

        if not all_same_xarraylike(xrlike_remove_none):
            raise TypeError(NseUtils.ERROR_MIXED_XR_DATA_TYPES)

    @staticmethod
    def make_metadataset(fcst, obs, weights) -> NseMetaDataset:
        """
        Consolidates ``XarrayLike`` - a union type - to a xarray dataset, in order to
        simplify API calls. Other utility functions assume that they are dealing with
        datasets.

        Returns:
            A tuple containing the lifted dataset, whether it was originally a dataarray,
            and whether or not a dummy name was assigned.

        .. note::

            FUTUREWORK: this should eventually be some sort of higher order structure,
                        common to other scores.
        """
        NseUtils.check_all_same_type(*[fcst, obs, weights])

        # use fcst as arbitrary reference name
        ref_name = fcst.name if isinstance(fcst, xr.DataArray) else None
        is_dataarray = False
        is_dummyname = False

        for xrlike in [fcst, obs, weights]:
            if xrlike is not None and isinstance(xrlike, xr.DataArray):
                is_dataarray = True
                if xrlike.name is None or ref_name != xrlike.name:
                    is_dummyname = True

        def _to_ds(_xrlike: XarrayLike) -> xr.Dataset:
            """
            helper to lift the input xarraylike depending on whether it is a data array
            and whether it needs a dummy name.
            """
            # assume dataset as default
            _ret = _xrlike
            # otherwise if its a dataarray, promote it to a dataset, and give it a dummy
            # name if needed
            if is_dataarray:
                assert isinstance(_ret, xr.DataArray)
                if is_dummyname:
                    return _ret.to_dataset(name=NseUtils._DATAARRAY_TEMPORARY_NAME)
                else:
                    return _ret.to_dataset()
            return _ret

        return NseMetaDataset(
            datasets=NseDatasets(
                fcst=_to_ds(fcst),
                obs=_to_ds(obs),
                weights=_to_ds(weights) if weights is not None else None,
            ),
            is_dataarray=is_dataarray,
            is_dummyname=is_dummyname,
        )

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
        weights_dims = None if weights is None else weights.dims

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=fcst.dims,
            obs_dims=obs.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        merged_sizes: dict[Hashable, int] = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension
        # - required to calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def merge_sizes(*ds) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key) in the
        given ``xr.Dataset`` object(s).

        Args:
            *ds: Variadic argument of each of type ``xr.Dataset``
        """
        assert len(ds) > 0

        def _merge_single(
            acc_sizes: dict[Hashable, int],
            curr_ds: xr.Dataset | None,
        ) -> dict[Hashable, int]:
            """
            merges ``sizes`` attribute from each dataset. ``sizes`` is a mapping:
            dimension name -> length. In theory this should also be compatible with data
            arrays.
            """
            if curr_ds is None:
                return acc_sizes
            merged_sizes: dict[Hashable, int] = acc_sizes | dict(curr_ds.sizes)
            return merged_sizes

        ret_sizes: dict[Hashable, int] = functools.reduce(_merge_single, ds, {})

        return ret_sizes

    @staticmethod
    def check_nonzero_obsvar(obs_var: xr.Dataset):
        """
        Warns if at least one obs variance term is zero in any variable in the dataset.
        """
        # Array of True/False singletons for each variable in dataset
        any_ds_var_has_all_zeros: xr.Dataset = (obs_var == 0).any()
        # Combine and check if ANY element in the ENTIRE dataset is zero
        any_elem_is_zero = any(any_ds_var_has_all_zeros.values())
        # At least one zero elem => warn
        if any_elem_is_zero:
            warnings.warn(NseUtils.WARN_ZERO_OBS_VARIANCE)

    @staticmethod
    def check_gathered_dims(
        gathered_dims: FlexibleDimensionTypes,
        dim_sizes: dict[Hashable, int],
    ):
        """
        Checks that gathered dimensions has at least one entry (key or hash) AND at least
        one of the gathered dimensions has more than 1 data point.

        (i.e. there exists a ``key`` in ``gathered_dims`` where ``dim_sizes[key] > 1``.)

        Args:
            gathered_dims: set of dimension names.
            dim_sizes: map between dimension names and their lengths.

        Raises:
            DimensionError: if gathered dimensions are not compatible

        .. note::

            An empty set of dimensions is normally allowed in scores, i.e. the score is
            computed for each element.

            This is not suitable for NSE, which requires calculation of variance, without
            more than 1 data point in each reduced group, variance is guarenteed to be
            zero for all entries.

        .. note::

            Commonly used in conjunction with :py:meth:`NseUtils.merge_sizes`

        .. see-also::

            For more info on the concept of ``sizes`` see :py:meth:`xarray.Dataset.sizes`
            or :py:meth:`xarray.DataArray.sizes`
        """
        if len(list(gathered_dims)) == 0:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)

        dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

        if not dim_has_more_than_one_obs:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_WITH_MULTIPLE_OBS)
