"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import functools
import warnings
import weakref
from enum import Enum
from dataclasses import KW_ONLY, dataclass
from typing import Callable, Hashable, Iterable, cast

import numpy as np
import scores.continuous as _cnt
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import check_dims, check_weights_positive, gather_dimensions


# --------------------------------------------------------------------------------------------------
# TODO: this can probably go into common utils at some point
class NseXarrayType(Enum):
    """
    Xarray type resolver - used to convert
    """

    #: invalid type 
    INVALID = -1
    #: maps to ``xr.Dataset``
    DATASET = 1
    #: maps to ``xr.DataArray``
    DATAARRAY = 2


@dataclass(frozen=True, kw_only=True, slots=True, init=False)
class LiftedDataset:
    """
    Higher order data type that lifts a data array into a dataset, this way it is SUFFICIENT for
    functions to ONLY be compatible with datasets, even if a data array is provided as input.

    .. important::

        There may be cases where the functions using this type may have to iteratively perform
        operations on the underlying data arrays in the dataset, because no native implementation
        exists for ``xr.Dataset`` (but it does for ``xr.DataArray``). However, this needs to be done
        anyway if we are supporting both types.

        This class exists as an "aid" to avoid repeated logic and branching.
    """
    ds: xr.Dataset
    raw_type: NseXarrayType
    var_names: list[str]

    #: data arrays don't necessarily have variable names so give it a ascii only dummy
    _DUMMY_DATAARRAY_VARNAME: str = "dummyvarname"

    def __init__(self, xr_data: XarrayLike):
        if isinstance(xr_data, xr.Dataset):
            self.ds = xr_data
            self.raw_type = NseXarrayType.DATASET
            self.var_names = [ k for k in xr_data.variables.keys() ]
        elif isinstance(xr_data, xr.Dataset):
            # a data array could have a name...
            da_name: str = (
                LiftedDataset._DUMMY_DATAARRAY_VARNAME
                if xr_data.name is None else xr_data.name
            )
            # but a dataset MUST have a name for its variables
            self.ds = xr_data.to_dataset(name=da_name)
            self.raw_type = NseXarrayType.DATAARRAY
            self.var_names = [da_name]
        # if we reached this point, then we're dealing with an illegal runtime type
        self.raw_type = NseXarrayType.INVALID
        raise TypeError("Invalid type for `XarrayLike`, must be a `xr.Dataset` or a `xr.DataArray`")

# --------------------------------------------------------------------------------------------------

class NseUtils:
    """
    Helper class with static methods for use in NSE computations only.
    """

    ERROR_SCORE_DIRECT_INIT_DISALLOWED: str = """
    Internal Class: `NseScore` is being directly initialised - this is disallowed, use
    `NseScoreBuilder` instead. This is not a user error - if it has been triggered please raise an
    issue in github.
    """

    ERROR_SCORE_ALREADY_BUILT: str = """
    Internal Class: `NseScore` has already been built - re-use the output of this builder instead
    of rebuilding, to avoid data bloat. This is not a user error - if it has been triggered please
    raise an issue in github.
    """

    ERROR_MIXING_DATASET_AND_DATAARRAY: str = """
    Mixing of xarray datasets and data arrays is not allowed for this score. ALL of `fcst`, `obs`
    and `weights` MUST be either `xr.Dataset` or `xr.DataArray` EXCLUSIVELY, AND MUST have common
    variables.
    """

    ERROR_NO_DIMS_TO_REDUCE: str = """
    NSE needs at least one dimension to be reduced. Check that `preserve_dims` is not preserving all
    dimensions, OR check that `reduce_dims` is specified with at least one dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = """
    NSE needs at least one dimension that is being reduced where its length > 1. Check that the
    input provided have at least 2 or more data points along the dimensions being reduced. It is not
    possible to calculate a non-zero obs variance from 1 point.
    """

    WARN_ZERO_OBS_VARIANCE: str = """
    Possible divide by zero: at least one element in the reduced obs variance array is 0. Any divide
    by zero entries will be filled in as `np.nan` if the forecast error is also 0, otherwise it will
    be `-np.inf`. This is so that any other valid entries are still computed and returned. The user
    should still verify that zero obs variance is expected for the given input data.
    """

    @staticmethod
    def preprocess_nse_inputs(
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> FlexibleDimensionTypes:
        """
        Checks the dimension compatibilty of the various input arguments. This is the main utility
        function that groups all the checks needed for NSE.
        """
        # lift data arrays to datasets
        if isinstance(fcst, xr.Dataset):
            mixed_types = not isinstance(obs, xr.Dataset)
            mixed_types = mixed_types or not isinstance(weights, xr.Dataset)
            raise TypeError(ERROR_MIXING_DATASET_AND_DATAARRAY)

        if isinstance(fcst, xr.DataArray):
            mixed_types = not isinstance(obs, xr.DataArray)
            mixed_types = mixed_types or not isinstance(weights, xr.DataArray)
            raise TypeError(ERROR_MIXING_DATASET_AND_DATAARRAY)

        # check weights conform to NSE computations
        NseUtils.check_weights(weights)
        weights_dims: Iterable[Hashable] | None = None if weights is None else weights.dims

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=fcst.dims,
            obs_dims=obs.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        merged_sizes: dict[Hashable, int] = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension - required to calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def check_xarray_type_homogeneity(xr_data_list: tuple[XarrayLike]):
        functools.reduce(xr_data_list)

    @staticmethod
    def lift_dataarrays_to_datasets(da_list: tuple[xr.Dataset]) -> ds_list:

    @staticmethod
    def merge_sizes(*xr_data) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key) (key) in a given
        `XarrayLike` object.
        """
        all_sizes: list[dict[Hashable, int]] = [dict(_x.sizes) for _x in xr_data if _x is not None]
        ret_sizes: dict[Hashable, int] = functools.reduce(lambda _acc, _x: _acc | _x, all_sizes)
        return ret_sizes

    @staticmethod
    def check_weights(weights: XarrayLike | None) -> None:
        """
        Wrapper around :py:func:`~scores.utils.check_weights_positive`, with a context specific
        warning message.
        """
        check_weights_positive(weights, context="`NSE` must have positive weights.")

    @staticmethod
    def check_nonzero_obsvar(obs_var: XarrayLike) -> None:
        """
        Warns if at least one obs variance term is zero.
        """
        if (obs_var == 0).any():
            import pdb; pdb.set_trace()
            warnings.warn(NseUtils.WARN_ZERO_OBS_VARIANCE)

    @staticmethod
    def check_gathered_dims(gathered_dims: FlexibleDimensionTypes, dim_sizes: dict[Hashable, int]) -> None:
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
            raise KeyError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)

        dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

        if not dim_has_more_than_one_obs:
            raise IndexError(NseUtils.ERROR_NO_DIMS_WITH_MULTIPLE_OBS)


@dataclass
class NseScoreBuilder:
    """
    Internal class that builds the `NseScore` object which is used to compute the NSE score.

    .. important::

        DEVELOPER-NOTE: This is the class you want to use if you need to instantiate the
        ``NseScore`` object for further computations NOT ``NseScore`` directly.
            - The builder focuses on the accuracy of the interface (i.e. the input arguments).
            - The score focuses on the accuracy of the computation assuming accuracy of the inputs.

    The scope of this class is to:
        - perform checks on input arrays and their dimensions
        - perform checks on weights
        - resolves dimensions to be reduced
        - raise any issues if the above are not conformant with the "NSE" score

    Raises:
        RuntimeError: If the builder has already expired after building a score - see notes below.

    .. see-also::
        All other errors raised are documented in the main api under :py:func:`scores.continous.nse`

    .. note::

        No data or reference to data is kept in the builder object - this is intentional. Normally
        the builder pattern _should_ self-destruct after building, and move all its resources to the
        built object, but this is not guarenteed with python's garbage collector.

        Therefore, we retain a weak reference to ``ref_nsescore`` just in case an instantiation of
        the builder is trying to create another score - and raise an error if it does.

        However, if the ``ref_nsescore`` has been garbage collected, the builder is free to rebuild
        - although the usecase for this is almost non-existant.
    """

    #: weakref because builder shouldn't circularly reference itself via ref_nsescore.ref_builder
    ref_nsescore: weakref.ref | None = None

    # NOTE: no explicit __init__ of other members since this object shouldn't store any data
    # (neither copy nor reference).

    def build(
        self,
        *,  # force KW_ONLY - note
        fcst: XarrayLike,
        obs: XarrayLike,
        reduce_dims: FlexibleDimensionTypes | None = None,
        preserve_dims: FlexibleDimensionTypes | None = None,
        weights: XarrayLike | None = None,
        is_angular: bool | None = False,
    ):
        """
        Builds an ``NseScore`` object.

        .. note::

            The ``RuntimeError`` raised would be an internal error is a development safety so that
            `NseScore` isn't instantiated directly. If this error is raised, the user would be
            prompted to raise an issue.
        """
        # it is unlikely that a single instantiation of a builder is used to create multiple scores,
        # simultaneously as they refer to the same data and arguments. If build is invoked on an
        # already built score from this builder -  it's likely that this internal class was used
        # and/or developed incorrectly - hence the runtime error.
        if self.ref_nsescore is not None and self.ref_nsescore() is not None:
            raise RuntimeError(NseUtils.ERROR_SCORE_ALREADY_BUILT)

        reduce_dims = NseUtils.preprocess_nse_inputs(
            fcst=fcst,
            obs=obs,
            weights=weights,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        ret_nsescore = NseScore(
            ref_builder=self,
            fcst=fcst,
            obs=obs,
            reduce_dims=reduce_dims,
            weights=weights,
            is_angular=is_angular,
        )

        self.ref_nsescore = weakref.ref(ret_nsescore)

        return ret_nsescore


# fields can only be set once
@dataclass(frozen=True)
class NseScore:
    """
    Internal class that performs the actual scoring logic.

    Assumes that all setups and checks have been done by :py:class:`NseScoreBuilder`, and hence should
    not be initialised directly. Instead, ``NseScoreBuilder`` should be used to do any setup.

    .. important::

        DEVELOPER-NOTE: DO NOT USE THIS CLASS TO INSTANTIATE ``NseScore``. This is only meant to be
        instantiated by the ``NseScoreBuilder`.`
            - The builder focuses on the accuracy of the interface (i.e. the input arguments).
            - The score focuses on the accuracy of the computation assuming accuracy of the inputs.

    .. note::

        Most of the methods in this class are cached properties, because they only need to be
        computed one time per score object.

    Raises:
        RuntimeError: If the ``NseScore`` object is being initialised directly. ``NseScoreBuilder``
            should be used instead as it performs input checking

    .. see-also::

        :py:func:`scores.continous.nse` for the full documentation of the score itself and any the
        full description of error states.

    .. important::

        divide by zero errors are intentionally ignored here and ``numpy`` will automatically fill
        them with ``np.nan``, or ``-np.inf``. ``NseScoreBuilder`` the class that constructs
        a ``NseScore``
        object, already handles warnings during these scenarios.
    """

    _: KW_ONLY
    #: strong ref to builder, since we want to keep the builder in scope to check how this object
    #: was created - see ``__post_init__``
    ref_builder: NseScoreBuilder
    obs: XarrayLike
    fcst: XarrayLike
    reduce_dims: FlexibleDimensionTypes | None
    weights: XarrayLike | None
    is_angular: bool | None = False

    def __post_init__(self) -> None:
        """
        Checks that this object has been initialised by a valid ``NseScoreBuilder``.

        .. note::

            The ``RuntimeError`` raised would be an internal error is a development safety so that
            `NseScore` isn't instantiated directly. If this error is raised, the user would be
            prompted to raise an issue.
        """
        if self.ref_builder is None or not isinstance(self.ref_builder, NseScoreBuilder):
            raise RuntimeError(NseUtils.ERROR_SCORE_DIRECT_INIT_DISALLOWED)

    @functools.cached_property
    def mse_callback(self) -> Callable[..., XarrayLike]:
        """
        Callback helper for calculating :py:func:`scores.mse`, with prefilled keyword args.
        """
        return functools.partial(
            _cnt.mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    @functools.cached_property
    def nse(self) -> XarrayLike:
        """
        The main calculation function for NSE:
        - denominator (D): calculates the obs variance using a combination of ``xr.mean`` and :py:func:`scores.mse`
        - numerator (N): calculates the forecast error using :py:func:`scores.mse` on ``obs`` and ``fcst``.
        - ``score = 1 - N / D`` reduced and broadcast appropriately by ``xarray``.
        """
        ret_nse: XarrayLike | None = None

        # intentional divide="ignore", as warning is already raised above.
        # `numpy` will fill divide by zero elements with NaNs.
        with np.errstate(divide="ignore"):
            ret_nse = 1.0 - (self.fcst_error / self.obs_variance)

        # cast strictly to `XarrayLike` from an `Optional`. If we got this far, it can't be `None`.
        cast(XarrayLike, ret_nse)

        return ret_nse

    @functools.cached_property
    def fcst_error(self) -> XarrayLike:
        """
        Calculates the forecast error which is essentially ``scores.mse(fcst, obs)``
        """
        return self.mse_callback(self.fcst, self.obs)

    @functools.cached_property
    def obs_variance(self) -> XarrayLike:
        """
        Calculates the obs variance which is essentially ``scores.mse(obs_mean, obs)``.
        """
        reduce_dims: Iterable[Hashable] = [] if self.reduce_dims is None else self.reduce_dims
        check_dims(self.obs, reduce_dims, mode="superset")

        obs_mean: XarrayLike = self.obs.mean(dim=self.reduce_dims)
        ret_obs_var: XarrayLike = self.mse_callback(obs_mean, self.obs)

        NseUtils.check_nonzero_obsvar(ret_obs_var)

        return ret_obs_var
