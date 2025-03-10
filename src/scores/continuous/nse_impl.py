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
from scores.typing import FlexibleDimensionTypes, XarrayLike
from scores.utils import DimensionError, check_weights, gather_dimensions


NseInputData = TypedDict(
    "NseInputData",
    obs=XarrayLike,
    fcst=XarrayLike,
    reduce_dims=FlexibleDimensionTypes,
    weights=XarrayLike | None,
    is_angular=bool,
)


def nse_impl_nse(score_data: NseScoreData) -> XarrayLike:
    """
    Computes the ``nse`` score.
        - computes the forecast error
        - computes the observation variance
        - checks for divide by zero errors

    See main API docs under :py:func:`scores.continuous.standard_impl.nseq
    """
    pass


def _nse_build_frozen_scoredata() -> frozenset
    """
    Creates a frozenset containing the score data as a "immutable" (*) structure. Without needing to
    re-do the computation for each function.
    """

def _nse_impl_obs_variance(nse_scoredata: NseScoreData) -> XarrayLike:
    """
    Calculates the observation variance which is essentially ``scores.mse(obs_mean, obs)``
    """
    return _nse_impl_mse_callback(self.fcst, self.obs)


def _nse_impl_fcst_error(nse_scoredata: NseScoreData) -> XarrayLike:
    """
    Calculates the forecast error which is essentially ``scores.mse(fcst, obs)``
    """
    return _nse_impl_mse_callback(self.fcst, self.obs)


class NseScoreData:
    """
    Internal class used to pass around score data for nse related functions.
        - does checks during initialisation see ``__post_init__`` so that these do not need to
          re-checked between functions.
        - Contains a "static"(*) mse callback helper pre-filled with initialized data.

    * static in that it won't change even if data members are modified
    """
    __slots__ =  ("obs", "fcst", "reduce_dims", "weights", "is_angular", "mse_callback")

    def __init__(
        self,
        * # force kwarg only
        obs: XarrayLike,
        fcst: XarrayLike,
        reduce_dims: FlexibleDimensionTypes,
        preserve_dims: FlexibleDimensionTypes,
        weights: Xarraylike | None=None,
        is_angular: bool=None,
    ):
        # check weights
        if weights is not None:
            check_weights(weights)

        # check and get effective dimensions to reduce
        reduce_dims = NseUtils.check_and_gather_dimensions(
            fcst=fcst,
            obs=obs,
            weights=weights,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        # initialize data members
        self.obs: XarrayLike = obs
        self.fcst: XarrayLike = fcst 
        self.weights: XarrayLike = weights
        self.reduce_dims: FlexibleDimensionTypes = reduce_dims
        self.is_angular: bool = is_angular
        self.mse_callback = functools.partial(

        )


class NseUtils:
    """
    Helper class with static methods for use in NSE computations only. Also contains error msg
    strings specific to NSE.
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

    ERROR_MIXED_XR_DATA_TYPES: str = """
    Triggered during NSE calculations` check `fcst`, `obs` and `weights` are of the same type
    `xr.Dataset` OR `xr.DataArray` EXCLUSIVELY; NOT a mix of the two types.
    """

    WARN_ZERO_OBS_VARIANCE: str = """
    Possible divide by zero: at least one element in the reduced obs variance array is 0. Any divide
    by zero entries will be filled in as `np.nan` if the forecast error is also 0, otherwise it will
    be `-np.inf`. This is so that any other valid entries are still computed and returned. The user
    should still verify that zero obs variance is expected for the given input data.
    """

    @staticmethod
    def check_and_gather_dimensions(
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: LiftedDataset | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> FlexibleDimensionTypes:
        """
        Checks the dimension compatibilty of the various input arguments.
        """
        weights_dims: Iterable[Hashable] | None = None if weights is None else weights.ds.dims

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
    def merge_sizes(*xr_data) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key) (key) in a given
        `XarrayLike` object.

        Args:
            *lifted_ds: Variadic argument of each of type ``LiftedDataset``
        """

        def _merge_single(
            acc_sizes: dict[Hashable, int],
            curr_xrdata: XarrayLike | None,
        ) -> dict[Hashable, int]:
            """
            merges sizes from each dataset/dataarray
            """
            if curr_lds is None:
                return acc_sizes

            merged_sizes = acc_sizes | curr_xrdata.sizes

            return merged_sizes 

        return functools.reduce(_merge_single, xr_data, {})

    @staticmethod
    def check_nonzero_obsvar(obs_var: LiftedDataset):
        """
        Warns if at least one obs variance term is zero in any variable in the dataset.
        """
        ref_ds: xr.Dataset = obs_var.ds
        # runs for each variable in dataset:
        ref_ds_has_zero_obs: xr.Dataset = (ref_ds == 0).any()
        # combines boolean flags for all variables:
        any_zero_obs = any(ref_ds_has_zero_obs.values())
        if any_zero_obs:
            warnings.warn(NseUtils.WARN_ZERO_OBS_VARIANCE)

    @staticmethod
    def check_gathered_dims(gathered_dims: FlexibleDimensionTypes, dim_sizes: dict[Hashable, int]):
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

    #: strong ref to builder, since we want to keep the builder in scope to check how this object
    #: was created - see ``__post_init__``
    ref_builder: NseScoreBuilder
    #: observations as a lifted dataset
    obs: LiftedDataset
    #: forecasts (or "simulation") as a lifted dataset
    fcst: LiftedDataset
    #: marker to keep track of the underlying (consistent) type the obs, fcst, and weights. Mainly
    #: for isomorphism i.e. if a user provides data arrays they get back data arrays and similarly
    #: for datasets.
    xr_type_marker: XarrayTypeMarker
    #: dimensions to reduce - ``preserve_dims`` and ``reduce_dims`` is converted to this equivilent
    #: representation.
    reduce_dims: FlexibleDimensionTypes
    #: optional weights as a lifted dataset
    weights: LiftedDataset | None
    #: optional setting to flag that the data is angular or directional (in degrees)
    is_angular: bool = False

    def __post_init__(self):
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
    def mse_callback(self) -> Callable[..., LiftedDataset]:
        """
        Callback helper for calculating :py:func:`scores.mse`, with prefilled keyword args.
        """
        # lifts `mse` so that it can work with LiftedDataset types
        return functools.partial(
            LiftedDatasetUtils.lift_fn_ret(scores.continuous.mse),
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    @functools.cached_property
    def nse(self) -> XarrayLike:
        """
        The main calculation function for NSE:

            - denominator (D): calculates the obs variance using a combination of ``xr.mean`` and
              :py:func:`scores.mse`

            - numerator (N): calculates the forecast error using :py:func:`scores.mse` on ``obs``
              and ``fcst``.

            - ``score = 1 - N / D`` reduced and broadcast appropriately by ``xarray``, ``numpy``
              and/or ``dask``.

        Raises:
            UserWarning: If divide by 0 detected in ANY element.
                         raised by :py:meth:`NseUtils.check_nonzero_obsvar`

        .. note::

            Divide by zero in itself is allowed, as the rare case where e.g. all entries along a
            coordinate may genuinely be 0 or missing - though rare.

            This shouldn't stop other entries from being calculated - so a warning is raised instead.

            .. code-block::

                n / 0  =  NaN   : if n == 0  (represented by np.nan)
                       = -Inf   : if n == 0  (represented by -np.inf)

        """
        ret_nse: XarrayLike | None = None

        # ignore divide-by-zero errors: already raised by other checks - see docstring
        with np.errstate(divide="ignore"):
            # compose final score
            ret_nse = 1.0 - (self.fcst_error.raw() / self.obs_variance.raw())

        # assert type: return type must be XarrayLike
        assert_xarraylike(ret_nse)

        return ret_nse

    @functools.cached_property
    def fcst_error(self) -> LiftedDataset:
        """
        Calculates the forecast error which is essentially ``scores.mse(fcst, obs)``
        """
        return self.mse_callback(self.fcst, self.obs)

    @functools.cached_property
    def obs_variance(self) -> LiftedDataset:
        """
        Calculates the obs variance which is essentially ``scores.mse(obs_mean, obs)``.
        """
        # get inner xarray data
        obs_ref: XarrayLike = self.obs.inner_ref()

        # perform calculations in "lifted" space
        xr_obs_mean: XarrayLike = obs_ref.mean(dim=self.reduce_dims)
        lds_obs_mean: LiftedDataset = LiftedDataset(xr_obs_mean)
        lds_obs_var: LiftedDataset = self.mse_callback(lds_obs_mean, self.obs)

        # check for 0 obs variance: divide by zero condition
        NseUtils.check_nonzero_obsvar(lds_obs_var)

        return lds_obs_var
