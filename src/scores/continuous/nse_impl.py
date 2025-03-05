"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import functools
import warnings
import weakref
from collections.abc import Hashable, Iterable
from dataclasses import KW_ONLY, dataclass
from typing import Callable, cast

import numpy as np

import scores.continuous as _cnt
from scores.typing import (
    FlexibleDimensionTypes,
    LiftedDataset,
    XarrayLike,
    XarrayTypeMarker,
)
from scores.utils import check_dims, check_weights_positive, gather_dimensions


class NseUtils:
    """
    Helper class with static methods for use in NSE computations only. Also contains error msg
    strings specific to NSE.

    .. important::

        DEVELOPER-NOTE:

            - All utility functions assume that we are dealing with ``LiftedDataset``s Any new
              helpers declared here should follow the same pattern. To dispatch to common utility
              functions, use ``.ds`` or ``LiftedDataset.as_ds()`` to get the underlying dataset.

            - Only call ``.raw()`` if you no longer intend to use the ``LiftedDataset`` and want to
              retract to the original xarray datatype. ``.raw()`` invalidates the ``LiftedDataset``
              object and it can no longer be used. This is usually the last operation in a chain.
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
    def preprocess_nse_inputs(
        fcst: LiftedDataset,
        obs: LiftedDataset,
        weights: LiftedDataset | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> FlexibleDimensionTypes:
        """
        Checks the dimension compatibilty of the various input arguments. This is the main utility
        function that groups all the checks needed for NSE.
        """
        # check weights conform to NSE computations
        NseUtils.check_weights(weights)
        weights_dims: Iterable[Hashable] | None = None if weights is None else weights.dims

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=fcst.ds.dims,
            obs_dims=obs.ds.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        merged_sizes: dict[Hashable, int] = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension - required to calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def get_xr_type_marker(
        lds_obs: LiftedDataset,
        lds_fcst: LiftedDataset,
        lds_weights: LiftedDataset,
    ) -> XarrayTypeMarker:
        try:
            return check_lds_same_type(fcst, obs, weights)
        except TypeError as _e:
            # add a note specific to NSE and re-raise
            _e.add_note(NseUtils.ERROR_MIXED_XR_DATA_TYPES)
            raise

    @staticmethod
    def merge_sizes(*lifted_ds) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key) (key) in a given
        `XarrayLike` object.

        Args:
            *lifted_ds: Variadic argument of each of type ``LiftedDataset``
        """

        # helper function to process one lifted dataset at a time
        def _get_sizes(acc_sizes: dict[Hashable, int], curr_lds: LiftedDataset | None) -> dict[Hashable, int]:
            # -- None => return the accumulation
            if curr_lds is None:
                return acc_sizes
            # -- Otherwise => get sizes
            # get reference to inner ds
            curr_ds: xr.Dataset = LiftedDataset.as_ds(curr_lds)
            # get sizes
            # note: this API is not per variable name at time of writing, it was changed to maintain
            # compatibility with DataArray.
            curr_sizes: dict[Hashable, int] = curr_ds.sizes
            # merge (take the union) of the sizes and return a new dict
            return acc_sizes | curr_sizes

        # fold _get_sizes on all lifted datasets
        return functools.reduce(_get_sizes, lds, {})

    @staticmethod
    def check_weights(weights: LiftedDataset | None):
        """
        Wrapper around :py:func:`~scores.utils.check_weights_positive`, with a context specific
        warning message.
        """
        if weights is not None:
            check_weights_positive(ds_weights, context="`NSE` must have positive weights.")

    @staticmethod
    def check_nonzero_obsvar(obs_var: LiftedDataset):
        """
        Warns if at least one obs variance term is zero in any variable in the dataset.
        """
        ref_ds: xr.Dataset = LiftedDataset.as_ds(obs_var)
        # check if variables have zero obs variance
        ref_ds_has_zero_obs: xr.Dataset = (obs_var == 0).any()
        # check if any element in the entire dataset has zero obs variance
        any_zero_obs = any(v for _, v in enumerate(ref_ds_has_zero_obs))
        # custom warning for NSE overriding numpy's default divide by 0 warning
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

    def lift_xrdata(
        cls,
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike | None = None,
        /,  # force positional ordering
    ) -> tuple[tuple[LiftedDataset], XarrayTypeMarker]:
        """
        The output of this function is a little complex since it has to return the original
        arguments in its lifted form AND also requires all the arguments to check for type
        consistency.

        However, its usecase is narrow and is merely used as a helper for this builder so its not
        abstracted to its own ``NseUtils`` helper. An example might help clarify its usage.

        Example:
            .. code-block :: python

                fcst: XarrayLike = ...
                obs: XarrayLike = ...
                weights: XarrayLike = ...

                (
                    (lds_fcst, lds_obs, lds_weights),  # (LiftedDataset, LiftedDataset, LiftedDataset)
                    xr_type_marker,                    # XarrayTypeMarker
                ) = lift_xrdata(fcst, obs, weights)

        here ``lds`` prefix stands for ``LiftedDataset`` and ``xr_type_marker`` is the common xr
        data type associated to all of ``fcst``, ``obs`` and ``weights``.

        .. important::

            - ``weights`` are optional, if so, lds_weights is None. This has to be handled
              appropriately by the caller.
            - the order of the outputs must match the inputs, the logic in this function relies on
              positional args (i.e. fcst is always first and weights is always last) to avoid
              complexity in the caller having to handle kwargs -> key-value pairs (which themselves
              may have their own quirks)

        Raises:
            TypeError: see :py:func:`scores.typing.check_lds_same_type` for more details
        """
        # lift and check
        lifted_all: tuple[LiftedDataset] = map(LiftedDataset, (fcst, obs, weights))
        xr_type_marker: XarrayTypeMarker = NseUtils.get_xr_type_marker(*lifted_all)
        # see docstring on how to use this output or associated tests
        return (lifted_all, xr_type_marker)

    def build(
        self,
        *,  # force KW_ONLY
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

        # order is important, see docstring for lift_xrdata
        ((lds_fcst, lds_obs, lds_weights), xr_type_marker) = self.lift_xrdata(fcst, obs, weights)

        reduce_dims = NseUtils.check_and_gather_dims(
            fcst=lds_fcst,
            obs=lds_obs,
            weights=lds_weights,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        ret_nsescore = NseScore(
            ref_builder=self,
            fcst=lds_fcst,
            obs=lds_obs,
            xr_type_marker=xr_type_marker,
            reduce_dims=reduce_dims,
            weights=lds_weights,
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
        return functools.partial(
            LiftedDataset.lift_fn(_cnt.mse),
            reduce_dims=self.reduce_dims,
            weights=self.weights.ds,
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

        # perform all operations as Datasets
        ret_nse: xr.Dataset | None = None
        ds_fcst: xr.Dataset = LiftedDataset.as_ds(self.fcst)
        ds_obs: xr.Dataset = LiftedDataset.as_ds(self.obs)

        # compute intermediate terms
        obs_mean: xr.Dataset = self.obs.ds.mean(dim=self.reduce_dims)
        obs_variance: LiftedDataset = self.mse_callback(obs_mean, self.obs)
        fcst_error: LiftedDataset = self.mse_callback(ds_fcst, ds_obs)

        # raise warnings due to divide by zero in fcst/obs

        # intentional divide="ignore", as warning is already raised above.
        # `numpy` will fill divide by zero elements with NaNs.
        with np.errstate(divide="ignore"):
            ret_nse = 1.0 - (self.fcst_error / self.obs_variance)
        # cast strictly to `xr.Dataset` from an `Optional`. If we got this far, it can't be `None`.
        cast(xr.Dataset, ret_nse)
        # convert return type strictly to original

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
