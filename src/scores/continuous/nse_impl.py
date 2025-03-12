"""
Internal module used to support the computation of Nash-Sutcliffe model Efficiency coefficient (NSE).

The only publically exposed function should be `nse`.

.. note::

    FUTUREWORK:

        the structures introduced in this score can be abstracted as they are useful checks that can
        geenrally apply to most scores, in the short-term it is likely the hydro scores can all
        adapt the same pattern.

        Further adoption can be determined if the pattern works well for hydro metrics.
"""

import functools
import warnings
from collections.abc import Hashable
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


def nse(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    weights: XarrayLike | None = None,
    is_angular: bool = False,
) -> XarrayLike:
    """
    The Nash-Sutcliffe model Efficiency coefficient (NSE) is primarily used in hydrology to assess
    the skill of model predictions (of e.g. "discharge").

    While NSE is often calculated over observations and model predictions in the time dimension, it
    is actually a fairly generic statistical measure that determines the relative magnitude of the
    residual variance ("noise") compared to the measured data variance ("information") (Nash and
    Sutcliffe, 1970). Incidentally, it is (inversely) related to the signal-to-noise ratio (SNR).

    The general formulation of NSE is as follows:

    .. math::

        \\text{NSE} = 1 - \\frac{\\sum_i{(O_i - S_i)^2}}{\\sum_i{(O_i - \\bar{O})^2}}

    where
        - :math:`i` is a generic "indexer" representing the set of datapoints along the dimensions
          being reduced e.g. time (:math:`t`) or xy-coordinates (:math:`(x, y)`). The latter
          represents reduction over two dimensions as an example.
        - :math:`O_i` is the observation at the :math:`\\text{i^{th}}` index.
        - :math:`S_i` is the "forecast" or model simulation at the :math:`\\text{i^{th}}` index.
        - :math:`\\bar{O}` is the mean observation of the set of indexed samples as specified by
          ``reduce_dims`` and ``preserve_dims``.

    Args:
        fcst: "Forecast" or predicted variables.
        obs: Observed variables.
        reduce_dims: dimensions to reduce when calculating the NSE. (i.e. NSE will be calculated
            using datapoints along these dimensions as samples, the other dimensions will be
            preserved).
        preserve_dims: dimensions to preserve. Mutually exclusive to ``reduce_dims``. All
            dimensions not specified here will be reduced as described in ``reduce_dims``.
            Note: ``preserve_dims="all"`` is not supported for NSE. See notes below.
        weights: Optional weighting to apply to the NSE computation. Typically weights are applied
            over `time_dim` but can vary by location as well. Weights must be non-negative and
            specified for each data point _(i.e. the user must not assume broadcasting will handle
            appropriate assignment of weights for this score)_.
        is_angular: specifies whether `fcst` and `obs` are angular data (e.g. wind direction). If
            True, a different function is used to calculate the difference between `fcst` and `obs`,
            which accounts for circularity. Angular `fcst` and `obs` data should be in degrees
            rather than radians.

    Returns:
        ``XarrayLike`` containing the NSE score for each preserved dimension.

    Raises:
        DimensionError: If any dimension checks fail.
        KeyError: If no dimensions are being reduced - NSE requries at least 1 dimension to be
            reduced to compute obs variance.
        IndexError: If the dimensions being reduced only have 1 value - not sufficient for
            computation of obs variance.
        UserWarning: If weights are negative, or invalid (e.g. all zeros or all NaNs).
        UserWarning: If attempting to divide by 0. The computation will still succeed but produce
            ``np.nan`` (numerator is also 0) or ``-np.inf`` where divide by zero would occur.
        Exception: Any other errors or warnings not otherwise listed due to calculations associated
            with utility functions such as `gather_dimensions`.
        RuntimeError: If something went wrong with the underlying implementation. The user will be
            prompted to report this as a github issue.

    Supplementary details:

        - Nash-Sutcliffe efficiencies range from -Inf to 1. Essentially, the closer to 1, the more
          accurate the model is.
            - NSE = 1, corresponds to a perfect match of the model to the obs.
            - NSE = 0, indicates that the model is as accurate as the mean obs.
            - -Inf < NSE < 0, indicates that the mean obs is better predictor than the model.

        - The optional ``weights`` argument can additionally be used to perform a weighted NSE
          (wNSE). Although, this is a generic set of weights, and it is the _user's responsiblility_
          to define them appropriately. Typically this is the observation itself (Hundecha, Y., &
          Bárdossy, A., 2004).

        - ``weights`` must be non-negative. Therefore, the observations must ideally also be
          non-negative (or formulated appropriately) if used as weights.

        - While ``is_angular`` is typically not used for this score, NSE is generic enough that it
          _could_ be used in wider context, and hence is kept as an option. It is defaulted to
          ``False`` as that's the typical use-case.

    .. important::

        This score does not allow mixed xarray data structures as inputs. Either provide all
        ``xr.DataArray`` or all ``xr.Dataset`` exclusively, for the ``fcst``, ``obs`` and
        (optionally) ``weights`` arguments.

        This is an intentionally imposed constraint to make sure the inner computations are simple
        to check and deterministic. See tips below for more information.

    .. warning::

        Operations between dataarrays are not guarenteed to preserve names. If the user is
        working with dataarrays, it is assumed that preserving names is not a major requirement. If
        a user needs the name preserved, they should explicitly convert all data array inputs to
        datasets using `xr.DataArray.to_dataset(...)` , and verify that the naming is retained
        appropriately before calling the score.

        For operations where ONLY ``xr.DataArray`` inputs are used, the returned score will have its
        name forced to the name of this score i.e. "NSE", for simplicity.

        See tips below for more information.

    .. note::

        For Hydrology in particular :math:`i = t` - the reduced dimension is usually the time
        dimension. However, in order to keep things generic, this function does not explicitly
        mandate a time dimension be provided. Instead it requires it has a hard requirement that _at
        least one_ dimension is being reduced from either a specification of ``reduce_dims`` or
        ``preserve_dims`` (mutually exclusive).

        The reason is that the observation variance cannot be non-zero if nothing is being reduced.

        As a side-effect of the above requirement, ``preserve_dims="all"`` is not allowed and will
        naturally throw an error.

    .. note::

        Divide by zero is ALLOWED - to accomodate scenarios where all obs entries in the group being
        reduced is constant (0 obs variance).

        While these may cause divide by zero errors, they should not halt execution of computations
        for other valid coordinates - so a warning is issued instead to prompt the user to double
        check the data.

        It may also be that divide by zero is unavoidable - in which case we still want to return
        the correctly calculated values. To this end, this is how ``numpy`` resolves divide by zero:

        .. code-block::

            n / 0  =  NaN   : if n == 0  (represented by np.nan)
                   = -Inf   : if n == 0  (represented by -np.inf)

    .. tip::

        When dealing with dask arrays dask, no computation will happen until ``.compute(...)`` is
        called on the returned score.

    .. tip::

        Work with datasets where possible with NSE, or for any score that supports datasets for that
        matter. Datasets maintain structural integrity better than their dataarray counterparts and
        also are compatible with higher order types like `xr.DataTree`.

        Operations between datasets are more predictable than operations with mixed types.
        Dataarrays on the other hand may ignore names and broadcast liberally even when names do not
        match, and this may not be consistent depending on the oepration. This may or may not be the
        intented behaviour the user expects.  Operations between ONLY dataarrays are fine as long as
        preserving names is not mandatory.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.continuous import nse
        >>> obs_raw = np.array(
        ...     [
        ...         [[1,2,3], [4,5,6]],
        ...         [[3,2,1], [6,5,4]],
        ...         [[3,2,5], [2,2,6]],
        ...         [[5,2,3], [4,-1,4]],
        ...     ]
        ...
        ... )  # dimension lengths: x=4, y=2, t=3
        >>> obs = xr.DataArray(obs_raw, dims=["x", "y", "t"])
        >>> fcst = obs * 1.2 + 0.1  # add some synthetic bias and variance
        >>> # Example 1:
        >>> # reduce over t - time - should produce a 4x2 array representing the xy coordinate grid
        >>> nse(obs, fcst, reduce_dims=["t"])
        <xarray.DataArray (x: 4, y: 2)> Size: 64B
        array([[ 0.71180556, -0.28819444],
               [ 0.71180556, -0.28819444],
               [ 0.70982143,  0.85742188],
               [ 0.70982143,  0.93208333]])
        Dimensions without coordinates: x, y
        >>> # Example 2:
        >>> # reduce over (x, y) - space - should produce a 3x1 representing a time series
        >>> nse(obs, fcst, reduce_dims=["x", "y"])
        <xarray.DataArray (t: 3)> Size: 24B
        array([0.77469136, 0.90123457, 0.74722222])
        Dimensions without coordinates: t

    References:

        1. Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models
           part I — A discussion of principles. In Journal of Hydrology (Vol. 10, Issue 3, pp. 282–
           290). Elsevier BV. https://doi.org/10.1016/0022-1694%2870%2990255-6

        2. Hundecha, Y., & Bárdossy, A. (2004). Modeling of the effect of land use changes on the
           runoff generation of a river basin through parameter regionalization of a watershed
           model. Journal of Hydrology, 292(1-4), 281-295. https://doi.org/10.1016/j.jhydrol.2004.01.002
    """
    # safety: assert that the input types are as expected. This is for early failure during
    # dev/testing only when incompatible types are detected at runtime.
    assert (
        is_xarraylike(obs)
        and is_xarraylike(fcst)
        # These are optionals (i.e. skipped if None):
        and (weights is None or is_xarraylike(weights))
        and (reduce_dims is None or is_flexibledimensiontypes(reduce_dims))
        and (preserve_dims is None or is_flexibledimensiontypes(preserve_dims))
        and isinstance(is_angular, bool)
    )

    # do any dimension checks and store the final reduced (gathered) dimensions
    gathered_dims = NseUtils.check_and_gather_dimensions(
        fcst=fcst,
        obs=obs,
        weights=weights,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    gathered_dims = list(gathered_dims)

    # prepare inputs to pass into internal scoring function - also lifts any
    # dataarrays to datasets storing appropriate metadata to undo this operation in
    # the final score
    meta_input: NseMetaInput = NseUtils.make_metainput(
        fcst=fcst,
        obs=obs,
        weights=weights,
        gathered_dims=gathered_dims,
        is_angular=is_angular,
    )

    # actual nse computation happens here.
    meta_score: NseMetaScore = _nse_metascore(meta_input)

    # extracts the raw score from the metascore, converting to data array if required
    nse_result = NseUtils.extract_result_from_metascore(meta_score, meta_input)
    # safety: assert xarraylike before returning. For dev/testing only.
    assert is_xarraylike(nse_result)

    return nse_result


class NseComponents(NamedTuple):
    """
    The individual components of the NSE score.
    """

    fcst_error: xr.Dataset
    obs_variance: xr.Dataset
    nse: xr.Dataset


class NseMetaScore(NamedTuple):
    """
    Meta score structure containing:
        components: individual components of the score
            - currently only NseComponents.nse is returned
            - NseComponents.fcst_error and NseComponents.obs_variance are only used for
              testing
        is_dataarray: whether the underlying datasets to revert back to its original
            form before returning
    """

    components: NseComponents
    is_dataarray: bool


class NseDatasets(NamedTuple):
    """
    Namespace for storing dataset information for NSE
    """

    fcst: xr.Dataset
    obs: xr.Dataset
    weights: xr.Dataset | None


class NseMetaInput(NamedTuple):
    """
    Namespace that wraps NseDatasets with metadata about the underlying datasets
    particularly:
        is_dataarray: whether the underlying datasets were originally data arrays
        is_dummyname: whether the underlying dataarrays were provided dummy names
        datasets: the NseDatasets being wrapped
    """

    datasets: NseDatasets
    gathered_dims: list[Hashable]
    is_dataarray: bool
    is_angular: bool

    # NOTE: these methods are read-only, due to Named tuple inheritence:

    def _mse(self, x1: xr.Dataset, x2: xr.Dataset) -> xr.Dataset:
        """
        Runs _mse with defaults prefilled from input data.

        This version assumes datasets are passed in.

        The order of x1 and x2 do not matter since mse is a symmetric score.
        """
        # safety: dev/testing only
        assert isinstance(x1, xr.Dataset) and isinstance(x2, xr.Dataset)
        ret: xr.Dataset = scores.continuous.mse(
            x1,
            x2,
            reduce_dims=self.gathered_dims,
            weights=self.datasets.weights,
            is_angular=self.is_angular,
        )
        # safety: dev/testing only
        assert isinstance(ret, xr.Dataset)
        return ret

    def forecast_error(self) -> xr.Dataset:
        """
        Returns the forecast error - calculated using ``scores.continuous.mse``
        """
        fcst_error = self._mse(self.datasets.fcst, self.datasets.obs)
        return fcst_error

    def observation_variance(self) -> xr.Dataset:
        """
        Returns the observation variance - calculated using ``scores.continuous.mse``
        """
        obs_mean = self.datasets.obs.mean(dim=self.gathered_dims)
        obs_variance = self._mse(obs_mean, self.datasets.obs)
        return obs_variance


def _nse_metascore(meta_input: NseMetaInput) -> NseMetaScore:
    """
    The actual internal score function that computes NSE. Takes in input parameters with
    associated metadata, NseMetaInput and returns a NseMetaScore - transferring any
    metadata to it.

    since NseMetaInput and NseMetaScore are namedtuples they cannot be mutated so their
    data can only be transferred.

    IMPORTANT: This internal function is only compatible with "Meta" types that deal
               solely with datasets.
    """
    # Warning would have been raised by `warn_nonzero_obsvar` instead
    with np.errstate(divide="ignore"):
        fcst_error = meta_input.forecast_error()
        obs_variance = meta_input.observation_variance()

        # raise error if divide by zero in any element in obs variance
        NseUtils.warn_nonzero_obsvar(obs_variance)

        # nse computation
        _nse: xr.Dataset = 1.0 - (fcst_error / obs_variance)

        # store components in NseMetaScore
        meta_score = NseMetaScore(
            components=NseComponents(
                nse=_nse,
                obs_variance=obs_variance,
                fcst_error=fcst_error,
            ),
            # propagate required return type
            is_dataarray=meta_input.is_dataarray,
        )

        return meta_score


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

    # score name used for promoted dataarrays
    SCORE_NAME: str = "NSE"

    # datasets require a name - used when promoting a dataarray without a name.
    # NOTE: this doesn't actually replace the input dataarray name.
    _DATAARRAY_TEMPORARY_NAME: str = "__NONAME"

    ERROR_CORRUPT_METADATA: str = f"""
    {SCORE_NAME}: CRITICAL FAILURE! The internal state of either NseMetaInputs or
    NseMetaScore has been corrupt.  This should not happen and is either a BUG or is being run on a
    mutated version of the code.

    Please raise an issue on github.com/nci/scores citing this error.
    """

    ERROR_NO_DIMS_TO_REDUCE: str = f"""
    {SCORE_NAME}: need at least one dimension to be reduced. Check that `preserve_dims` is
    not preserving all dimensions, OR check that `reduce_dims` is specified with at least one
    dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = f"""
    {SCORE_NAME}: need at least one dimension that is being reduced where its length > 1.
    Check that the input provided have at least 2 or more data points along the dimensions being
    reduced. It is not possible to calculate a non-zero obs variance from 1 point.
    """

    ERROR_MIXED_XR_DATA_TYPES: str = f"""
    {SCORE_NAME}: Triggered during NSE calculations` check `fcst`, `obs` and `weights` are
    of the same type `xr.Dataset` OR `xr.DataArray` EXCLUSIVELY; NOT a mix of the two types.
    """

    WARN_ZERO_OBS_VARIANCE: str = f"""
    {SCORE_NAME}: possible divide by zero - at least one element in the reduced obs
    variance array is 0.  Any divide by zero entries will be filled in as `np.nan` if the forecast
    error is also 0, otherwise it will be `-np.inf`. This is so that any other valid entries are
    still computed and returned.  The user should still verify that zero obs variance is expected
    for the given input data.
    """

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
    def make_metainput(
        *,  # enforce keyword-only
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike,
        gathered_dims: list[Hashable],
        is_angular: bool,
    ) -> NseMetaInput:
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
        # enforce all of `fcst`, `obs` and (optionally) `weights` have the same type
        NseUtils.check_all_same_type(*[fcst, obs, weights])

        # same types already enforced - we can arbitrarily use the type of fcst
        is_dataarray = isinstance(fcst, xr.DataArray)

        fcst_ds, obs_ds, weights_ds = map(NseUtils.xarraylike_to_dataset, (fcst, obs, weights))

        return NseMetaInput(
            datasets=NseDatasets(
                fcst=fcst_ds,
                obs=obs_ds,
                weights=weights_ds,
            ),
            gathered_dims=gathered_dims,
            is_angular=is_angular,
            is_dataarray=is_dataarray,
        )

    @staticmethod
    def xarraylike_to_dataset(xrlike: XarrayLike) -> xr.Dataset:
        """
        Helper that promotes dataarrays to datasets or does nothing if None or a dataset is given.

        .. note ::

            operations between dataarrays are not guarenteed to preserve names. If the user is
            working with dataarrays, it is assumed that preserving names is not a major requirement
            (unlike datasets).
        """
        ret_ds = xrlike

        if isinstance(xrlike, xr.DataArray):
            # keep original name for assertion - see below
            original_name = xrlike.name

            # obfuscate the name of the dataarray when it is promoted to dataset so that operations
            # can broadcast with each other. NOTE: this doesn't actually change the name of the
            # underlying data array.
            ret_ds = xrlike.to_dataset(name=NseUtils._DATAARRAY_TEMPORARY_NAME)

            # assertion below to make sure this behaviour does not change between xarray versions.
            # assert that referenced dataarray still has the same name.
            assert xrlike.name == original_name

        return ret_ds

    @staticmethod
    def try_extract_singleton_dataarray(ds: xr.Dataset) -> xr.DataArray | None:
        """
        Tries to extract a dataarray if there is only one key.

        Returns xr.DataArray, if a singleton is found
                None, if multiple dataarrays or dataset is empty i.e. no keys
        """
        keys_ = list(ds.keys())
        if len(keys_) == 1:
            da = ds[keys_[0]]
            return da
        return None

    @staticmethod
    def check_metadata_consistency(
        meta_score: NseMetaScore,
        ref_meta_input: NseMetaInput,
    ):
        """
        Checks against corruption of data. This usually cannot happen unless there is a bug in the
        code that tries to copy and replace one of the named tuples - normally their shallow
        properties shouldn't be mutable.

        Raises:
            RuntimeError: If a corruption is detected this is critical failure. User will be
                prompted to create a github issue, so a developer can address it.
        """
        corrupt = False

        if meta_score.is_dataarray != ref_meta_input.is_dataarray:
            corrupt = True

        def _ds_has_single_key(fieldname):
            """
            Helper that checks whether a component in meta_score has a single
            dataarray (as determined by the number of keys in `data_vars` key)
            """
            _ds = getattr(meta_score.components, fieldname)
            _single_key = len(_ds.data_vars.keys()) == 1
            return _single_key

        # A dataset produced from a dataarray MUST only have ONE data_var key
        if meta_score.is_dataarray:
            all_single_key = all(map(_ds_has_single_key, meta_score.components._fields))
            if not all_single_key:
                corrupt = True

        if corrupt:
            raise RuntimeError(NseUtils.ERROR_CORRUPT_METADATA)

    @staticmethod
    def extract_result_from_metascore(
        meta_score: NseMetaScore,
        ref_meta_input: NseMetaInput,
    ) -> XarrayLike:
        """
        If inputs were originally dataarrays, downgrades the score to a dataarray, naming it with
        the score i.e. "NSE".

        Otherwise returns the score as is (as a dataset)

        Checks that the meta score is consistent with the meta input - i.e. no corruption has
        happened to the underlying datasets or metadata. see: ``check_metadata_consistency``

        Args:
            meta_score: the meta score to extract the raw NSE score from
            ref_meta_input: reference to the meta input for checking consistency
        """
        NseUtils.check_metadata_consistency(meta_score, ref_meta_input)

        ret_nse = meta_score.components.nse

        # demote if originally a data array and force the name to "NSE"
        if meta_score.is_dataarray:
            # returns None if extraction fails
            ret_nse = NseUtils.try_extract_singleton_dataarray(meta_score.components.nse)
            # safety: check_metadata_consistency should have already handled this
            assert ret_nse is not None

        return ret_nse

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
            - offloads dimensions checks to generic utility ``gather_dimensions`` does
            - the above is also used to gather dimensions.
            - checks that the input weights are conformant
            - checks that the reduced dimensions have at least one value, otherwise NSE cannot be
              calculated as it needs ``count(obs) > 1`` along the reduced dimension for variance
              calculations.
        """
        weights_dims = getattr(weights, "dims", None)

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=fcst.dims,
            obs_dims=obs.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        # check weights if its provided
        if weights is not None:
            check_weights(weights)

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
    def warn_nonzero_obsvar(obs_var: xr.Dataset):
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
