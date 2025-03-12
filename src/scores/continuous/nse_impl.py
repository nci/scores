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

        For simplicity, when working with ``xr.DataArray`` the returned score will have its name
        forced to the name of this score i.e. "nse". If this is an issue - use ``xr.Dataset``
        instead. See tips below for more information.

    .. note::

        For Hydrology in particular :math:`i = t` i.e. the time dimension. In order to keep things
        generic, this function does not explicitly require the user to provide a "time" dimension
        along which to apply the score. Instead, it sets a hard requirement that _at least one_
        dimension is being reduced from either a specification of ``reduce_dims`` or
        ``preserve_dims`` (mutually exclusive).

        The reason is that the observation variance cannot be non-zero if nothing is being reduced.

        If the user is particularly interested in applying the NSE computation over the "time"
        dimension, they should provide the "time" dimension as one of the elements in the
        ``reduce_dims`` argument, and ensure that the input datasets and associated variables have
        this dimension. Or otherwise, omit it from the ``preserve_dims`` argument.

        As a side-effect of the above requirement, ``preserve_dims="all"`` is not allowed and will
        naturally throw an error.

    .. note::

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

    .. tip::

        As a general tip - work with datasets where possible with NSE, or for any score that
        supports datasets for that matter. Datasets maintain structural integrity better than their
        dataarray counterparts and also are compatible with higher order types like `xr.DataTree`.

        To see why this is recommended, consider that ``xr.DataArray`` does not respect names
        (worse, its not consistent in whether or not it respects names either depending on the
        operation and operands). In fact the same argument can trickle down to ``np.array`` which
        has even less metadata constraints - useful for speed, but not for public API interfaces in
        scores.

        In particular, when operating with ``xr.Dataset``s, or even other ``xr.DataArray``s for
        that matter. A named dataarray would broadcast its operations to all variables in the
        dataset regardless of name - It's unclear if the user wants it or not.

        For example, if the dataarray was originally extracted from a dataset in the first place
        (and hence named) - we have two sets of different operations that could be equally valid
        without knowing intent.  One broadcasted, and the other only applying to variables that
        share the name.

        It is much clearer if the user intentionally works with datasets as this will clarify
        intent, either by manually broadcasting (using a helper to replicate the data across
        variables) or refraining from broadcasting because they want the default behaviour.

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
        >>>
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
    # safety: assert that the input types are as expected. This aimed more for
    # catching early errors while development/testing, and will be re-checked anyway
    # appropriately by `check_weights` and `check_and_gather_dimensions` as well as
    # during any computational checks performed in `_nse_metascore`
    assert (
        isinstance(fcst, xr.Dataset)
        and isinstance(obs, xr.Dataset)
        # These are optionals (i.e. skipped if None):
        and (weights is None or isinstance(weights, xr.Dataset))
        and (reduce_dims is None or is_flexibledimensiontypes(reduce_dims))
        and (preserve_dims is None or is_flexibledimensiontypes(preserve_dims))
        and isinstance(is_angular, bool)
    )

    # check weights if its provided
    if weights is not None:
        check_weights(weights)

    # do any dimension checks and store the final reduced (gathered) dimensions
    gathered_dims = NseUtils.check_and_gather_dimensions(
        fcst=fcst,
        obs=obs,
        weights=weights,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

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

    # check that metadata hasn't been corrupted by any operations so far.
    ...  # TODO

    # extracts the result from the metascore, converting back to a data array if
    # needed
    nse_result: XarrayLike = NseUtils.extract_result_from_metascore(
        meta_input,
        meta_score,
    )

    # safety: assert xarraylike before returning
    assert is_xarraylike(nse_result)

    return nse_result


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
        # observation variance computation
        obs_mean = meta_input.datasets.obs.mean(dim=meta_input.gathered_dims)
        obs_variance = meta_input.mse_prefilled(obs_mean, self.obs)

        # forecast error computation
        fcst_error = meta_input.mse_prefilled(fcst, obs)

        # nse computation - raise warning if divide by zero
        NseUtils.warn_nonzero_obsvar(obs_var)
        nse = 1.0 - (fcst_err / obs_var)

        # store components in NseMetaScore
        meta_score = NseMetaScore(
            components=NseComponents(
                nse=nse,
                obs_variance=obs_var,
                fcst_error=fcst_err,
            ),
            # propagate required return type
            is_dataarray=is_dataarray,
        )


class NseMetaScore(NamedTuple):
    """
    Meta score structure containing:
        components: individual components of the score
            - currently only NseComponents.score is returned
            - NseComponents.fcst_error and NseComponents.obs_variance are only used for
              testing
        is_dataarray: whether the underlying datasets to revert back to its original
            form before returning
    """

    components: NseComponents
    is_dataarray: bool


class NseComponents(NamedTuple):
    """
    The individual components of the NSE score.
    """

    fcst_error: xr.Dataset
    obs_variance: xr.Dataset
    score: xr.Dataset


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
    def make_metainput(
        *,  # enforce keyword-only
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike,
        gathered_dims: FlexibleDimensionTypes,
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

        def _to_ds(_xrlike: XarrayLike | None) -> xr.Dataset:
            """
            Helper to convert to dataset if not None, assigning it a dummy name if
            its a data array.

            .. note::

                names are obscured if dataarray since the final name will be replaced
                by "nse" instead - these datasets are transient objects. This
                simplifies the need to deal with names.
            """
            if _xrlike is None:
                return None
            # default - assume no lifting to dataset needed
            _ret = _xrlike
            if is_dataarray:
                # if its an array change it to a dataset, giving it a temporary name
                assert isinstance(_ret, xr.DataArray)
                return _ret.to_dataset(name=NseUtils._DATAARRAY_TEMPORARY_NAME)
            return _ret

        fcst_ds, obs_ds, weights_ds = map(_to_ds, (fcst, obs, weights))

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
    def check_metadata_consistency(
        metainput: NseMetaInput,
        metascore: NseMetaScore,
    ): ...  # TODO

    @staticmethod
    def extract_result_from_metascore(
        metascore: NseMetaScore,
    ) -> XarrayLike:
        """
        Runs the scorer to compute nse and use metadata from NseMetaInput to return the
        result to its original xarraylike.

        Its name is forced to "nse" for simplicity, rather than having to resolve name compatibility
        and conflicts. If this is an issue - the user is encouraged to use datasets natively - see
        tips in the public API docstrings.

        DataArrays operate against themselves regardless of name anyway, and whether or not they
        preserve their name is very operation dependent.
        """
        if metads.is_dataarray:
            # demote if originally a data array, must have only one key if this is the case
            da_keys = list(result.data_vars.keys())
            if len(da_keys) != 1:
                raise RuntimeError(NseUtils.ERROR_DATAARRAY_NOT_MAPPED_TO_SINGLE_KEY)
            result = result.data_vars[da_keys[0]]
            if metads.is_dummyname:
                result.name = None

        return result

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
