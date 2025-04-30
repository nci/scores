"""
Internal module used to support the computation of Nash-Sutcliffe model
Efficiency coefficient (NSE).

The only publically exposed function should be `nse_impl.nse`.

FUTUREWORK:

    The data structures introduced in this score can be abstracted as they are
    useful checks that can be applied in general to most scores. In the
    short-term, it is likely the hydro scores (e.g. KGE, Pbias) can easily
    adapt to this pattern .

    Broader adoption can be determined in the future if the pattern works well
    for hydro metrics.
"""

import functools
import warnings
from collections.abc import Hashable
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import xarray as xr

import scores.continuous
from scores.processing import broadcast_and_match_nan
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
    The Nash-Sutcliffe model Efficiency coefficient (NSE) is primarily used in
    hydrology to assess the skill of model predictions (of e.g. "discharge").

    While NSE is often calculated over observations and model predictions in
    the time dimension, it is actually a fairly generic statistical measure
    that determines the relative magnitude of the residual variance ("noise")
    compared to the measured data variance ("information") (Nash and Sutcliffe,
    1970). Incidentally, it is (inversely) related to the signal-to-noise ratio
    (SNR).

    The general formulation of NSE is as follows:

    .. math::

        \\text{NSE} = 1 - \\frac{\\sum_i{(O_i - S_i)^2}}{\\sum_i{(O_i - \\bar{O})^2}}

    where
        - :math:`i` is a generic "indexer" representing the set of datapoints
          along the dimensions being reduced e.g. time (:math:`t`) or
          xy-coordinates (:math:`(x, y)`). The latter represents reduction over
          two dimensions as an example.
        - :math:`O_i` is the observation at index :math:`i`.
        - :math:`S_i` is the "forecast" or model simulation at index :math:`i`.
        - :math:`\\bar{O}` is the mean observation of the set of indexed
          samples as specified by ``reduce_dims`` and ``preserve_dims``.

    Args:

        fcst: "Forecast" or predicted variables.

        obs: Observed variables.

        reduce_dims: dimensions to reduce when calculating the NSE. (i.e. NSE
            will be calculated using datapoints along these dimensions as
            samples, the other dimensions will be preserved).

        preserve_dims: dimensions to preserve. Mutually exclusive to
            ``reduce_dims``. All dimensions not specified here will be reduced
            as described in ``reduce_dims``.  Note: ``preserve_dims="all"`` is
            not supported for NSE. See notes below.

        weights: Optional weighting to apply to the NSE computation. Typically
            weights are applied over the time dimension but can vary by
            location as well. Weights must be non-negative and specified for
            each data point *(i.e. the user must not assume broadcasting will
            handle appropriate assignment of weights for this score)*.

        is_angular: specifies whether ``fcst`` and ``obs`` are angular data
            (e.g. wind direction).  If True, a different function is used to
            calculate the difference between ``fcst`` and ``obs``, which
            accounts for circularity. Angular ``fcst`` and ``obs`` data should
            be in degrees rather than radians.

    Returns:

        NSE score for each preserved dimension

        ``xr.Dataset``:  if ``fcst``, ``obs`` and optionally ``weights`` are all datasets.

        ``xr.DataArray``: ditto above - where inputs are all dataarrays

        See comments below for more information on mixed xarray data types
        (which this score does **not** handle)  and type isomorphism.

    Raises:

        DimensionError: If any dimension checks fail.

        KeyError: If no dimensions are being reduced - NSE requries at least 1
            dimension to be reduced to compute the observation variance.

        IndexError: If the dimensions being reduced only have 1 value - not
            sufficient for computation of obs variance.

        UserWarning: If weights are negative, or invalid
            (e.g. all zeros or all NaNs).

        UserWarning: If attempting to divide by 0. The computation will still
            succeed but produce ``np.nan`` (numerator is also 0) or ``-np.inf``
            where divide by zero would occur.

        Exception: Any other errors or warnings not otherwise listed due to
            calculations associated with utility functions such as
            ``gather_dimensions``.

        RuntimeError: If something went wrong with the underlying
            implementation. The user will be prompted to report this as a
            github issue.

    Supplementary details:
        - Nash-Sutcliffe efficiencies range from -Inf to 1. Essentially, the
          closer to 1, the more accurate the model is.
            - NSE = 1, corresponds to a perfect match of the model to the obs.
            - NSE = 0, indicates that the model is as accurate as the mean obs.
            - -Inf < NSE < 0, indicates that the mean obs is better predictor
              than the model.
        - The optional ``weights`` argument can additionally be used to perform
          a weighted NSE (wNSE). Although, this is a generic set of weights,
          and it is the *user's responsiblility* to define them appropriately.
          Typically this is the observation itself
          (Hundecha, Y., & Bárdossy, A., 2004).
        - ``weights`` must be non-negative. Therefore, the observations must
          ideally also be non-negative (or formulated appropriately) if used as
          weights.
        - While ``is_angular`` is typically not used for this score, NSE is
          generic enough that it _could_ be used in wider context, and hence is
          kept as an option. It is defaulted to ``False`` as that's the typical
          use-case.

    .. important::

        This score does not allow mixed xarray data structures as inputs.
        Either provide all ``xr.DataArray`` or all ``xr.Dataset`` exclusively,
        for the ``fcst``, ``obs`` and (optionally) ``weights`` arguments.

        This is an intentionally imposed constraint to make sure the inner
        computations are simple to check and deterministic. See tips below for
        more information.

    .. warning::

        Operations between dataarrays are not guarenteed to preserve names. If
        the user is working with dataarrays, it is assumed that preserving
        names is not a major requirement. If a user needs the name preserved,
        they should explicitly convert all data array inputs to datasets using
        ``xr.DataArray.to_dataset(...)`` , and *verify* that the naming is
        retained appropriately before calling the score.

        For operations where ONLY ``xr.DataArray`` inputs are used, the
        returned score will have its name forced to the name of this score i.e.
        "NSE", for simplicity.

        See tips below for more information.

    .. note::

        For Hydrology in particular :math:`i = t`,  the reduced dimension is
        usually the time dimension. However, in order to keep things generic,
        this function does not explicitly mandate a time dimension be provided.
        Instead it requires it has a hard requirement that *at least one*
        dimension is being reduced from either a specification of
        ``reduce_dims`` or ``preserve_dims`` (mutually exclusive).

        The reason is that the observation variance cannot be non-zero if
        nothing is being reduced.

        As a side-effect of the above requirement, ``preserve_dims="all"`` is
        not allowed and will naturally throw an error.

    .. note::

        Divide by zero *is allowed* - to accomodate scenarios where all obs
        entries in the group being reduced is constant (0 obs variance).

        While these may cause divide by zero errors, they should not halt
        execution of computations for other valid coordinates - so a warning is
        issued instead to prompt the user to double check the data.

        It may also be that divide by zero is unavoidable - in which case we
        still want to return the correctly calculated values. To this end, this
        is how ``numpy`` resolves divide by zero:

        .. code-block::

            # note: psuedocode
            n / 0  =  NaN   : if n == 0  (represented by np.nan)
                   = -Inf   : if n == 0  (represented by -np.inf)

    .. tip::

        When dealing with dask arrays dask, no computation will happen until
        ``.compute(...)`` is called on the returned score.

    .. tip::

        Work with datasets where possible with NSE, or for any score that
        supports datasets for that matter. Datasets maintain structural
        integrity better than their dataarray counterparts and also are
        compatible with higher order types like ``xr.DataTree``.

        Operations between datasets are more predictable than operations with
        mixed types. Data arrays on the other hand may ignore names and
        broadcast liberally even when names do not match, and this may not be
        consistent depending on the operation. This may or may not be the
        intented behaviour the user expects. Operations between **only**
        dataarrays are fine as long as preserving names is not mandatory.

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
        >>> # reduce over t - time - should produce a xy-grid (4 by 2)
        >>> nse(obs, fcst, reduce_dims=["t"])
        <xarray.DataArray (x: 4, y: 2)> Size: 64B
        array([[ 0.71180556, -0.28819444],
               [ 0.71180556, -0.28819444],
               [ 0.70982143,  0.85742188],
               [ 0.70982143,  0.93208333]])
        Dimensions without coordinates: x, y
        >>> # Example 2:
        >>> # reduce over (x, y) - space - should a t-vector (3 by 1)
        >>> nse(obs, fcst, reduce_dims=["x", "y"])
        <xarray.DataArray (t: 3)> Size: 24B
        array([0.77469136, 0.90123457, 0.74722222])
        Dimensions without coordinates: t

    References:

        1. Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting
           through conceptual models part I — A discussion of principles. In
           Journal of Hydrology (Vol. 10, Issue 3, pp. 282– 290). Elsevier BV.
           https://doi.org/10.1016/0022-1694%2870%2990255-6

        2. Hundecha, Y., & Bárdossy, A. (2004). Modeling of the effect of land
           use changes on the runoff generation of a river basin through parameter
           regionalization of a watershed model. Journal of Hydrology, 292(1-4),
           281-295. https://doi.org/10.1016/j.jhydrol.2004.01.002
    """
    # safety: assert that the input types are as expected. This is for early
    # failure during dev/testing only when incompatible types are detected at
    # runtime.
    assert (
        is_xarraylike(obs)
        and is_xarraylike(fcst)
        # These are optionals (i.e. skipped if None):
        and (weights is None or is_xarraylike(weights))
        and (reduce_dims is None or is_flexibledimensiontypes(reduce_dims))
        and (preserve_dims is None or is_flexibledimensiontypes(preserve_dims))
        and isinstance(is_angular, bool)
    )

    # wrap inputs in a container with metadata
    # NOTE: this needs to happen as early as possible before any other internal
    #       calls, as they will all rely on these "lifted" structures to
    #       minimize branching (choice) logic.
    meta_input: NseMetaInput = NseMetaHandler.make_metainput(
        fcst=fcst,
        obs=obs,
        weights=weights,
        preserve_dims=preserve_dims,
        reduce_dims=reduce_dims,
        is_angular=is_angular,
    )

    # actual nse computation happens here.
    meta_score: NseMetaScore = _nse_metascore(meta_input)

    # extracts the raw score from the metascore, doing any type conversions
    nse_result = NseMetaHandler.extract_result_from_metascore(
        meta_score=meta_score,
        orig_meta_input=meta_input,
    )

    # safety: assert xarraylike before returning. For dev/testing only.
    assert is_xarraylike(nse_result)

    return nse_result


class NseDatasets(NamedTuple):
    """
    Namespace for storing dataset information for NSE
    """

    fcst: xr.Dataset
    obs: xr.Dataset

    # optional
    weights: xr.Dataset | None


class NseMetaInput(NamedTuple):
    """
    Namespace that wraps NseDatasets with metadata about the underlying datasets
    particularly:
        datasets: the input datasets: `fcst`, `obs` and optionally `weights`.
        gathered_dims: the result of resolving preserve_dims/reduce_dims
        is_dataarray: whether the datasets were originally data arrays
        is_angular: whether the input data is angular [0, 360)
    """

    datasets: NseDatasets
    is_dataarray: bool
    gathered_dims: list[Hashable]

    # optional
    is_angular: bool = False

    # NOTE: the methods in this class are read-only, due to Named tuple inheritence:

    def _mse(self, x1: xr.Dataset, x2: xr.Dataset) -> xr.Dataset:
        """
        Runs mse (mean square error) with defaults prefilled from input data.

        This version assumes datasets are passed in.

        The order of x1 and x2 do not matter since mse is a symmetric score.

        This same helper is used for fcst_error and obs_variance for
        consistency, but also to maintain consistency with the API for `mse`
        itself.
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
        Returns forecast error - uses ``scores.continuous.mse``
        """
        fcst_error = self._mse(self.datasets.fcst, self.datasets.obs)
        return fcst_error

    def observation_variance(self) -> xr.Dataset:
        """
        Returns observation variance - uses ``scores.continuous.mse``
        """
        obs_mean = self.datasets.obs.mean(dim=self.gathered_dims)
        obs_variance = self._mse(obs_mean, self.datasets.obs)
        return obs_variance


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
            - NseComponents.fcst_error and NseComponents.obs_variance are only
                  used for testing
        is_dataarray: whether the underlying datasets to revert back to its
            original form before returning
    """

    components: NseComponents
    ref_meta_input: NseMetaInput


def _nse_metascore(meta_input: NseMetaInput) -> NseMetaScore:
    """
    The actual score logic is for NSE is presented here.

    Takes in NseMetaInput with all the information from the input data the user
    provided with checks performed on them (see docstring for NseMetaInput for
    what the collection contains).

    Returns NseMetaScore, which retains some of the metadata of NseMetaInput,
    particularly whether the input data contained dataarrays or datasets.

    .. important::

        For predictability of the xarray APIs used to perform the computations,
        this internal function is only compatible with "Meta" types that deal
        solely with datasets.

        NseMetaScore can be retracted back to its "raw" form using
        :py:func:`NseMetaHandler.extract_result_from_metascore`
    """
    # Warning would have been raised by `warn_nonzero_obsvar` instead
    with np.errstate(divide="ignore"):
        fcst_error = meta_input.forecast_error()
        obs_variance = meta_input.observation_variance()

        # raise error if divide by zero in any element in obs variance
        NseUtils.warn_nonzero_obsvar(obs_variance)

        # nse computation
        _nse: xr.Dataset = 1.0 - (fcst_error / obs_variance)

        # prepare components
        components = NseComponents(
            nse=_nse,
            obs_variance=obs_variance,
            fcst_error=fcst_error,
        )

        # make metascore - also runs post-score checks
        meta_score = NseMetaHandler.make_metascore(
            components=components,
            meta_input=meta_input,
        )

        return meta_score


class NseMetaHandler(SimpleNamespace):
    """
    This namespace contains functions responsible for converting to and from
    "Meta" types which are actually quite simple "higher order" structures that
    contain input/output data with extra state information. Importantly it
    preserves whether the input xarray data are datasets or dataarrays allowing
    for operations that preserve isomorphism between inputs and output types.

    i.e. if a user uses datasets, they will receive a dataset back, conversely
    if they use a dataarray they will receive a dataarray away. Note: scores
    that return components will always return either a dataset or a dict-like.

    .. note::

        This type of handling is not unheard of - in fact xarray itself does
        some of its structural operations on data arrays by "lifting" to a
        dataset first. However, since these are internal to xarray, we can't
        use them directly.

        Further, not all xarray operations guarentee this lifting, ironically
        we need it for the operations that xarray *does not* use it. (e.g.
        simple checks and broadcasting mathematical operations, where the
        latter is more tied to ``numpy`` and doesn't typically require a lift
        to datasets. However, as we see in the reference below, the
        ``_to_temp_dataset`` lift *can* obfuscate variable names even in
        inbuilt xarray logic)

        see: https://github.com/pydata/xarray/blob/v2025.01.2/xarray/core/dataarray.py#L597
        *(in particular ``_to_temp_dataset`` and places its used)*

    .. important::

        To reiterate - unlike xarray, our "Meta" structures are *always*
        shallow wrapper s

        This does not mean make them better, just more relevant to our usecase:
            unifying xarraylike checks (read-only) and mathematical
            computations (also read-only - for inputs). This is enforced by the
            use of namedtuples which by nature are read-only though the onus is
            on the developer to not do any referencial manipulation of its
            internals.

        whereas xarray's ``_to_temp_dataset`` often is used to create a deeper
        copy, which may be necessary for more complex structural mutations that
        benefit from having a copy in memory.

    In particular:

        NseMetaHandler.make_metainput:
            converts any XarrayLike objects in the input into datasets,
            recording its original type as state information.

        NseMetaHandler.pre_score_checks:
            checks to run before computing* the score.

        NseMetaHandler.post_score_checks:
            checks to run after computing* the score.

        NseMetaHandler.make_metascore:
            creates a metascore namedtuple obj given the components of the
            score and a reference to the meta input structure

        NseMetaHandler.extract_result_from_metascore:
            does an undo of the above operation returning the original data
            type to the user.

        (*) with dask the score is delayed in computation, and only data that
            are relevant for any checks are computed. If the score relies
            heavily on dask, the developer should ensure that any
            ``post_scoring_checks`` are done efficiently.

          E.g. for a check like (variance == 0).any(), one would expect that
          chunks are no longer dispatched to workers once the logic has
          converged to a deterministic value, i.e. if any one chunk returns
          True, it can no longer be False - though this needs verification
          (beyond the scope of current implementation)

        FUTUREWORK:

            It is quite apparent that this type of design need not be exclusive
            to NSE, other scores may benefit from this added structure
            especially helping with behavioural predictability of the different
            scores, and reducing the need for adhoc checks and branching logic.

            The developer only needs to design `fn_score`. Where `do_checks`
            can inherit from a namespace like this that abstracts common checks
            (albeit it needs a more commonly named utility not "Nse..").

            This also helps the reviewer down the track, because the only thing
            they need to check for is the soundness of the mathematical
            implementation and **computational** checks rather than structural.
            As an example, a typical pattern would look like:

            code-block ::

                # make meta input using args and kwargs from nse public API
                # [Transform]: lift to dataset
                meta_input = NseMetaHandler.make_metainput(*args, **kwargs)

                # any internal checks happen here
                # [Pre-score checks]: uses transformed objs only ("Meta...")
                NseUtils.do_some_preliminary_checks(meta_input)

                # a score always takes a meta_input and returns a meta_score
                meta_result: meta_score = _internal_score_logic(meta_input)

                # any post-score checks happen here.
                # [Post-score checks]: uses transformed objs only ("Meta...")
                NseUtils.do_some_post_scoring_checks(meta_input, meta_score)

                # [Undo Transform]: revert to original form
                raw_result: XarrayLike = NseMetaHandler.extract_result_from_metascore(meta_score)

                # return result (of isomorphic type - same as input) back to
                # the user.
                # [User output]: the above internal structures should be opaque
                #                to the user and should not manipulate any
                #                input data.
                return raw_result
    """

    @staticmethod
    def make_metainput(
        *,  # enforce keyword-only
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike,
        preserve_dims: FlexibleDimensionTypes | None,
        reduce_dims: FlexibleDimensionTypes | None,
        is_angular: bool,
    ) -> NseMetaInput:
        """
        Consolidates ``XarrayLike`` - a union type - to a xarray dataset, in
        order to simplify API calls.

        Also does some operations common to *most* scores:
            - gather dimensions (and implicitly check their compatiblity with
              inputs)
            - check weights are positive (if not None)

        Returns:
            A tuple containing the lifted dataset, whether it was originally a
            dataarray, and whether or not a dummy name was assigned.

        .. note::

            FUTUREWORK: this should eventually be some sort of higher order structure,
                        common to other scores.
        """
        # all of `fcst`, `obs` and (optionally) `weights` must be the same type
        NseUtils.check_all_same_type(*[fcst, obs, weights])

        # same types already enforced - we can arbitrarily use the type of fcst
        is_dataarray = isinstance(fcst, xr.DataArray)

        ds_fcst, ds_obs, ds_weights = map(
            NseUtils.xarraylike_to_dataset,
            (fcst, obs, weights),
        )

        # ---------------------------------------------------------------------
        # START COMMONCHECKS
        # Checks that should be common to MOST scores.
        #
        # These should be abstracted, but for now they need to be guarenteed to
        # happen *before* the meta_input is created but *after* the xrlike have
        # been lifted to datasets.
        #
        # Further, they rely on the input arguments from the public API so this
        # is the most efficient spot to place them currently.

        weights_dims = getattr(ds_weights, "dims", None)

        # check that weights are valid
        if ds_weights is not None:
            check_weights(ds_weights)

        # gather dimensions based on input args
        gathered_dims: FlexibleDimensionTypes = gather_dimensions(
            fcst_dims=ds_fcst.dims,
            obs_dims=ds_obs.dims,
            weights_dims=weights_dims,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )
        gathered_dims = list(gathered_dims)

        # check that gathered dims is in obs and fcst - they must be present in both
        # should be done before `broadcast_and_match_nan`
        NseUtils.check_vars_have_reducible_dims(ds_fcst, gathered_dims)
        NseUtils.check_vars_have_reducible_dims(ds_obs, gathered_dims)

        # broadcast nans between fcst and obs
        ds_fcst, ds_obs = broadcast_and_match_nan(ds_fcst, ds_obs)

        # END COMMONCHECKS
        # ---------------------------------------------------------------------

        meta_input = NseMetaInput(
            datasets=NseDatasets(
                fcst=ds_fcst,
                obs=ds_obs,
                weights=ds_weights,
            ),
            is_angular=is_angular,
            is_dataarray=is_dataarray,
            gathered_dims=gathered_dims,
        )

        # perform score *specific* checks here
        NseMetaHandler.pre_score_checks(meta_input)

        return meta_input

    @staticmethod
    def make_metascore(
        components: NseComponents,
        meta_input: NseMetaInput,
    ) -> NseMetaScore:
        """
        A simple constructor to make a metascore.
            - Runs common consistency checks against the meta input.
              (Currently none)
            - Runs specific post scoring checks.

        .. note::

            This is likely to be more score-dependent than make_metainput
        """
        meta_score = NseMetaScore(
            components=components,
            ref_meta_input=meta_input,
        )

        # ---------------------------------------------------------------------
        # START COMMONCHECKS

        # NOTE: there are currently no concrete common checks, since a common
        #       paradigm for checking scores outputs is less apparent compared
        #       to scores inputs.
        #
        #       Any metadata consistency checks SHOULD be common to most scores
        #       but performed JUST BEFORE dispatching the final score back to
        #       the user, not here. This is to prevent the checks themselves
        #       from manipulating metadata and going undetected.
        #
        # END COMMON CHECKS
        # ---------------------------------------------------------------------

        # perform score *specific* checks here
        NseMetaHandler.post_score_checks(meta_score)

        return meta_score

    @staticmethod
    def extract_result_from_metascore(
        meta_score: NseMetaScore,
        orig_meta_input: NseMetaInput,
    ) -> XarrayLike:
        """
        If inputs were originally dataarrays, downgrades the score to a
        dataarray, naming it with the score i.e. "NSE".

        Otherwise returns the score as is (as a dataset)

        Checks that the meta score is consistent with the meta input - i.e. no
        corruption has happened to the underlying datasets or metadata. see:
        ``check_metadata_consistency``

        Args:
            meta_score: the meta score to extract the raw NSE score from
            ref_meta_input: reference to the meta input for checking
                consistency
        """
        # metadata consistency should be the last thing checked before
        # extracting the result
        NseUtils.check_metadata_consistency(
            meta_score=meta_score,
            orig_meta_input=orig_meta_input,
        )

        ret_nse = meta_score.components.nse

        # demote if originally a data array and force the name to "NSE"
        if meta_score.ref_meta_input.is_dataarray:
            # returns None if extraction fails
            ret_nse = NseUtils.try_extract_singleton_dataarray(meta_score.components.nse)
            # safety: check_metadata_consistency should have handled this
            assert ret_nse is not None
            # force score to NSE
            ret_nse.name = NseUtils.SCORE_NAME

        return ret_nse

    @staticmethod
    def pre_score_checks(meta_input: NseMetaInput):
        """
        Checks to run before computing the score - this is specific to a
        particular score.

        For NSE:
            - checks that that there are sufficient obs for variance
              calculations
        """
        merged_sizes: dict[Hashable, int] = NseUtils.merge_sizes(
            meta_input.datasets.obs,
            meta_input.datasets.fcst,
            meta_input.datasets.weights,
        )
        NseUtils.check_gathered_dims(
            meta_input.gathered_dims,
            merged_sizes,
        )

    @staticmethod
    def post_score_checks(meta_score: NseMetaScore):
        """
        Checks to run before computing the score - this is specific to a
        particular score. Also runs common consistency checks between
        ``meta_input`` and ``meta_score``.

        for NSE:
            - asserts that all NSE outputs are <=1
        """
        # safety: dev/testing assert that the output components are datasets.
        assert (
            isinstance(meta_score.components.fcst_error, xr.Dataset)
            and isinstance(meta_score.components.obs_variance, xr.Dataset)
            and isinstance(meta_score.components.nse, xr.Dataset)
        )

        # safety: for dev/testing only since weights cannot be negative,
        #     neither can the score.
        # optimization: we don't want this check to run in optimized mode
        #     because dask may trigger a compute, hence an assert makes
        #     more sense.
        vars_in_expected_score_range = (
            xr.ufuncs.isnan(meta_score.components.nse) | (meta_score.components.nse <= 1)
        ).all()

        assert all(vars_in_expected_score_range)


class NseUtils(SimpleNamespace):
    """
    Helper class with static methods for use in NSE computations only. Also
    contains error msg strings specific to NSE, and some utility used by
    NseMetaHandler.

    FUTUREWORK:

        A lot of utility in this function can be abstracted to somewhere more
        general. Only the error messages need to be score specific.

    .. note::

        This class is essentially a simple namespace type (dict) in a
        class-like structure for improved clarity in documentation and
        typehints.

    .. important::

        To simplify things, and reduce the chance of error and
        incompatibilities of mixed types and APIs - all utils are assumed to be
        compatible with "Meta" types.

    .. important ::

        It is worth mentioning that the utility module is not very strict with
        runtime type assertions, other than some important functions
        highlighted above.

        This is intentional as this would increase code bloat - and potential
        circular logic.  Instead it assumes that this is performed at a higher
        level at the earliest point of the call chain before computing the
        scores, or via implicit checks within scores.utils which the functions
        in this namespace rely on heavily.
    """

    # score name used for promoted dataarrays
    SCORE_NAME: str = "NSE"

    # datasets require a name - used when promoting a dataarray without a name.
    # NOTE: this doesn't actually replace the input dataarray name.
    _DATAARRAY_TEMPORARY_NAME: str = "__NONAME"

    ERROR_CORRUPT_METADATA: str = f"""
    {SCORE_NAME}: CRITICAL FAILURE! The internal state of either NseMetaInputs
    or NseMetaScore has been corrupt.  This should not happen and is either a
    BUG or is being run on a mutated version of the code.

    Please raise an issue on github.com/nci/scores citing this error.
    """

    ERROR_NO_DIMS_TO_REDUCE: str = f"""
    {SCORE_NAME}: need at least one dimension to be reduced. Check that
    `preserve_dims` is not preserving all dimensions, OR check that
    `reduce_dims` is specified with at least one dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = f"""
    {SCORE_NAME}: need at least one dimension that is being reduced where its
    length > 1.  Check that the input provided have at least 2 or more data
    points along the dimensions being reduced. It is not possible to calculate
    a non-zero obs variance from 1 point.
    """

    ERROR_MIXED_XR_DATA_TYPES: str = f"""
    {SCORE_NAME}: Triggered during NSE calculations` check `fcst`, `obs` and
    `weights` are of the same type `xr.Dataset` OR `xr.DataArray` EXCLUSIVELY;
    NOT a mix of the two types.
    """

    WARN_ZERO_OBS_VARIANCE: str = f"""
    {SCORE_NAME}: possible divide by zero - at least one element in the reduced
    obs variance array is 0.  Any divide by zero entries will be filled in as
    `np.nan` if the forecast error is also 0, otherwise it will be `-np.inf`.
    This is so that any other valid entries are still computed and returned.
    The user should still verify that zero obs variance is expected for the
    given input data.
    """

    @staticmethod
    def check_all_same_type(*xrlike):
        """
        Checks that all XarrayLike inputs are of the same type, i.e. either ALL
        datasets OR (exclusive OR i.e. xor) ALL dataarrays.

        Raises:
            TypeError: If mixed types are provided or if input isn't
                XarrayLike.
        """
        assert len(xrlike) > 0

        xrlike_remove_none = [_x for _x in xrlike if _x is not None]

        if not all_same_xarraylike(xrlike_remove_none):
            raise TypeError(NseUtils.ERROR_MIXED_XR_DATA_TYPES)

    @staticmethod
    def xarraylike_to_dataset(xrlike: XarrayLike) -> xr.Dataset:
        """
        Helper that promotes dataarrays to datasets or does nothing if None or
        a dataset is given.

        .. note ::

            operations between dataarrays are not guarenteed to preserve names.
            If the user is working with dataarrays, it is assumed that
            preserving names is not a major requirement (unlike datasets).
        """
        ret_ds = xrlike

        if isinstance(xrlike, xr.DataArray):
            # keep original name for assertion - see below
            original_name = xrlike.name

            # obfuscate the name of the dataarray when it is promoted to
            # dataset so that operations can broadcast with each other.
            # NOTE: this doesn't actually change the name of the underlying
            # data array.
            ret_ds = xrlike.to_dataset(name=NseUtils._DATAARRAY_TEMPORARY_NAME)

            # assertion below to make sure this behaviour does not change
            # between xarray versions.  assert that referenced dataarray still
            # has the same name.
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
        orig_meta_input: NseMetaInput,
    ):
        """
        Checks against corruption of data. This usually cannot happen unless
        there is a bug in the code that tries to copy and replace one of the
        named tuples - normally their shallow properties shouldn't be mutable.

        Raises:
            RuntimeError: If a corruption is detected this is critical failure.
                User will be prompted to create a github issue, so a developer
                can address it.
        """
        corrupt = False

        def _ds_has_single_key(fieldname):
            """
            Helper that checks whether a component in meta_score has a single
            dataarray (as determined by the number of keys in `data_vars` key)
            """
            _ds = getattr(meta_score.components, fieldname)
            _single_key = len(_ds.data_vars.keys()) == 1
            return _single_key

        # is_dataarray should be consistent
        if meta_score.ref_meta_input.is_dataarray != orig_meta_input.is_dataarray:
            corrupt = True

        # A dataset produced from a dataarray MUST only have ONE data_var key
        if meta_score.ref_meta_input.is_dataarray:
            all_single_key = all(map(_ds_has_single_key, meta_score.components._fields))
            if not all_single_key:
                corrupt = True

        if corrupt:
            raise RuntimeError(NseUtils.ERROR_CORRUPT_METADATA)

        # safety: for dev/testing only, this is just a guard against accidental
        # deep copying of data. The meta_input tuple should always hold the same
        # reference once created (for any given call).
        assert meta_score.ref_meta_input is orig_meta_input

    @staticmethod
    def merge_sizes(*ds) -> dict[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key)
        in the given ``xr.Dataset`` object(s).

        Args:
            *ds: Variadic argument of each of type ``xr.Dataset``
        """
        assert len(ds) > 0

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
            merged_sizes: dict[Hashable, int] = acc_sizes | dict(curr_ds.sizes)
            return merged_sizes

        ret_sizes: dict[Hashable, int] = functools.reduce(_merge_single, ds, {})

        return ret_sizes

    @staticmethod
    def warn_nonzero_obsvar(obs_var: xr.Dataset):
        """
        Warns if at least one obs variance term is zero in any variable in the
        dataset.
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
        Checks that gathered dimensions has at least one entry (key or hash)
        AND at least one of the gathered dimensions has more than 1 data point.

        (i.e. there exists a ``key`` in ``gathered_dims`` where
        ``dim_sizes[key] > 1``.)

        Args:
            gathered_dims: set of dimension names.
            dim_sizes: map between dimension names and their lengths.

        Raises:
            DimensionError: if gathered dimensions are not compatible

        .. note::

            An empty set of dimensions is normally allowed in scores, i.e. the
            score is computed for each element.

            This is not suitable for NSE, which requires calculation of
            variance, without more than 1 data point in each reduced group,
            variance is guarenteed to be zero for all entries.

        .. note::

            Commonly used in conjunction with :py:meth:`NseUtils.merge_sizes`

        .. see-also::

            For more info on the concept of ``sizes`` see
            :py:meth:`xarray.Dataset.sizes` or
            :py:meth:`xarray.DataArray.sizes`
        """
        # this check is specific to NSE which requires at least one reduce dim
        if len(list(gathered_dims)) == 0:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)

        dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

        if not dim_has_more_than_one_obs:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_WITH_MULTIPLE_OBS)

    @staticmethod
    def check_vars_have_reducible_dims(ds: xr.Dataset, ref_dims: FlexibleDimensionTypes):
        """
        Raises error if any var is missing a dimension by comparing with the
        reference.
        """
        # ignore if ref_dims is empty
        if len(list(ref_dims)) == 0:
            return

        # otherwise (if ref_dims has at least one entry),
        # check that each variable has at least one dim contained in ref_dims
        var_names = list(ds.data_vars.keys())
        fn_checkvar = lambda _v: any(_d in ref_dims for _d in ds[_v].dims)
        atleastone_reducedim = all(map(fn_checkvar, var_names))

        if not atleastone_reducedim:
            raise DimensionError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)
