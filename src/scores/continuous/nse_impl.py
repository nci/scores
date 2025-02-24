"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import scores.continuous.mse
import weakref
import functools
from typing import Unpack, Mapping, TypeAlias, Hashable

from scores.typing import XarrayLike, FlexibleDimensionTypes
from scores.utils import gather_dimensions, check_weights_positive


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

    ERROR_NO_DIMS_TO_REDUCE: str = """
    NSE needs at least one dimension to be reduced. Check that `preserve_dims` is not preserving all
    dimensions, OR check that `reduce_dims` is specified with at least one dimension.
    """

    ERROR_NO_DIMS_WITH_MULTIPLE_OBS: str = """
    NSE needs at least one dimension that is being reduced where its length > 1. Check that the
    input provided have at least 2 or more data points along the dimensions being reduced. It is not
    possible to calculate a non-zero obs variance from 1 point.
    """

    WARN_TIME_DIMENSION_REQUIRED: str = """
    NSE is usually reduced along the time dimension. This is required, otherwise, the variance of
    the observations cannot be computed.
    """

    WARN_ZERO_OBS_VARIANCE: str = """
    Possible divide by zero: at least one element in the reduced obs variance array is 0. Any divide
    by zero entries will be filled in as `np.nan`. This is so that any other valid entries are still
    computed and returned. The user should still verify that zero obs variance is expected for the
    given input data.
    """

    @staticmethod
    def check_and_gather_dimensions(
        fcst: XarrayLike,
        obs: XarrayLike,
        weights: XarrayLike | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> FlexibleDimensionTypes:
        """
        Checks the dimension compatibilty of the various input arguments.
        """
        # check weights conform to NSE computations
        NseUtils.check_weights(weights)

        # perform default gather dimensions
        ret_dims: FlexibleDimensionTypes = scores.utils.gather_dimensions(
            obs_dims,
            fcst_dims,
            weights_dims,
            reduce_dims,
            preserve_dims,
        )

        merged_sizes: Mapping[Hashable, int] = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension - required to calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def merge_sizes(*xr_data: Unpack[tuple[XarrayLike]]) -> Mapping[Hashable, int]:
        """
        Merges the maps that contain the size (value) for each dimension (key) (key) in a given
        `XarrayLike` object.
        """
        # TODO: this should be a general utility
        ret_sizes = functools.reduce(lambda _acc, _x: _acc | _x,  xr_data)
        return ret_sizes

    @staticmethod
    def check_weights(weights: XarrayLike) -> None:
        """
        Wrapper around :py:func:`~scores.utils.check_weights_positive`, with a context specific
        warning message.
        """
        check_weights_positive(weights, msg="`NSE` must have positive weights.")

    @staticmethod
    def check_nonzero_obsvar(obs_var: XarrayLike) -> None:
        """
        Warns if at least one obs variance term is zero.
        """
        if np.any(obs_var == 0):
            warnings.warn(NseUtils.WARN_ZERO_OBS_VARIANCE)

    @staticmethod
    def check_gathered_dims(gathered_dims: FlexibleDimensionsTypes, dim_sizes: Mapping[Hashable, int]) -> None:
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
        # TODO: this should be a general utility
        if len(gathered_dims) == 0:
            raise KeyError(NseUtils.ERROR_NO_DIMS_TO_REDUCE)

        dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

        if not dim_has_more_than_one_obs:
            raise NotImplementedError(NseUtils.ERROR_NO_DIMS_WITH_MULTIPLE_OBS)


@dataclass
class NseScoreBuilder:
    """
    Internal namespace that builds the `NseScore` object which is used to compute the NSE score.

    The scope of this class is to:
    - perform any validity checks on the input
    - resolve optional inputs
    - raise any issues if the above are not conformant with the "NSE" score
    """

    # weakref to scores - to avoid circular reference
    ref_nsescorer: weakref.ref | None = None

    def build(
        self,
        *,  # force KW_ONLY - note
        fcst: XarrayLike,
        obs: XarrayLike,
        reduce_dims: FlexibleDimensionsTypes | None = None,
        preserve_dims: FlexibleDimensionsTypes | None = None,
        weights: XarrayLike | None = None,
        is_angular: bool | None = False,
    ):
        """
        Builds an ``NseScore`` object
        - performs any checks and preliminary computations.
        - stores a weak reference to the score.
        - cannot be run twice (unless the score object has been garbage collected).

        raises a ``RuntimeError`` if ``NseScore`` has already been built.

        .. note::

            The ``RuntimeError`` raised would be an internal error is a development safety so that
            `NseScore` isn't instantiated directly. If this error is raised, the user would be
            prompted to raise an issue.
        """
        if ref_nsescorer is not None and self.ref_nsescorer() is not None:
            raise RuntimeError(NseUtils.ERROR_SCORE_ALREADY_BUILT)

        reduce_dims = NseUtils.check_and_gather_dimensions(
            fcst=fcst,
            obs=obs,
            weights=weights,
            reduce_dims=reduce_dims,
            preserve_dims=preserve_dims,
        )

        ret_nsescorer = NseScore(
            builder=weakref.ref(self),
            fcst=fcst,
            obs=obs,
            reduce_dims=reduce_dims,
            weights=weights,
            is_angular=is_angular,
        )

        self.ref_nsescorer = weakref.ref(score)

        return ret_nsescorer


# fields can only be set once
@dataclass(frozen=True)
class NseScore:
    """
    Internal class that performs the actual scoring logic.

    Assumes that all setups and checks have been done by :py:class:`NseScoreBuilder`, and hence should
    not be initialised directly. Instead, ``NseScoreBuilder`` should be used to do any setup.

    .. math::
        \\text{nse} = 1 - \\text{weighted_fct_error} / \\text{weighted_obs_variance}

    In order to produce this result, ``NseScore`` requires the ``fcst_error``, the ``obs_variance``
    as well as a callback to ``mse`` which is used to assign weighting and perform any appropriate
    reduction.

    The rest of the operations are simple broadcastable operations e.g. subtraction and division

    .. important ::

        divide by zero errors are intentionally ignored here and ``numpy`` will automatically
        fill them with ``np.nan``. ``NseScoreBuilder`` the class that constructs a ``NseScore``
        object, already handles warnings during these scenarios.
    """

    _: KW_ONLY
    #: ref to builder - to keep it in scope
    builder: NseScoreBuilder
    obs: XarrayLike
    fcst: XarrayLike
    reduce_dims: FlexibleDimensionsTypes | None
    weights: XarrayLike | None
    is_angular: bool = False

    def __post_init__(ref_builder: NseScoreBuilder) -> None:
        """
        Checks referenced builder has not already initialised, otherwise raises a ``RuntimeError``.

        .. note::

            The ``RuntimeError`` raised would be an internal error is a development safety so that
            `NseScore` isn't instantiated directly. If this error is raised, the user would be
            prompted to raise an issue.
        """
        if ref_builder is None or ref_builder() is None:
            raise RuntimeError(NseUtils.ERROR_SCORE_DIRECT_INIT_DISALLOWED)

    def mse_callback(self) -> Callable[..., XarrayLike]:
        """
        Callback helper for calculating :py:func:`scores.mse`, with prefilled keyword args.
        """
        return functools.partial(
            mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    def calculate(self) -> XarrayLike:
        """
        The main calculation function for NSE:
        - denominator (D): calculates the obs variance using a combination of ``xr.mean`` and :py:func:`scores.mse`
        - numerator (N): calculates
        - score = ``1 - N / D`` reduced and broadcast appropriately by ``xarray``.
        """
        obs_var: XarrayLike = self.calculate_obs_variance()
        fcst_err: XarrayLike = self.calculate_forecast_error()
        ret_nse: XarrayLike | None = None

        # intentional divide="ignore", as warning is already raised above.
        # `numpy` will fill divide by zero elements with NaNs.
        with np.errstate(divide="ignore"):
            ret_nse = 1.0 - (fcst_err / obs_var)

        # cast strictly to `XarrayLike` from an `Optional`
        assert isinstance(ret_nse, XarrayLike)  # nosec - for internal safety & tests only.
        cast(ret_nse, XarrayLike)

        return ret_nse

    def calculate_forecast_error(self) -> XarrayLike:
        """
        Calculates the forecast error which is essentially ``scores.mse(fcst, obs)``
        """
        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        ret_fcst_err: XarrayLike = mse_cb(self.fcst, self.obs)

        return ret_fcst_err

    def calculate_obs_variance(self) -> XarrayLike:
        """
        Calculates the obs variance which is essentially ``scores.mse(obs_mean, obs)``.
        """
        utils.check_dims(self.obs, self.reduce_dims, "superset")

        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        obs_mean: XarrayLike = self.obs.mean(dims=self.reduce_dims)
        ret_obs_var: XarrayLike = mse_cb(obs_mean, self.obs)

        NseUtils.check_nonzero_obsvar(ret_obs_var)

        return ret_obs_var
