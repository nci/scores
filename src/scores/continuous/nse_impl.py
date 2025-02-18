"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import scores.continuous.mse
import weakref
import functools

from scores.typing import override, DimName, DimNameCollection, XarrayLike, check_dimnamecollection, dimnamecollection_to_list
from scores.utils import gather_dimensions, check_weights_positive


ERROR_SCORE_DIRECT_INIT_DISALLOWED: str = """
Internal Class: NseScore is being directly initialised - this is disallowed, use `NseScoreBuilder`
instead. This is not a user error - if it has been triggered please raise an issue in github.
"""

ERROR_SCORE_ALREADY_BUILT: str = """
Internal Class: `NseScore` has already been built - re-use the output of this builder instead of
rebuilding, to avoid data bloat. This is not a user error - if it has been triggered please raise an
issue in github.
"""

WARN_TIME_DIMENSION_REQUIRED: str = """
NSE is usually reduced along the time dimension. This is required, otherwise, the variance of the
observations cannot be computed.
"""


class NseUtils:
    """
    Static helper class
    """

    @staticmethod
    def check_and_gather_dimensions(
        obs: DimNamesCollection,
        fcst: DimNamesCollection,
        weights: DimNamesCollection | None,
        reduce_dims: DimNamesCollection | None,
        preserve_dims: DimNamesCollection | None,
    ) -> list[DimName]:
        # check weights conform to NSE computations
        NseUtils.check_weights(weights)
        # perform default gather dimensions
        gathered_dims: set[Hashable] = scores.utils.gather_dimensions(
            obs_dims,
            fcst_dims,
            weights_dims,
            reduce_dims,
            preserve_dims,
        )
        # check result has at least one reducable dimension - required to calculate obs variance
        NseUtils.check_gathered_dims(gathered_dims)
        ret_dims: list[DimName] = dimnamecollection_to_list(gathered_dims)
        return ret_dims

    @staticmethod
    def check_weights(weights: XarrayLike) -> None:
        check_weights_positive(weights, msg="`NSE` must have positive weights.")

    @staticmethod
    def check_nonzero_obsvar(obs_var: XarrayLike) -> None:
        """
        Warns if at least one obs variance term is zero.
        """
        if np.any(obs_var == 0):
            raise NotImplementedError("TODO")  # TODO: raise warning

    @staticmethod
    def check_gathered_dims(gathered_dims: DimNameCollection):
        """
        Currently, NSE only supports a single time dimension per call. Other dimensions can still be
        reduced by manually specifying `reduce_dims` or `preserve_dims`.
        """
        check_dimnamecollection(gathered_dims)
        gathered_dims = dimnamecollection_to_list(gathered_dims)
        if len(gathered_dims) == 0:
            raise NotImplementedError("TODO")  # TODO: raise error
        if no_gathered_dim_has_more_than_1_datum:  # requires finding len of each gathered dim
            raise NotImplementedError("TODO")  # TODO: raise error


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
    ref_score: weakref.ref = None

    def build(
        self,
        *,  # force KW_ONLY
        obs: ...,
        fcst: ...,
        reduce_dims: ...,
        preserve_dims: ...,
        weights: ...,
        is_angular: ...
    ):
        """
        Builds an `NseScore` object
        - performs any checks and preliminary computations.
        - stores a weak reference to the score.
        - cannot be run twice (unless the score object has been garbage collected).
        """
        if ref_score is not None and ref_score() is not None:
            raise RuntimeError(ERROR_SCORE_ALREADY_BUILT)

        reduce_dims = NseUtils.check_and_gather_dimensions(...)

        score = NseScore(
            builder=weakref.ref(self),
            fcst=fcst,
            obs=obs,
            reduce_dims=reduce_dims,
            weights=weights,
            is_angular=is_angular,
        )

        self.score = weakref.ref(score)

        return score


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
    reduce_dims: DimNameCollection
    weights: XarrayLike
    is_angular: bool = False

    def __post_init__(ref_builder: NseScoreBuilder) -> None:
        if ref_builder is None or ref_builder() is None:
            raise RuntimeError(ERROR_SCORE_DIRECT_INIT_DISALLOWED)

    def calculate(self) -> XarrayLike:
        obs_var: XarrayLike = self.calculate_obs_variance()
        fcst_err: XarrayLike = self.calculate_forecast_error()
        nse: XarrayLike | None = None

        NseUtils.check_nonzero_obsvar(obs_var)

        # intentional divide="ignore", as warning is already raised above.
        # `numpy` will fill divide by zero elements with NaNs.
        with np.errstate(divide="ignore"):
            nse = 1.0 - (fcst_err / obs_var)

        # explicit checking and casting
        NseUtils.check_nse_type(nse)
        NseUtils.verify_nse_structure(nse)
        cast(nse, XarrayLike)

        return nse

    def mse_callback(self) -> Callable[..., XarrayLike]:
        return functools.partial(
            mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    def calculate_forecast_error(self) -> ...:
        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        fcst_err: XarrayLike = mse_cb(self.fcst, self.obs)

        return fcst_err

    def calculate_obs_variance(self) -> ...:
        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        utils.check_dims(self.obs, self.reduce_dims, "superset")

        obs_mean: XarrayLike = self.obs.mean(dims=self.reduce_dims)
        obs_var: XarrayLike = mse_cb(obs_mean, self.obs)

        return obs_var
