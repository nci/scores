"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import scores.continuous.mse
import weakref
import functools
from typing import Optional, Unpack, Mapping

from scores.typing import override, DimName, DimNameCollection, XarrayLike, check_dimnamecollection, dimnames_to_list
from scores.utils import gather_dimensions, check_weights_positive


class NseUtils:
    """
    Helper class with static methods for use in NSE computations only.
    """

    ERROR_SCORE_DIRECT_INIT_DISALLOWED: str = """
    Internal Class: NseScore is being directly initialised - this is disallowed, use
    `NseScoreBuilder` instead. This is not a user error - if it has been triggered please raise an
    issue in github.
    """

    ERROR_SCORE_ALREADY_BUILT: str = """
    Internal Class: `NseScore` has already been built - re-use the output of this builder instead
    of rebuilding, to avoid data bloat. This is not a user error - if it has been triggered please
    raise an issue in github.
    """

    WARN_TIME_DIMENSION_REQUIRED: str = """
    NSE is usually reduced along the time dimension. This is required, otherwise, the variance of
    the observations cannot be computed.
    """

    @staticmethod
    def check_and_gather_dimensions(
        fcst: DimNamesCollection,
        obs: DimNamesCollection,
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

        ret_dims: list[DimName] = dimnames_to_list(gathered_dims)

        merged_sizes = NseUtils.merge_sizes(obs, fcst, weights)

        # check result has at least one reducable dimension - required to calculate obs variance
        NseUtils.check_gathered_dims(ret_dims, merged_sizes)

        return ret_dims

    @staticmethod
    def merge_sizes(*xr_data: Unpack[tuple[XarrayLike]]) -> Mapping[DimName, int]:
        # merge operator
        ret_sizes = functools.reduce(lambda _acc, _x: _acc | _x,  xr_data)
        return ret_sizes

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
    def check_gathered_dims(gathered_dims: DimNameCollection, dim_sizes: Mapping[DimName, int]) -> list[DimNames]:
        if len(gathered_dims) == 0:
            raise NotImplementedError("TODO")  # TODO: raise error

        # TODO: use dim_sizes to figure out if gathered dims would do a useful reduction

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
    ref_nsescorer: weakref.ref | None = None

    def build(
        self,
        *,  # force KW_ONLY - note
        fcst: XarrayLike,
        obs: XarrayLike,
        reduce_dims: Optional[DimNameCollection] = None,
        preserve_dims: Optional[DimNameCollection] = None,
        weights: Optional[XarrayLike] = None,
        is_angular: Optional[bool] = False,
    ):
        """
        Builds an `NseScore` object
        - performs any checks and preliminary computations.
        - stores a weak reference to the score.
        - cannot be run twice (unless the score object has been garbage collected).
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
    reduce_dims: DimNameCollection | None
    weights: XarrayLike | None
    is_angular: bool = False

    def __post_init__(ref_builder: NseScoreBuilder) -> None:
        if ref_builder is None or ref_builder() is None:
            raise RuntimeError(NseUtils.ERROR_SCORE_DIRECT_INIT_DISALLOWED)

    def mse_callback(self) -> Callable[..., XarrayLike]:
        return functools.partial(
            mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    def calculate(self) -> XarrayLike:
        obs_var: XarrayLike = self.calculate_obs_variance()
        fcst_err: XarrayLike = self.calculate_forecast_error()
        ret_nse: XarrayLike | None = None

        # intentional divide="ignore", as warning is already raised above.
        # `numpy` will fill divide by zero elements with NaNs.
        with np.errstate(divide="ignore"):
            ret_nse = 1.0 - (fcst_err / obs_var)

        # explicit checking and casting
        NseUtils.check_nse_type(ret_nse)  # todo: isinstance xr.dataset or xr.dataarray
        NseUtils.verify_nse_structure(ret_nse)  # todo: 
        cast(ret_nse, XarrayLike)

        return ret_nse

    def calculate_forecast_error(self) -> ...:
        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        ret_fcst_err: XarrayLike = mse_cb(self.fcst, self.obs)
        return ret_fcst_err

    def calculate_obs_variance(self) -> ...:
        utils.check_dims(self.obs, self.reduce_dims, "superset")

        mse_cb: Callable[..., XarrayLike] = self.mse_callback()
        obs_mean: XarrayLike = self.obs.mean(dims=self.reduce_dims)
        ret_obs_var: XarrayLike = mse_cb(obs_mean, self.obs)

        NseUtils.check_nonzero_obsvar(ret_obs_var)

        return ret_obs_var
