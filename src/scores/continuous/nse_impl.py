"""
Internal module used to support the computation of NSE. Not to be used directly.
"""

import scores.continuous.mse
import weakref
import functools

from scores.typing import override


ERROR_SCORE_DIRECT_INIT_DISALLOWED: str = """
Internal Class: NseScore is being directly initialised - this is disallowed, use `NseScoreBuilder`
instead. This is not a user error - if it has been triggered please raise an issue in github.
"""

ERROR_SCORE_ALREADY_BUILT: str = """
Internal Class: `NseScore` has already been built - re-use the output of this builder instead of
rebuilding, to avoid data bloat. This is not a user error - if it has been triggered please raise an
issue in github.
"""


@dataclass
class Checkable:
    """
    Base class to dictatke if an input is checked or not
    """

    def __post_init__(self):
        self.check()

    def check():
        # default: do nothing - force checked
        self.checked = True


@dataclass
class CheckableWeights(Checkable):
    _: KW_ONLY
    weights: weakref.ref

    @override
    def check():
        check_weights_positive(self.unwrap(), context="NSE requires positive weights.")


@dataclass
class CheckableFcst(Checkable):
    # `gather_dimensions handles forecast checks`
    pass


class NseUtils:
    """
    Static helper class
    """

    @staticmethod
    def gather_dimensions() -> DimCollection:
        # perform default gather dimensions
        ret_dims = scores.utils.gather_dimensions(
            obs,
            fcst,
            weights,
            reduce_dims,
            preserve_dims,
        )

        # perform additional check to ensure that `time_dim` is present in all datasets
        for xr_data in [obs, fcst, weights]:
            # `.sizes` is consistent between datasets and dataarrays
            if "time_dim" not in xr_data.sizes.keys():
                raise NotImplementedError("TODO")

        # perform additional check to ensure that `time_dim` is present in the result
        if ret_dims is None or len(ret_dims) == 0:
            raise NotImplementedError("TODO")

    @staticmethod
    def check_weights():
        pass

    @staticmethod
    def prepare_dims_from_input(
        preserve_dims: DimCollection,
        reduce_dims: DimCollection,
        time_dim: DimCollection,
    ):
        NseUtils.gather_dimensions(obs, fcst, weights, reduce_dims, time_dim)
        if preserve_dims == "any":
            ValueError("'any' is not supported for `preserve_dims` use `reduce_dims` instead")


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
    ):
        """
        Builds an `NseScore` object
        - performs any checks and preliminary computations.
        - stores a weak reference to the score.
        - cannot be run twice (unless the score object has been garbage collected).
        """

        if ref_score is not None and ref_score() is not None:
            raise RuntimeError(ERROR_SCORE_ALREADY_BUILT)

        # --- perform checks here ---

        # ---
        score = NseScore(
            fcst=fcst_error,
            obs_variance=obs_variance,
            builder=weakref.ref(self),
            mse_cb=mse_cb,
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
    weights: XarrayLike
    reduce_dims: XarrayLike

    def mse_callback(self) -> Callable[..., XarrayLike]:
        return functools.partial(
            mse,
            reduce_dims=self.reduce_dims,
            weights=self.weights,
            is_angular=self.is_angular,
        )

    @static_method
    def calculate_forecast_error(self) -> ...:
        raise NotImplementedError("TODO")

    @static_method
    def calculate_obs_variance(self) -> ...:
        raise NotImplementedError("TODO")

    def __post_init__(ref_builder: NseScoreBuilder):
        # NOTE: dask can cause the following to be delayed
        fcst_error = NseScore.calculate_forecast_error()
        obs_variance = NseScore.calculate_obs_variance()
        if ref_builder is None or ref_builder() is None:
            raise RuntimeError(ERROR_SCORE_DIRECT_INIT_DISALLOWED)
