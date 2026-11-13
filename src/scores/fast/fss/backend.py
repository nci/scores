"""
Backend implementation of the FSS algorithm. Currently calculations are
performed mainly by the `numpy` backend.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt

from scores.fast.fss.typing import FssComputeMethod, FssDecomposed
from scores.utils import BinaryOperator, CompatibilityError, DimensionError


@dataclass
class FssBackend:  # pylint: disable=too-many-instance-attributes
    """
    Abstract base class for computing fss.

    required methods:
        - :py:meth:`_compute_fss_components`
        - :py:meth:`_compute_integral_field`
    """

    compute_method = FssComputeMethod.INVALID

    fcst: npt.NDArray[np.float64]
    obs: npt.NDArray[np.float64]
    # KW_ONLY in dataclasses only supported for python >=3.10
    # _: KW_ONLY
    event_threshold: np.float64
    threshold_operator: BinaryOperator
    window_size: Tuple[int, int]
    zero_padding: bool
    # minimum proportion of non-missing points required for a window to be included in the score
    min_valid_proportion: float = 0.0

    # internal buffers
    _obs_pop: npt.NDArray[np.int64] = field(init=False)
    _fcst_pop: npt.NDArray[np.int64] = field(init=False)
    _valid: npt.NDArray[np.bool_] = field(init=False)
    _obs_img: npt.NDArray[np.float64] = field(init=False)
    _fcst_img: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization checks go here.
        """
        self._check_dims()
        self._check_min_valid_proportion()
        self._check_compatibility()

    def _check_compatibility(self):  # pragma: no cover
        raise CompatibilityError(
            f"Unable to run `{self.compute_method}`. Are you missing extra/optional dependencies in your install?"
        )

    def _check_dims(self):
        if self.fcst.shape != self.obs.shape:
            raise DimensionError("fcst and obs shapes do not match")
        if (
            self.window_size[0] > self.fcst.shape[0]
            or self.window_size[1] > self.fcst.shape[1]
            or self.window_size[0] < 1
            or self.window_size[1] < 1
        ):
            raise DimensionError(
                "invalid window size, `window_size` must be smaller than input data shape and greater than 0"
            )

    def _check_min_valid_proportion(self):
        if not 0 <= self.min_valid_proportion <= 1:
            raise ValueError("`min_valid_proportion` must be between 0 and 1 (inclusive)")

    def _apply_event_threshold(self):
        """
        Default implementation of converting the input fields into binary fields by comparing
        against a event threshold; using numpy. This is a relatively cheap operation, so a specific
        implementation in derived backends is optional.

        Missing data (NaN) is synchronised between the two fields: a point that is NaN in either
        ``fcst`` or ``obs`` is treated as missing in both. The mask of non-missing points is
        cached in ``self._valid`` so that derived backends can exclude missing points when
        computing fractions. Missing points are set to 0 (non-event) in the binary fields so that
        cumulative sums remain finite; they are then excluded via ``self._valid``.
        """
        _op = self.threshold_operator.get()
        missing = np.isnan(self.fcst) | np.isnan(self.obs)
        self._obs_pop = np.where(missing, 0, _op(self.obs, self.event_threshold))
        self._fcst_pop = np.where(missing, 0, _op(self.fcst, self.event_threshold))
        self._valid = ~missing
        return self

    def _compute_integral_field(self):
        """
        Computes the event fractions for every window position over the entire observation &
        forecast fields. The resulting fraction fields can be cached in `self._obs_img` for
        observations, and `self._fcst_img` for forecast.

        Missing data must be handled as follows (Skok and Kosovelj, 2025):

        - The fraction of a window is the number of events divided by the number of
          non-missing points in that window, i.e. missing points (where ``self._valid`` is
          ``False``) are excluded from both the numerator and the denominator rather than being
          treated as non-events. Points in the zero-padded border (if ``zero_padding`` is set)
          count as non-missing, non-event points.
        - A window is included in the score if it contains at least one non-missing point, and
          the proportion of non-missing points in the window is at least
          ``self.min_valid_proportion``. The fractions of all other windows must be set to NaN
          in both fields, so that they do not contribute to the score.

        See also: https://en.wikipedia.org/wiki/Summed-area_table

        Reference: Skok, G. and Kosovelj, K., 2025: Smoothing and spatial verification of global
        fields. Geoscientific Model Development, 18, 7417–7433,
        https://doi.org/10.5194/gmd-18-7417-2025.

        Note: for non-`python` backends e.g. `rust`, its often the case that the integral field is
        computed & cached in the native language. In which case this method can just be overriden
        with a `return self`.
        """
        raise NotImplementedError(f"_compute_integral_field not implemented for {self.compute_method}")

    def _compute_fss_components(self) -> FssDecomposed:
        """
        FSS is roughly defined as
        ```
            fss = 1 - sum_w((p_o - p_f)^2) / (sum_w(p_o^2) + sum_w(p_f^2))

            where,
            p_o: fraction of (non-missing) observation points that are events, in one window
            p_f: fraction of (non-missing) forecast points that are events, in one window
            sum_w: sum over all windows that have a non-NaN fraction
        ````

        In order to accumulate scores over non spatial dimensions at a higher level operation, we
        need to keep track of the de-composed sums as they need to be accumulated separately.

        The "fss components", hence, consist of:
        - `power_diff :: float = sum_w((p_o - p_f)^2)`
        - `power_obs :: float = sum_w(p_o^2)`
        - `power_fcst :: float = sum_w(p_f^2)`

        Windows excluded due to insufficient non-missing data have NaN fractions (see
        :py:meth:`_compute_integral_field`) and do not contribute to the sums. Adding the
        components of multiple fields is therefore equivalent to pooling all of their windows. If
        no windows are included, all components are 0 (and the FSS is NaN).

        Note: the fractions are in [0, 1], so each term is at most 1 and the sums cannot overflow.

        Returns:
            Tuple[float, float, float]: (power_fcst, power_obs, power_diff) as described above
        """
        raise NotImplementedError(f"`_compute_fss_components` not implemented for {self.compute_method}")

    def compute_fss_decomposed(self) -> FssDecomposed:
        """
        Calls the main pipeline to compute the fss score.

        Returns the decomposed scores for aggregation. This is because aggregation cannot be
        done on the final score directly.

        Note: FSS computations of stand-alone 2-D fields should use :py:meth:`compute_fss` instead.
        """
        # fmt: off
        return (
            self._apply_event_threshold()  # pylint: disable=protected-access
                ._compute_integral_field()
                ._compute_fss_components()
        )
        # fmt: on

    def compute_fss(self) -> np.float64:
        """
        Uses the components from :py:method:`compute_fss_decomposed` to compute the final
        Fractions Skill Score.
        """
        (fcst, obs, diff) = self.compute_fss_decomposed()

        denom = fcst + obs

        if denom > 0.0:
            fss = 1.0 - diff / denom
            fss_clamped = max(min(fss, 1.0), 0.0)
        else:
            fss_clamped = np.float64(np.nan)
        return fss_clamped
