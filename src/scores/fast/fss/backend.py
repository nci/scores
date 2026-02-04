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

    # internal buffers
    _obs_pop: npt.NDArray[np.int64] = field(init=False)
    _fcst_pop: npt.NDArray[np.int64] = field(init=False)
    _obs_img: npt.NDArray[np.int64] = field(init=False)
    _fcst_img: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization checks go here.
        """
        self._check_dims()
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

    def _apply_event_threshold(self):
        """
        Default implementation of converting the input fields into binary fields by comparing
        against a event threshold; using numpy. This is a relatively cheap operation, so a specific
        implementation in derived backends is optional.
        """
        _op = self.threshold_operator.get()
        self._obs_pop = _op(self.obs, self.event_threshold)
        self._fcst_pop = _op(self.fcst, self.event_threshold)
        return self

    def _compute_integral_field(self):
        """
        Computes the rolling windowed sums over the entire observation & forecast fields. The
        resulting "integral field (or image)" can be cached in `self._obs_img` for observations, and
        `self._fcst_img` for forecast.

        See also: https://en.wikipedia.org/wiki/Summed-area_table

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
            p_o: observation populace > event_threshold, in one window
            p_f: forecast populace > event_threshold, in one window
            sum_w: sum over all windows
        ````

        In order to accumulate scores over non spatial dimensions at a higher level operation, we
        need to keep track of the de-composed sums as they need to be accumulated separately.

        The "fss components", hence, consist of:
        - `power_diff :: float = sum_w((p_o - p_f)^2)`
        - `power_obs :: float = sum_w(p_o^2)`
        - `power_fcst :: float = sum_w(p_f^2)`

        **WARNING:** returning sums of powers can result in overflows on aggregation. It is advisable
        to return the means instead, as it is equivilent with minimal computational trade-off.

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
        fss: np.float = 0.0

        if denom > 0.0:
            fss = 1.0 - diff / denom

        fss_clamped = max(min(fss, 1.0), 0.0)

        return fss_clamped
