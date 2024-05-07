"""
This module contains methods related to the FSS score
"""
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from scores.typing import FlexibleDimensionTypes


class FssComputeMethod(Enum):
    NUMPY = 1
    NUMBA_NATIVE = 2
    NUMBA_PARALLEL = 3
    RUST_NATIVE = 4
    RUST_OCL = 5


def fss(
    fcst: npt.NDArray[np.float64],  # TODO: change to XarrayLike once support is implemented
    obs: npt.NDArray[np.float64],  # TODO: change to XarrayLike once support is implemented
    *,
    threshold: np.float64,  # TODO: support for multiple thresholds
    window: Tuple[int, int],  # TODO: support for multiple windows
    compute_method: Optional[FssComputeMethod] = FssComputeMethod.NUMPY,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,  # FIXME: unused
    preserve_dims: Optional[FlexibleDimensionTypes] = None,  # FIXME: unused
    weights: Optional[xr.DataArray] = None,  # FIXME: unused
    check_args: bool = False,  # FIXME: unused
):
    """
    Calculates the FSS (Fractions Skill Score) for forecast and observed data.
    For an explanation of the FSS, and implementation considerations,
    see: [Fast calculation of the Fractions Skill Score][fss_ref]

    [fss_ref]:
    https://www.researchgate.net/publication/269222763_Fast_calculation_of_the_Fractions_Skill_Score

    Currently only supports a single field.

    Compute Methods:
    - Supported
        - `FssComputeMethod.NUMPY` (default)
    - Optimized
        - `FssComputeMethod.NUMBA_NATIVE`
        - `FssComputeMethod.NUMBA_PARALLEL`
    - Experimental
        - `FssComputeMethod.RUST_NATIVE`
        - `FssComputeMethod.RUST_OCL` (highly unstable, but potentially very fast)

    TODO: docstring is WIP
    """
    # Temporary assert unimplemented arguments, as aggregation is not implemented.
    # TODO: To be removed once implemented.
    assert reduce_dims is None, "`reduce_dims` is not implemented."
    assert preserve_dims is None, "`preserve_dims` is not implemented."
    assert weights is None, "`weights` is not implemented."
    assert check_args is True, "`check_args` is not implemented."


@dataclass
class FssBackend:
    """
    Backend for computing fss.
    """

    fcst: npt.NDArray[np.float64]
    obs: npt.NDArray[np.float64]
    _: KW_ONLY
    thr: np.float64
    window: Tuple[int, int]
    obs_pop: npt.NDArray[np.int64] = field(init=False)
    fcst_pop: npt.NDArray[np.int64] = field(init=False)

    def _apply_threshold(self) -> FssBackend:
        raise NotImplementedError("_compute_integral_field not implemented")
        return self

    def _compute_integral_field(self) -> FssBackend:
        raise NotImplementedError("_compute_integral_field not implemented")
        return self

    def _compute_fss_score(self) -> np.float64:
        raise NotImplementedError("_compute_integral_field not implemented")
        return 0.0

    def compute_fss(self) -> np.float64:
        fss_result = self._apply_threshold()._compute_integral_field()._compute_fss_score()
        return fss_result


@dataclass
class FssNumpy(FssBackend):
    compute_method = FssComputeMethod.NUMPY

    def _apply_threshold(self) -> FssNumpy:
        self.obs_pop = self.obs > self.thr
        self.fcst_pop = self.fcst > self.thr
        return self

    def _compute_integral_field(self) -> FssNumpy:
        raise NotImplementedError("_compute_integral_field not implemented")
        return self

    def _compute_fss_score(self) -> np.float64:
        raise NotImplementedError("_compute_integral_field not implemented")
        return 0.0
