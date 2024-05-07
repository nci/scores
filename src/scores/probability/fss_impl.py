"""
This module contains methods related to the FSS score
"""
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from scores.typing import FlexibleDimensionTypes


class FssComputeMethod(Enum):
    NUMPY = 1  # (default)
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
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,  # FIXME: unused
    preserve_dims: Optional[FlexibleDimensionTypes] = None,  # FIXME: unused
    weights: Optional[xr.DataArray] = None,  # FIXME: unused
    check_args: bool = False,  # FIXME: unused
) -> np.float64:
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
    assert check_args is False, "`check_args` is not implemented."

    fss_backend = FssBackend.get_compute_backend(compute_method)
    fb_obj = fss_backend(fcst, obs, threshold=threshold, window=window)
    fss_score = fb_obj.compute_fss()

    return fss_score


@dataclass
class FssBackend(ABC):
    """
    Backend for computing fss.
    """

    fcst: npt.NDArray[np.float64]
    obs: npt.NDArray[np.float64]
    _: KW_ONLY
    threshold: np.float64
    window: Tuple[int, int]

    # internal buffers
    _obs_pop: npt.NDArray[np.int64] = field(init=False)
    _fcst_pop: npt.NDArray[np.int64] = field(init=False)
    _obs_img: npt.NDArray[np.int64] = field(init=False)
    _fcst_img: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        self._check_dims()

    @staticmethod
    def get_compute_backend(compute_method: FssComputeMethod):
        # We should use python >=3.10 and use match statements. This is not the most elegant:
        if compute_method == FssComputeMethod.NUMPY:
            return FssNumpy
        elif compute_method == FssComputeMethod.NUMBA_NATIVE:
            raise NotImplementedError(f"compute method not implemented: {compute_method}")
        elif compute_method == FssComputeMethod.NUMBA_PARALLEL:
            raise NotImplementedError(f"compute method not implemented: {compute_method}")
        elif compute_method == FssComputeMethod.RUST_NATIVE:
            raise NotImplementedError(f"compute method not implemented: {compute_method}")
        elif compute_method == FssComputeMethod.RUST_OCL:
            raise NotImplementedError(f"compute method not implemented: {compute_method}")
        else:
            raise ValueError(f"Invalid FSS compute backend, valid values: {list(FssComputeMethod)}")

    def _check_dims(self):
        assert self.fcst.shape == self.obs.shape, "fcst and obs shapes do not match"
        assert self.window[0] < self.fcst.shape[0], "window must be smaller than data shape"
        assert self.window[1] < self.fcst.shape[1], "window must be smaller than data shape"

    @abstractmethod
    def _apply_threshold(self):
        raise NotImplementedError("_compute_integral_field not implemented")

    @abstractmethod
    def _compute_integral_field(self):
        raise NotImplementedError("_compute_integral_field not implemented")

    @abstractmethod
    def _compute_fss_score(self) -> np.float64:
        raise NotImplementedError("_compute_integral_field not implemented")

    def compute_fss(self) -> np.float64:
        fss_result = self._apply_threshold()._compute_integral_field()._compute_fss_score()
        return fss_result


@dataclass
class FssNumpy(FssBackend):
    compute_method = FssComputeMethod.NUMPY

    def _apply_threshold(self):
        self._obs_pop = self.obs > self.threshold
        self._fcst_pop = self.fcst > self.threshold
        return self

    def _compute_integral_field(self):
        # NOTE: no padding introduced in this implementation. Generally
        # 0-padding is not equivilent to trimming and also includes statistics
        # with partial windows, which may not be accurate - but will be equally
        # weighted. I believe that padding is needed in FFT computations by
        # nature of the algorithms and to avoid spectral issues. However, this
        # is not strictly necessary for sum area table methods.
        obs_partial_sums = self._obs_pop.cumsum(1).cumsum(0)
        fcst_partial_sums = self._fcst_pop.cumsum(1).cumsum(0)
        im_h = self.fcst.shape[0] - self.window[0]
        im_w = self.fcst.shape[1] - self.window[1]

        # -----------------------------
        #     A          B
        # tl.0,tl.1    tl.0,br.1
        #     +----------+
        #     |          |
        #     |          |
        #     |          |
        #     +----------+
        # br.0,tl.1   br.0,br.1
        #     C          D
        #
        # area = D - B - C + A
        # ------------------------------
        mesh_tl = np.mgrid[0:im_h, 0:im_w]
        mesh_br = (mesh_tl[0] + self.window[0], mesh_tl[1] + self.window[1])

        obs_a = obs_partial_sums[mesh_tl[0], mesh_tl[1]]
        obs_b = obs_partial_sums[mesh_tl[0], mesh_br[1]]
        obs_c = obs_partial_sums[mesh_br[0], mesh_tl[1]]
        obs_d = obs_partial_sums[mesh_br[0], mesh_br[1]]
        self._obs_img = obs_d - obs_b - obs_c + obs_a

        fcst_a = fcst_partial_sums[mesh_tl[0], mesh_tl[1]]
        fcst_b = fcst_partial_sums[mesh_tl[0], mesh_br[1]]
        fcst_c = fcst_partial_sums[mesh_br[0], mesh_tl[1]]
        fcst_d = fcst_partial_sums[mesh_br[0], mesh_br[1]]
        self._fcst_img = fcst_d - fcst_b - fcst_c + fcst_a

        return self

    def _compute_fss_score(self) -> np.float64:
        num = np.nanmean(np.power(self._obs_img - self._fcst_img, 2))
        denom = np.nanmean(np.power(self._fcst_img, 2) + np.power(self._obs_img, 2))
        fss = 0.0
        if denom > 0:
            fss = 1.0 - float(num) / float(denom)
            # TODO: assert negative or > 1.0 values with error tolerance to avoid masking large errors.
            fss = min(max(fss, 0.0), 1.0)
        return fss
