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

from scores.typing import FlexibleDimensionTypes, XarrayLike


class FssComputeMethod(Enum):
    NUMPY = 1  # (default)
    NUMBA_NATIVE = 2
    NUMBA_PARALLEL = 3
    RUST_NATIVE = 4
    RUST_OCL = 5


def fss_2d(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold: np.float64,
    spatial_dims: Tuple[str, str],
    *,  # Force keywords arguments to be keyword-only
    reduce_dims: Optional[Iterable[str]] = None,
    preserve_dims: Optional[Iterable[str]] = None,
):
    def _spatial_dims_exist(_dims):
        s_spatial_dims = set(spatial_dims)
        s_dims = set(_dims)
        return s_spatial_dims.intersection(s_dims) == s_spatial_dims

    assert _spatial_dims_exist(fcst), f"missing spatial dims {spatial_dims} in fcst"
    assert _spatial_dims_exist(obs), f"missing spatial dims {spatial_dims} in obs"

    # apply_ufunc here to reduce spatial dims and calculate fss score for all 2D fields.
    xr.apply_ufunc(...)

    # gather dimensions to keep.
    dims = scores.utils.gather_dimensions2(
        fcst,
        obs,
        weights=None,  # weighted score currently not supported
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    all_dims = set(fcst.dims).union(set(obs.dims))
    dims_reduce = all_dims - set(dims) - set(spatial_dims)

    if len(dims_reduce) == 0:
        # if dimensions to reduce is empty, we're done - return output with fss
        # scores along all dims.
        ...
    else:
        # otherwise, we want to further reduce dimensions. Do a second
        # apply_ufunc, but this time to accumulate (mean) the decomposed fss
        # score along the core axes (i.e. dims_to_reduce)
        xr.apply_ufunc(...)


def fss_2d_single_field(
    fcst: npt.NDArray[np.float64],
    obs: npt.NDArray[np.float64],
    *,
    threshold: np.float64,
    window: Tuple[int, int],
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
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
        # TODO: return decomposed fss score instead
        return fss
