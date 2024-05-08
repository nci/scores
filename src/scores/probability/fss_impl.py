"""
This module contains methods related to the FSS score

For an explanation of the FSS, and implementation considerations,
see: [Fast calculation of the Fractions Skill Score][fss_ref]

[fss_ref]:
https://www.researchgate.net/publication/269222763_Fast_calculation_of_the_Fractions_Skill_Score
"""
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import Optional, Tuple, Iterable

from scores import utils

import functools
import numpy as np
import numpy.typing as npt
import xarray as xr

from scores.typing import FlexibleDimensionTypes, XarrayLike

# Note: soft keyword `type` only support from >=3.10
DecomposedFss = np.dtype("f8, f8, f8")

class FssComputeMethod(Enum):
    NUMPY = 1  # (default)
    NUMBA_NATIVE = 2
    NUMBA_PARALLEL = 3
    RUST_NATIVE = 4
    RUST_OCL = 5


def fss_2d(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    *,  # Force keywords arguments to be keyword-only
    threshold: np.float64,
    window: Tuple[int, int],
    spatial_dims: Tuple[str, str],
    reduce_dims: Optional[Iterable[str]] = None,
    preserve_dims: Optional[Iterable[str]] = None,
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
    dask: str = "forbidden",  # see: `xarray.apply_ufunc` for options
) -> XarrayLike:
    """
    Aggregates the Fractions Skill Score (FSS) over 2-D spatial coordinates specified by
    `spatial_dims`.

    For implementation for a single 2-D field see: :py:func:`fss_2d_single_field`

    TODO: docstring is WIP
    """
    def _spatial_dims_exist(_dims):
        s_spatial_dims = set(spatial_dims)
        s_dims = set(_dims)
        return s_spatial_dims.intersection(s_dims) == s_spatial_dims

    assert _spatial_dims_exist(fcst.dims), f"missing spatial dims {spatial_dims} in fcst"
    assert _spatial_dims_exist(obs.dims), f"missing spatial dims {spatial_dims} in obs"

    # wrapper defined for convenience, since it's too long for a lambda.
    def fss_wrapper(da_fcst: xr.DataArray, da_obs: xr.DataArray) -> DecomposedFss:
        fss_backend = FssBackend.get_compute_backend(compute_method)
        fb_obj = fss_backend(da_fcst, da_obs, threshold=threshold, window=window)
        return fb_obj.compute_fss_decomposed()

    da_fss = xr.apply_ufunc(
        fss_wrapper,
        fcst,
        obs,
        input_core_dims=[list(spatial_dims), list(spatial_dims)],
        vectorize=True,
        dask=dask,
    )

    # gather dimensions to keep.
    dims = utils.gather_dimensions2(
        fcst,
        obs,
        weights=None,  # weighted score currently not supported
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    all_dims = set(fcst.dims).union(set(obs.dims))
    dims_reduce = set(dims) - set(spatial_dims)

    if len(dims_reduce) == 0:
        # if dimensions to reduce is empty, return output with fss
        # scores along all dims.
        dims_reduce = None

    # apply ufunc again but this time to compute the fss, reducing
    # any non-spatial dimensions.
    da_fss = xr.apply_ufunc(
        _aggregate_fss_decomposed,
        da_fss,
        input_core_dims=[list(dims_reduce)],
        vectorize=True,
        dask=dask,
    )

    return da_fss

def fss_2d_single_field(
    fcst: npt.NDArray[np.float64],
    obs: npt.NDArray[np.float64],
    *,
    threshold: np.float64,
    window: Tuple[int, int],
    compute_method: FssComputeMethod = FssComputeMethod.NUMPY,
) -> np.float64:
    """
    Calculates the Fractions Skill Score (FSS) for a given forecast and observed 2-D field.

    The caller is responsible for making sure the input fields are in the 2-D spatial domain.

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


def _aggregate_fss_decomposed(fss_d: npt.NDArray[DecomposedFss]) -> np.float64:
    """
    Aggregates the results of individual fss scores from 2d fields
    """
    # can't do ufuncs over custom void types currently...
    l = fss_d.size

    if l <= 0:
        return 0.0

    l = float(l)
    fcst_sum = 0.0
    obs_sum = 0.0
    diff_sum = 0.0

    with np.nditer(fss_d) as it:
        for elem in it:
            (fcst_, obs_, diff_) = elem.item()
            fcst_sum += fcst_ / l
            obs_sum += obs_ / l
            diff_sum += diff_ / l

    fss = 0.0
    denom = obs_sum + fcst_sum

    if denom >= 0.0:
        fss = 1.0 - diff_sum / denom

    fss_clamped = max(min(fss, 1.0), 0.0)

    return fss_clamped


@dataclass
class FssBackend(ABC):
    """
    Abstract base class for computing fss.

    required methods:
        - :py:meth:`_compute_fss_components`
        - :py:meth:`_integral_field`
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

    def _apply_threshold(self):
        """
        Default implementation of converting the input fields into binary fields by comparing
        against a threshold; using numpy. This is a relatively cheap operation, so a specific
        implementation in derived backends is optional.
        """
        self._obs_pop = self.obs > self.threshold
        self._fcst_pop = self.fcst > self.threshold
        return self

    @abstractmethod
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
        raise NotImplementedError("_compute_integral_field not implemented")

    @abstractmethod
    def _compute_fss_components(self) -> np.float64:
        """
        FSS is roughly defined as
        ```
            fss = 1 - sum_w((p_o - p_f)^2) / (sum_w(p_o^2) + sum_w(p_f^2))

            where,
            p_o: observation populace > threshold, in one window
            p_f: forecast populace > threshold, in one window
            sum_w: sum over all windows
        ````

        In order to accumulate scores over non spatial dimensions at a higher level operation, we
        need to keep track of the de-composed sums as they need to be accumulated separately.

        The "fss components", hence, consist of:
        - `power_diff :: float = sum_w((p_o - p_f)^2)`
        - `power_obs :: float = sum_w(p_o^2)`
        - `power_fcst :: float = sum_w(p_f^2)`

        WARNING: returning sums of powers can result in overflows on aggregation. It is advisable
        to return the means instead, as it is equivilent with minimal computational trade-off.

        Returns:
            Tuple[float, float, float]: (power_fcst, power_obs, power_diff) as described above
        """
        raise NotImplementedError("_compute_fss_components not implemented")

    def compute_fss_decomposed(self) -> DecomposedFss:
        return self._apply_threshold()._compute_integral_field()._compute_fss_components()

    def compute_fss(self) -> np.float64:
        """
        Uses the components from :py:method:`compute_fss_decomposed` to compute the final
        Fractions Skill Score.
        """
        (fcst, obs, diff) = self.compute_fss_decomposed()
        denom = fcst + obs
        fss = 0.0

        if denom >= 0.0:
            fss = 1.0 - diff / denom

        fss_clamped = max(min(fss, 1.0), 0.0)

        return fss_clamped



@dataclass
class FssNumpy(FssBackend):
    """
    Implementation of numpy backend for computing FSS
    """
    compute_method = FssComputeMethod.NUMPY

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

    def _compute_fss_components(self) -> np.float64:
        diff = np.nanmean(np.power(self._obs_img - self._fcst_img, 2))
        fcst = np.nanmean(np.power(self._fcst_img, 2))
        obs = np.nanmean(np.power(self._obs_img, 2))
        return np.void((fcst, obs, diff), dtype=DecomposedFss)
