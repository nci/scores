"""
Backend for computing Fractions Skill Score (FSS) using numpy
"""

from dataclasses import dataclass

import numpy as np

from scores.fast.fss.backend import FssBackend
from scores.fast.fss.typing import FssComputeMethod, FssDecomposed, f8x3

_COMPATIBLE = True


@dataclass
class FssNumpy(FssBackend):
    """
    Implementation of numpy backend for computing FSS
    """

    compute_method = FssComputeMethod.NUMPY

    def _check_compatibility(self):  # pragma: no cover
        """
        Numpy should be compatible by default.
        """
        # false branch - should always return
        # kept here to preserve the pattern consistency with other backends
        if _COMPATIBLE:
            return
        # unreachable
        super()._check_compatibility()

    def _compute_integral_field(self):  # pylint: disable=too-many-locals
        obs_partial_sums = self._obs_pop.cumsum(1).cumsum(0)
        fcst_partial_sums = self._fcst_pop.cumsum(1).cumsum(0)

        if self.zero_padding:
            # ------------------------------
            # 0-padding
            # ------------------------------
            #    ..................
            #    ..................
            #    ..+------------+..
            #    ..|            |..
            #    ..|            |..
            #    ..|            |..
            #    ..+------------+..
            #    ..................
            #    ..................
            #
            # ".." represents 0 padding
            # the rectangle represents the
            # FSS computation region.
            # ------------------------------
            # use ceil to avoid 1x1 windows to be coerced to 0x0
            # w_h = window height, w_w = window width
            half_w_h = int(np.ceil(self.window_size[0] / 2))
            half_w_w = int(np.ceil(self.window_size[1] / 2))
            im_h = self.fcst.shape[0]
            im_w = self.fcst.shape[1]

            # Zero padding is equivilent to clamping the coordinate values at the border
            # r = rows, c = cols, tl = top-left, br = bottom-right
            r_tl = np.clip(np.arange(-half_w_h, im_h - half_w_h), 0, im_h - half_w_h - 1)
            c_tl = np.clip(np.arange(-half_w_w, im_w - half_w_w), 0, im_w - half_w_w - 1)
            mesh_tl = np.meshgrid(r_tl, c_tl, indexing="ij")

            r_br = np.clip(np.arange(half_w_h, im_h + half_w_h), half_w_h, im_h - 1)
            c_br = np.clip(np.arange(half_w_w, im_w + half_w_w), half_w_w, im_w - 1)
            mesh_br = np.meshgrid(r_br, c_br, indexing="ij")
        else:
            # ------------------------------
            # interior border
            # ------------------------------
            # data at the border is trim
            # -ed and used for padding
            #    +---------------+
            #    | +-----------+ |
            #    | |///////////| |
            #    | |///////////| |
            #    | |///////////| |
            #    | +-----------+ |
            #    +---------------+
            #
            # "//" represents the effective
            # FSS computation region
            # ------------------------------
            im_h = self.fcst.shape[0] - self.window_size[0]
            im_w = self.fcst.shape[1] - self.window_size[1]
            mesh_tl = np.mgrid[0:im_h, 0:im_w]
            mesh_br = (mesh_tl[0] + self.window_size[0], mesh_tl[1] + self.window_size[1])

        # ----------------------------------
        # Computing window area from sum
        # area table
        # ----------------------------------
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
        # -----------------------------------

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

    def _compute_fss_components(self) -> FssDecomposed:
        diff = np.nanmean(np.power(self._obs_img - self._fcst_img, 2))
        fcst = np.nanmean(np.power(self._fcst_img, 2))
        obs = np.nanmean(np.power(self._obs_img, 2))
        return np.void((fcst, obs, diff), dtype=f8x3)
