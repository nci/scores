"""
Backend for computing Fractions Skill Score (FSS) using numpy
"""

# pylint: disable=R0801

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from scores.fast.fss.backend import FssBackend
from scores.fast.fss.typing import FssComputeMethod, FssDecomposed, f8x4

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
        obs_partial_sums = _summed_area_table(self._obs_pop)
        fcst_partial_sums = _summed_area_table(self._fcst_pop)
        missing_partial_sums = _summed_area_table(~self._valid)

        (w_h, w_w) = (self.window_size[0], self.window_size[1])
        half_w_h = int(w_h / 2)
        half_w_w = int(w_w / 2)

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
            im_h = fcst_partial_sums.shape[0]
            im_w = fcst_partial_sums.shape[1]

            # get remaining window size, incase window size is odd
            rem_h = w_h - half_w_h
            rem_w = w_w - half_w_w

            # Zero padding is equivilent to clamping the coordinate values at the border
            # r = rows, c = cols, tl = top-left, br = bottom-right
            r_tl = np.clip(np.arange(-half_w_h, im_h - half_w_h), 0, im_h - 1)
            c_tl = np.clip(np.arange(-half_w_w, im_w - half_w_w), 0, im_w - 1)
            mesh_tl = np.meshgrid(r_tl, c_tl, indexing="ij")

            r_br = np.clip(np.arange(rem_h, im_h + rem_h), 1, im_h - 1)
            c_br = np.clip(np.arange(rem_w, im_w + rem_w), 1, im_w - 1)
            mesh_br = np.meshgrid(r_br, c_br, indexing="ij")

            # The window at output index (i, j) is anchored on grid point (i, j). Note: there is
            # one more row/column of windows than grid points; the anchors of the last row/column
            # lie in the zero padding, which is never missing.
            anchor_valid = np.pad(
                self._valid,
                ((0, im_h - self._valid.shape[0]), (0, im_w - self._valid.shape[1])),
                constant_values=True,
            )

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
            im_h = fcst_partial_sums.shape[0] - w_h
            im_w = fcst_partial_sums.shape[1] - w_w
            mesh_tl = np.mgrid[0:im_h, 0:im_w]
            mesh_br = (mesh_tl[0] + w_h, mesh_tl[1] + w_w)

            # The window at output index (i, j) covers grid points (i, j) to
            # (i + w_h - 1, j + w_w - 1), and is anchored on its centre point.
            anchor_valid = self._valid[half_w_h : half_w_h + im_h, half_w_w : half_w_w + im_w]

        obs_sums = _window_sums(obs_partial_sums, mesh_tl, mesh_br)
        fcst_sums = _window_sums(fcst_partial_sums, mesh_tl, mesh_br)
        missing_counts = _window_sums(missing_partial_sums, mesh_tl, mesh_br)

        # Fractions are computed over the non-missing points in each window. Points in the zero
        # padding (if any) count as non-missing non-events.
        # Note: a valid anchor lies inside its own window, so `valid_counts >= 1` wherever the
        # fraction is computed.
        valid_counts = w_h * w_w - missing_counts
        self._obs_img = np.divide(obs_sums, valid_counts, out=np.full(obs_sums.shape, np.nan), where=anchor_valid)
        self._fcst_img = np.divide(fcst_sums, valid_counts, out=np.full(fcst_sums.shape, np.nan), where=anchor_valid)

        return self

    def _compute_fss_components(self) -> FssDecomposed:
        # windows anchored on a missing point are NaN in both fraction fields
        valid_windows = ~np.isnan(self._obs_img)
        window_count = np.count_nonzero(valid_windows)

        if window_count == 0:
            return np.void((np.nan, np.nan, np.nan, 0.0), dtype=f8x4)

        obs_img = self._obs_img[valid_windows]
        fcst_img = self._fcst_img[valid_windows]
        diff = np.mean(np.power(obs_img - fcst_img, 2))
        fcst = np.mean(np.power(fcst_img, 2))
        obs = np.mean(np.power(obs_img, 2))
        return np.void((fcst, obs, diff, window_count), dtype=f8x4)


def _summed_area_table(field: npt.NDArray) -> npt.NDArray[np.float64]:
    """
    Computes the summed area table (2-D prefix sum) of ``field``, with a leading row and column of
    0s. The 0 padding addresses off-by-one window sizes for the first row/column due to the way
    meshgrid indexing works.
    """
    table = np.zeros((field.shape[0] + 1, field.shape[1] + 1))
    table[1:, 1:] = field.cumsum(1).cumsum(0)
    return table


def _window_sums(
    table: npt.NDArray[np.float64],
    mesh_tl: Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    mesh_br: Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """
    Computes the sum over each window from a summed area table, given the top-left and
    bottom-right corner coordinates of each window.

    ----------------------------------
    Computing window area from sum
    area table
    ----------------------------------
        A          B
    tl.0,tl.1    tl.0,br.1
        +----------+
        |          |
        |          |
        |          |
        +----------+
    br.0,tl.1   br.0,br.1
        C          D

    area = D - B - C + A
    -----------------------------------
    """
    a = table[mesh_tl[0], mesh_tl[1]]
    b = table[mesh_tl[0], mesh_br[1]]
    c = table[mesh_br[0], mesh_tl[1]]
    d = table[mesh_br[0], mesh_br[1]]
    return d - b - c + a
