"""
Utilty module to provide methods to represent axis information
"""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from scores.emerging.block_bootstrap.methods import FitBlocksMethod


@dataclass
class AxisInfo:
    """
    Structure to hold axis information
    """

    axis_length_in: int
    axis_block_size: int
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.PARTIAL

    # derived members:
    axis_length_out: int = field(init=False)
    axis_num_blocks: int = field(init=False)
    axis_block_size_partial: int = field(init=False)

    def __post_init__(self):
        """
        Adjust the axis length, block size and number of blocks based
        on the method used to fit blocks in the axis.
        """

        (l, b) = (self.axis_length_in, self.axis_block_size)

        assert b <= l

        n = l // b  # multiple
        r = l - (n * b)  # remainder

        if (n, r) == (0, 0):
            raise ValueError("Empty block")
        if r == 0:
            self.axis_num_blocks = n
            self.axis_length_out = l
            self.axis_block_size_partial = 0
        else:
            # key=method, value=(axis_length_out, axis_num_blocks, axis_block_size_partial)
            fit_blocks_params = {
                FitBlocksMethod.PARTIAL: (l, n + 1, r),
                FitBlocksMethod.SHRINK_TO_FIT: (n * b, n, 0),
                FitBlocksMethod.EXPAND_TO_FIT: ((n + 1) * b, n + 1, 0),
            }
            try:
                (
                    self.axis_length_out,
                    self.axis_num_blocks,
                    self.axis_block_size_partial,
                ) = fit_blocks_params[self.axis_fit_blocks_method]
            except KeyError as e:
                raise NotImplementedError(f"Unsupported method: {self.axis_fit_blocks_method}") from e

        self._validate()

    def _validate(self):
        """
        TODO: Add more validation checks here
        """
        assert self.axis_length_out > 0 and self.axis_block_size < self.axis_length_out


def make_axis_info(
    arr: npt.NDArray,
    block_sizes: list[int],
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.PARTIAL,
) -> list[AxisInfo]:
    """
    Returns list of AxisInfo (outer-most axis -> inner-most axis), given a numpy
    ndarray as input
    """
    assert len(arr.shape) == len(block_sizes)

    return [
        AxisInfo(axis_length_in=l, axis_block_size=b, axis_fit_blocks_method=axis_fit_blocks_method)
        for l, b in zip(np.shape(arr), block_sizes)
    ]
