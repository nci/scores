from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt


class FitBlocksMethod(Enum):
    """
    Method to use to fit blocks into axis e.g. if the axis length is not a
    multiple of blocksize.

    Supported:

    - FitBlocksMethod.TRIM: trims axis to largest multiple of blocksize < axis length

    Currently unimplemented:

    - FitBlocksMethod.PAD: pads axis based on interpolation
    - FitBlocksMethod.PARTIAL_BLOCKS: allows sampling of partial blocks
    """

    TRIM = 0
    PAD = 1
    PARTIAL_BLOCKS = 2


@dataclass
class AxisInfo:
    """
    Structure to hold axis information
    """

    axis_length: int
    axis_block_size: int
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.TRIM
    axis_num_blocks: int = field(init=False)

    def __post_init__(self):
        """
        Adjust the axis length, block size and number of blocks based
        on the method used to fit blocks in the axis.
        """
        (l, b) = (self.axis_length, self.axis_block_size)
        n = l // b  # multiple
        r = l - n  # remainder

        if (n, r) == (0, 0):
            raise ValueError("Empty block")
        elif r == 0:
            self.axis_num_blocks = n
        else:
            if self.axis_fit_blocks_method == FitBlocksMethod.TRIM:
                self.axis_num_blocks = n
                self.axis_length = n * b
            else:
                raise NotImplementedError(f"Unsupported method: {self.axis_fit_blocks_method}")

        self._validate()

    def _validate(self):
        """
        TODO: Aditional validation checks here
        """
        # TODO: WIP
        assert self.axis_length > 0 and self.axis_block_size < self.axis_length


def make_axis_info(
    arr: npt.NDArray,
    block_sizes: list[int],
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.TRIM,
) -> list[AxisInfo]:
    """
    Returns list of AxisInfo (outer-most axis -> inner-most axis), given a numpy ndarray as input
    """
    assert len(arr.shape) == len(block_sizes)

    return [AxisInfo(axis_length=l, axis_block_size=b) for l, b in zip(np.shape(arr), block_sizes)]


def sample_axis_block_indices(ax_info: list[AxisInfo], cyclic: bool = True) -> list[npt.NDArray]:
    """
    cyclic:
         True => indices can cycle around if they overflow past the axis length
         False => blocks will not be sampled from indices that overflow past the axis length

    Returns a list of 2-D arrays N by B of random block indices for each axis,

    where

    N = number of blocks
    B = block size
    """
    rng = np.random.default_rng()
    ax_block_samples = []

    # TODO: this can probably just be a for loop, but separated out for clarity
    def _cyclic_expand(idx, block_size, length_):
        i = 0
        while i < block_size:
            yield (idx + i) % length_
            i = i + 1

    for axi in ax_info:
        (l, b, n) = (axi.axis_length, axi.axis_block_size, axi.axis_num_blocks)

        block_idx = None

        if cyclic:
            # sample from 0 -> l - 1, since wrap around is allowed
            start_idx = rng.integers(low=0, high=l - 1, size=n)
            block_idx = np.apply_along_axis(lambda x: np.array(list(_cyclic_expand(x, b, l))), 0, start_idx)
        else:
            # sample from 0 -> l - b, to avoid overflow
            start_idx = rng.integers(low=0, high=l - 1 - b, size=n)
            block_idx = np.apply_along_axis(lambda x: np.arange(start=x, stop=x + b), 0, start_idx)

        assert not (block_idx is None)
        ax_block_samples.append(block_idx)

    return ax_block_samples


def construct_block_bootstrap_array(
    input_arr: npt.NDArray,
    block_sizes: list[int],
    cyclic: bool = True,
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.TRIM,
) -> npt.NDArray:
    """
    takes a numpy array and performs block resampling for 1 iteration,

    note that the output array size is not guarenteed to be the same shape,
    depending on FitBlocksMethod
    """
    assert len(block_sizes) == len(input_arr.shape)

    # construct axis info based on block sizes and input array
    ax_info = make_axis_info(input_arr, block_sizes, fit_blocks_method)

    # get sample block indices for each axis
    block_sample_idx = sample_axis_block_indices(ax_info)

    # TODO: this is potentially slow there may be better indexing strategies
    # TODO: move function outside
    # NOTE: this will probably be superceded by some sort of dask/numba option
    # see also: https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
    def _sample_block(input_arr_, block_sizes_, ax_idx_, block_bootstrap_idx_):
        """
        Suppose
            ax_idx_ = [1,2,0,3]
            block_bootstrap_idx_ = [
                # outermost axis
                [[1,2,3], [4,5,6]],  # ax_i = 0, block_size = 3, num_blocks = 2
                [[1], [9], [5]],     # ax_i = 1, block_size = 1, num_blocks = 3
                [[1,2,3,4,5]],       # ax_i = 2, block_size = 5, num_blocks = 1
                [[1,1], [1,2], [3,4], [4,0]],  # ax_i = 3, block_size = 2, num_blocks = 4
                # inner most axis
            ]

        then we would expect the output block sample to be based on the following indices:
        ax_0 = [4,5,6]
        ax_1 = [5]
        ax_2 = [1,2,3,4,5]
        ax_3 = [4,0]

        Which will result in the following multi indices from the input array:
        [4,5,1,4], [4,5,1,0], [4,5,2,4] ... [6,5,5,4], [6,5,5,0]

        mapped to the following block in the output array:
        arr_output[1:4, 2:3, 0:5, 3:5]
        """
        _block_sample = np.empty(block_sizes_)
        _block_it = np.nditer(_block_sample, flags=["multi_index"], op_flags=["writeonly"])
        _bootstrap_ax_idx = [b.T[i] for b, i in zip(block_bootstrap_idx_, ax_idx_)]
        with _block_it:
            for x in _block_it:
                _bootstrap_idx = tuple(b[i] for b, i in zip(_bootstrap_ax_idx, _block_it.multi_index))
                x[...] = input_arr_[_bootstrap_idx]
        return _block_sample

    # re-fetch block-sizes (since internal functions may update this, e.g. based on dask chunks)
    ax_blk_sizes = [axi.axis_block_size for axi in ax_info]
    ax_num_blks = [axi.axis_num_blocks for axi in ax_info]
    ax_len = [axi.axis_length for axi in ax_info]

    # construct output array, based on axis info
    output_arr = np.empty(ax_len)

    # dummy array for block index looping
    dummy_block_idx_arr = np.empty(ax_num_blks)

    with np.nditer(dummy_block_idx_arr, flags=["multi_index"], op_flags=["readonly"]) as it:
        for _ in it:
            output_idx = tuple(slice(i * b, (i * b + b)) for i, b in zip(it.multi_index, ax_blk_sizes))
            block_sample = _sample_block(input_arr, ax_blk_sizes, it.multi_index, block_sample_idx)
            output_arr[output_idx] = block_sample

    return (output_arr, block_sample_idx)


if __name__ == "__main__":
    # generate test data
    axis_len = [12, 9, 8, 6]
    block_sizes = [4, 3, 2, 3]
    rng = np.random.default_rng(seed=42)
    # random dataset with integers so its easy to visualize
    input_arr = rng.integers(low=0, high=10, size=axis_len)

    print("--- input array ---")
    print(input_arr.shape)
    res = np.histogram(input_arr, bins=5)
    print(res)

    print("BOOTSTRAPPING...")
    (output_arr, block_sample_idx) = construct_block_bootstrap_array(input_arr, block_sizes)

    print("--- sample axis block indices ---")
    print(block_sample_idx)

    print("--- output array ---")
    print(output_arr.shape)
    res = np.histogram(output_arr, bins=5)
    print(res)
