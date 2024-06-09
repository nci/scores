from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt


class FitBlocksMethod(Enum):
    """
    Method to use to fit blocks into axis e.g. if the axis length is not a
    multiple of blocksize.

    Supported:

    - FitBlocksMethod.SHRINK_TO_FIT: shrinks axis to fit whole blocks

    Currently unimplemented:

    - FitBlocksMethod.EXPAND_TO_FIT: expands axis to fit whole blocks
    - FitBlocksMethod.PARTIAL: allows sampling of partial blocks
    """

    SHRINK_TO_FIT = 0
    EXPAND_TO_FIT = 1
    PARTIAL = 2


@dataclass
class AxisInfo:
    """
    Structure to hold axis information
    """

    axis_length_in: int
    axis_block_size: int
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.SHRINK_TO_FIT

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
        elif r == 0:
            self.axis_num_blocks = n
            self.axis_length_out = l
            self.axis_block_size_partial = 0
        else:
            # key=method, value=(axis_length_out, axis_num_blocks, axis_block_size_partial)
            fit_blocks_params = {
                FitBlocksMethod.SHRINK_TO_FIT: (n * b, n, 0),
                FitBlocksMethod.EXPAND_TO_FIT: ((n + 1) * b, n + 1, 0),
                FitBlocksMethod.PARTIAL: (l, n + 1, r),
            }

            try:
                (
                    self.axis_length_out,
                    self.axis_num_blocks,
                    self.axis_block_size_partial,
                ) = fit_blocks_params[self.axis_fit_blocks_method]
            except KeyError:
                raise NotImplementedError(f"Unsupported method: {self.axis_fit_blocks_method}")

        self._validate()

    def _validate(self):
        """
        TODO: Add more validation checks here
        """
        assert self.axis_length_out > 0 and self.axis_block_size < self.axis_length_out


def make_axis_info(
    arr: npt.NDArray,
    block_sizes: list[int],
    axis_fit_blocks_method: FitBlocksMethod = FitBlocksMethod.SHRINK_TO_FIT,
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


def sample_axis_block_indices(
    ax_info: list[AxisInfo],
    cyclic: bool = True,
) -> list[npt.NDArray]:
    """
    Args:
        ax_info: (ordered) list of information for each axis.
        cyclic:
            True  => indices can cycle around if they overflow past the axis
                     length
            False => blocks will not be sampled from indices that overflow past
                     the axis length

    Returns a list of 2-D arrays N by B of random block indices for each axis, where

        - N = number of blocks
        - B = block size
    """
    rng = np.random.default_rng()
    ax_block_indices = []

    # note: this can probably just be a for loop, but separated out for clarity
    def _cyclic_expand(idx_, block_size_, length_):
        i = 0
        while i < block_size_:
            yield (idx_ + i) % length_
            i = i + 1

    for axi in ax_info:
        (l, b, n) = (axi.axis_length_in, axi.axis_block_size, axi.axis_num_blocks)

        if cyclic:
            # sample from 0 -> l - 1, since wrap around is allowed
            start_idx = rng.integers(low=0, high=l - 1, size=n)
            block_idx = np.apply_along_axis(lambda x: np.array(list(_cyclic_expand(x, b, l))), 0, start_idx).T
        else:
            # sample from 0 -> l - b, to avoid overflow
            start_idx = rng.integers(low=0, high=l - b, size=n)
            block_idx = np.apply_along_axis(lambda x: np.arange(start=x, stop=x + b), 0, start_idx).T

        ax_block_indices.append(block_idx)

    return ax_block_indices


def sample_block_values(
    input_arr: npt.NDArray,
    ax_info: list[AxisInfo],
    ax_idx: list[int],
    ax_block_indices: list[npt.NDArray],
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.SHRINK_TO_FIT,
):
    """
    Returns a sampled block of values from the input array, using a mult-index
    pivot point and a reference pre-sampled block indices for each axis.
    Usually used by an outer loop to retrieve a sample from ``ax_block_indices``
    iteratively.

    Args:
        input_arr: input array N-dimensional input array.
        ax_info: (ordered) list of information for each axis.
        ax_idx: 1-D multi-index to determine which block to retrieve the sample
            values from.
        ax_block_indices: pre-sampled list of N * B array of indices,
            where, N = number of blocks for axis, B = block size for axis.
        axis_fit_blocks_method: method to use to fit block samples,
            see: :class:`FitBlocksMethod`

    suppose

    .. code-block::

        ax_idx = [1,2,0,3]
        ax_block_indices = [
            # outermost axis
            [[1,2,3], [4,5,6]],  # ax_i = 0, block_size = 3, num_blocks = 2
            [[1], [9], [5]],     # ax_i = 1, block_size = 1, num_blocks = 3
            [[1,2,3,4,5]],       # ax_i = 2, block_size = 5, num_blocks = 1
            [[0,1], [1,2], [3,4], [4,0]],  # ax_i = 3, block_size = 2, num_blocks = 4
            # inner most axis
        ]

    then we would expect the output block sample to be based on the following indices,

    .. code-block::

        ax_0 = [4,5,6]
        ax_1 = [5]
        ax_2 = [1,2,3,4,5]
        ax_3 = [4,0]

    which will result in the following multi indices from the input array,

    .. code-block::

        [4,5,1,4], [4,5,1,0], [4,5,2,4] ... [6,5,5,4], [6,5,5,0]

    this can be mapped to the following block in the output array:

    .. code-block::

        output_arr[1:4, 2:3, 0:5, 3:5]

    .. note::

        - There may be advanced indexing alternatives that may also work, see:
        https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
        - Likely to be superseeded by `numba`/`dask` implementation.
    """
    # trim final blocks if partial blocks are allowed
    if fit_blocks_method == FitBlocksMethod.PARTIAL:
        bootstrap_ax_idx = []
        for b, i, axi in zip(ax_block_indices, ax_idx, ax_info):
            (n, bp) = (axi.axis_num_blocks, axi.axis_block_size_partial)
            if i == (n - 1) and bp > 0:
                bootstrap_ax_idx.append(b[i][0:bp])
            else:
                bootstrap_ax_idx.append(b[i])
    else:
        # no trimming required in other methods
        bootstrap_ax_idx = [b[i] for b, i in zip(ax_block_indices, ax_idx)]

    # initialize empty block sample
    block_sample = np.empty([len(i) for i in bootstrap_ax_idx])
    block_size_it = np.nditer(block_sample, flags=["multi_index"], op_flags=["writeonly"])

    # retrieve block values from input array and write it to block sample
    with block_size_it:
        for x in block_size_it:
            bootstrap_idx = tuple(b[i] for b, i in zip(bootstrap_ax_idx, block_size_it.multi_index))
            x[...] = input_arr[bootstrap_idx]

    return block_sample


def construct_block_bootstrap_array(
    input_arr: npt.NDArray,
    block_sizes: list[int],
    cyclic: bool = True,
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.SHRINK_TO_FIT,
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
    ax_block_indices = sample_axis_block_indices(ax_info)

    # re-fetch block-sizes (since internal functions may update this, e.g. based on dask chunks)
    ax_blk_sizes = [axi.axis_block_size for axi in ax_info]
    ax_num_blks = [axi.axis_num_blocks for axi in ax_info]
    ax_len_out = [axi.axis_length_out for axi in ax_info]

    # construct output array, based on axis info
    output_arr = np.empty(ax_len_out)

    # dummy array for block index looping
    dummy_block_idx_arr = np.empty(ax_num_blks)
    num_blocks_iter = np.nditer(dummy_block_idx_arr, flags=["multi_index"], op_flags=["readonly"])

    with num_blocks_iter:
        for _ in num_blocks_iter:
            # increment multi-index by block size intervals
            output_idx = tuple(
                slice(i * b, min((i + 1) * b, axi.axis_length_out))
                for i, b, axi in zip(num_blocks_iter.multi_index, ax_blk_sizes, ax_info)
            )
            # get input block sample from `ax_block_indices`
            block_sample = sample_block_values(
                input_arr,
                ax_info,
                num_blocks_iter.multi_index,
                ax_block_indices,
                fit_blocks_method,
            )
            # write block sample to output array
            output_arr[output_idx] = block_sample

    return (output_arr, ax_block_indices)


if __name__ == "__main__":
    import pprint

    # generate test data
    axis_len = [13, 10, 8, 7]
    block_sizes = [4, 3, 2, 3]
    method = FitBlocksMethod.PARTIAL
    rng = np.random.default_rng(seed=42)
    # random dataset with integers so its easy to visualize
    input_arr = rng.integers(low=0, high=10, size=axis_len)

    print("--- axis info ---")
    axis_info = make_axis_info(input_arr, block_sizes, method)
    pprint.pp(axis_info)

    print("\n--- input array ---")
    pprint.pp(input_arr.shape)
    res = np.histogram(input_arr, bins=5)
    pprint.pp(res)

    print("\nBOOTSTRAPPING...")
    (output_arr, block_sample_idx) = construct_block_bootstrap_array(
        input_arr,
        block_sizes,
        cyclic=True,
        fit_blocks_method=method,
    )

    print("\n--- sample axis block indices ---")
    pprint.pp(block_sample_idx)

    print("\n--- output array ---")
    pprint.pp(output_arr.shape)
    # print(output_arr)
    res = np.histogram(output_arr, bins=5)
    pprint.pp(res)
