"""
Sample implementation of block bootstrapping.

.. note::

    currently only does a single bootstrap iteration on a single numpy array.

TODO:
    - support for xarray
    - support for multiple array inputs of with different subset of axes
    - expand dims with multiple bootstrap iterations
    - support for dask & chunking
    - match api call with ``xbootstrap`` for ease of transition for existing code that relies on it
"""

import functools

import numpy as np
import numpy.typing as npt

from scores.emerging.block_bootstrap.axis_info import AxisInfo, make_axis_info
from scores.emerging.block_bootstrap.methods import FitBlocksMethod


# TODO: A lot of the same variables are common to many of the functions, causing a lot of
# repetition there should probably be two classes (namespaces) to hold state:
# 1. atomic computation: using numpy/numba to read/write the actual block sample
# 2. public interface: for handling xaray inputs and wider options as well as parallelization
#    options.

def block_bootstrap(
    arrs: list[xr.DataArray],
    bootstrap_dims: list[str],
    block_sizes: list[int],
    iterations: int,
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.PARTIAL,
    cyclic: bool = True,
    auto_order_missing: bool = True,
) -> list[xr.DataArray]:

    # validate inputs
    if iterations <= 0:
        ValueError("`iterations` must be greater than 0.")

    if len(block_sizes) == len(bootstrap_dims):
        ValueError("`block_sizes` must be the same size as `bootstrap_dims`.")


    raise NotImplementedError("`block_bootstrap` is currently a stub.")

    # reorder dimensions to align across all arrays
    arrs_reordered = reorder_all_arr_dims(arrs, bootstrap_dims, auto_order_missing)

    # TODO:
    # - collect all array dims
    # - make axis info for each dimension
    # - add dimension name for axis info to make it uniquely identifiable
    # - create axis block indices for dimension `[ (axis_info, axis_block_sample_indices) ]`
    # - if dimension does not exist in bootstrap_dims, set bootstrap = False when creating index,
    #   so that parts of the algorithm can use it.

    arrs_bootstrapped = []

    for arr in arrs:
        # TODO:
        # - get axis subset for array
        # - perform ufunc on `_construct_block_bootstrap_array` with core dims as bootstrap_dims
        # - expand to `iterations`
        # - append to result list.
        pass

    return arrs_bootstrapped


def _construct_block_bootstrap_array(
    input_arr: npt.NDArray,
    block_sizes: list[int],
    cyclic: bool = True,
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.PARTIAL,
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
    ax_block_indices = _sample_axis_block_indices(ax_info, cyclic)

    # re-fetch block-sizes (since internal functions may update this, e.g. based on dask chunks)
    ax_blk_sizes = [axi.block_size for axi in ax_info]
    ax_num_blks = [axi.num_blocks for axi in ax_info]
    ax_len_out = [axi.length_out for axi in ax_info]

    # construct output array, based on axis info
    output_arr = np.empty(ax_len_out)

    # dummy array for block index looping
    dummy_block_idx_arr = np.empty(ax_num_blks)
    num_blocks_iter = np.nditer(dummy_block_idx_arr, flags=["multi_index"], op_flags=["readonly"])

    with num_blocks_iter:
        for _ in num_blocks_iter:
            # increment multi-index by block size intervals
            output_idx = tuple(
                slice(i * b, min((i + 1) * b, axi.length_out))
                for i, b, axi in zip(num_blocks_iter.multi_index, ax_blk_sizes, ax_info)
            )
            # get input block sample from `ax_block_indices`
            block_sample = _sample_block_values(
                input_arr,
                ax_info,
                num_blocks_iter.multi_index,
                ax_block_indices,
                fit_blocks_method,
            )
            # write block sample to output array
            output_arr[output_idx] = block_sample

    return (output_arr, ax_block_indices)


def _sample_block_indices(
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

    def _linear_expand(idx_, block_size_):
        return np.arange(start=idx_, stop=idx_ + block_size_)

    for axi in ax_info:
        (l, b, n) = (axi.length_in, axi.block_size, axi.num_blocks)
        cyc_fn = functools.partial(_cyclic_expand, block_size_=b, length_=l)
        lin_fn = functools.partial(_linear_expand, block_size_=b)

        if axi.bootstrap:
            if cyclic:
                # sample from 0 -> l - 1, since wrap around is allowed
                start_idx = rng.integers(low=0, high=l - 1, size=n)
                block_idx = np.apply_along_axis(lambda x: np.array(list(cyc_fn(x))), 0, start_idx).T
            else:
                # sample from 0 -> l - b, to avoid overflow
                start_idx = rng.integers(low=0, high=l - b, size=n)
                block_idx = np.apply_along_axis(lin_fn, 0, start_idx).T
        else:
            # no bootstrapping for this axis => append entire axis domain
            block_idx = np.array([_linear_expand(0, l)])

        ax_block_indices.append(block_idx)

    return ax_block_indices


def _sample_block_values(
    input_arr: npt.NDArray,
    ax_info: list[AxisInfo],
    ax_idx: list[int],
    ax_block_indices: list[npt.NDArray],
    fit_blocks_method: FitBlocksMethod = FitBlocksMethod.PARTIAL,
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
        fit_blocks_method: method to use to fit block samples,
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
            (n, bp) = (axi.num_blocks, axi.block_size_partial)
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
