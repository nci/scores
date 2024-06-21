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
import xarray as xr

from scores.emerging.block_bootstrap.axis_info import (
    AxisInfo,
    make_axis_info,
    make_axis_info_collection,
)
from scores.emerging.block_bootstrap.helpers import reorder_all_arr_dims
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
    auto_order_unspecified_dims: bool = True,
) -> list[xr.DataArray]:
    """
    Performs block bootstrapping on a list of input arrays.

    Block bootstrapping is a resampling method used to account for dependency
    relationships between data points prior to performing statistical tests.
    This is done by shuffling a data array randomly (with replacement), in a way
    that keeps dependency relationships intact.

    In order to do so, the user specifies a set of dimensions in the input
    array, ``bootstrap_dims``, and ``block_sizes`` representing the size of
    the dependent data points along any given dimension. A rule of thumb is
    to set ``b = sqrt(n)`` for each dimension, where ``n`` is the dimension
    size and ``b`` is the block size. This is just a guideline, and it is the
    user's responsiblity to derive an appropriate block size for representing
    dependency.

    The bootstrapping process can be repeated for several iterations until all
    the blocks are "well represented".

    For example, if our data array is 2 dimensional with "time" (length=12)
    and "x" (size=100) as dependent dimensions. Then, specifying
    ``block_sizes=[6,10]`` will slice the domain into 20, 6 by 10 rectangles.
    Instead of resampling each point, the algorithm will instead sample and
    reconstruct the array from the 20 sliced blocks (with replacement), until
    the output size conforms (based on ``fit_blocks_method``).

    For data that are naturally cyclic (or "circular" as used in some
    literature) in nature, ``cyclic`` can be set to ``True`` to allow the block
    sampler to wrap around. This is common, for example, in day cycles where the
    hour 24 is also the hour 0. For example, the distance (in the measurable
    sense) from hour 24 to hour 4 is 4. If our block size is 6 hours, then the
    hour 24 and hour 4 can fall in the same block. ``cylic = True`` allows this
    to happen, whereas with ``cyclic = False`` the distance is 20 and they can't
    co-exist in a block.

    By default ``fit_blocks_method`` is set to ``PARTIAL`` which is done so
    that the output array shapes conform to the input array shapes. This is
    done to resolve cases where the specified ``block_sizes`` are not factors of
    the dimension sizes of the input arrays. It may be that partial blocks may
    negatively affect results for a statistical experiment. In which case, there
    are other methods that may be applicable (see: :class:`FitBlocksMethod`).

    .. note::

        A natural implication of using block bootstrapping is that dependency
        constrains of the input datum have to be in the form of a N-dimensional
        (cyclic) hyperrectangle. For complex and non-linear dependency
        relationships, the domain would have to be broken up into smaller
        chunks where a hyperrectangle is a sufficient representation of
        dependence.

    .. note::

        While the sampling process is random per iteration, it is consistent
        across all arrays for any one iteration. i.e. if ``arrs = [a1, a2, a3]``
        then the indices sampled for common dimensions in ``a1``, ``a2`` and
        ``a3`` are the same for a given iteration.

        In other words, the block sampling only happens once per iteration, for
        all dimensions, and is cached and applied across all arrays.

    .. important::

        The ordering of ``bootstrap_dims`` is important, as the input array
        dimensions will be re-arranged to match this order so that all the
        arrays are sampled the same way. The first element in ``bootstrap_dims``
        should be the outermost dimension and the last should be the innermost.

    Args:

        arrs: list of arrays to be bootstrapped.

        bootstrap_dims: specifies which dimensions should be used for
            bootstrapping. As mentioned above, the input arrays will be re-
            arranged from outermost to innermost dimension as per the order in
            this list.

        block_sizes: list of integers representing the size (in each axis) of
            a N-dimensional hyperrectangle where any point in this space has an
            observable or expected dependency to the central point.

        iterations: number of times to perform block bootstrapping.

        fit_blocks_method: how to fit blocks if the dimension size is not
            a multiple of the block size. see: :class:`FitBlocksMethod`

        cyclic: whether or not the block sampler is allowed to wrap around if
            the sampled block index exceeds the dimension size. If ``True``,
            the sampled block can start anywhere in $$[0, l - 1]$$ and will wrap
            around, e.g. if $$i + b >= l$$, where $$l$$ is the dimension size,
            $$i$$ is the first index in the block, and $$b$$ is the block size.
            Otherwise (``False``), the block can start anywhere in $$[0, l - b]$$.

        auto_order_unspecified_dims: whether or not to automatically order
            the dimensions in the input array, if they are not specified in
            ``bootstrap_dims``. Note that setting this to ``False`` will throw
            an exception if a dimension in an array is not explicitly specified
            in ``bootstrap_dims``. Setting it to ``True`` will arrange and
            append the non-bootstrap dimensions alphebetically so that they have
            consistent ordering across the input arrays.

    Returns:

        list of arrays that have been bootstrapped, each with an extra
        ``iteration`` dimension representing the iteration index.

    .. note::

        - currently all dimensions are used for the sampling process.
        - only dimensions specified in ``bootstrap_dims`` be sampled with
          replacement.
        - dimensions that are not bootstrapped will use their entire index
          (equivilent to block size = axis length, and without shuffling).
        - if we end up using ``dask``, this design could be improved to utilize
          the underlying chunking strategy of non-bootstrapped dims.

    """
    # validate inputs
    if iterations <= 0:
        ValueError("`iterations` must be greater than 0.")

    if len(block_sizes) != len(bootstrap_dims):
        ValueError("`block_sizes` must be the same size as `bootstrap_dims`.")

    raise NotImplementedError("`block_bootstrap` is currently a stub.")

    # reorder dimensions to align across all arrays
    (arrs_reordered, all_arr_dims_ord) = reorder_all_arr_dims(
        arrs,
        bootstrap_dims,
        auto_order_unspecified_dims,
    )

    axi_collection = make_axis_info_collection(
        arrs,
        bootstrap_dims=bootstrap_dims,
        block_sizes=block_sizes,
        fit_blocks_method=fit_blocks_method,
    )

    # pre-generate samples for each axis
    ax_block_indices = _sample_block_indices(list(axi_collection.iter()), cyclic)
    ax_block_mapping = {d: ax_block_indices[i] for i, d in enumerate(axi_collection.dims_order)}
    arrs_bootstrapped = []

    for arr in arrs_reordered:
        ax_block_indices_for_arr = [ax_block_mapping[d] for d in arr.dims]
        pass
        # TODO:
        # - perform ufunc on `_construct_block_bootstrap_array` with core dims as bootstrap_dims
        # - expand to `iterations`
        # - append to result list.

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
    ax_block_indices = _sample_block_indices(ax_info, cyclic)

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
