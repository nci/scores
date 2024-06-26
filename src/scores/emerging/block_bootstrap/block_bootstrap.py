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
    if iterations <= 0:
        ValueError("`iterations` must be greater than 0.")

    raise NotImplementedError("`block_bootstrap` is currently a stub.")

    # note: this will re-organize the arrays
    array_info_cln = ArrayInfoCollection.make_from_arrays(
        arrs,
        bootstrap_dims,
        block_sizes,
        fit_blocks_method,
        auto_order_missing,
    )

    block_sampler = BlockSampler(
        array_info_collection=array_info_cln,
        cyclic=cyclic,
    )

    # ---
    # broadcast block sampling onto multiple iterations...
    #
    # TODO: for parallel operations, iter_arr should be chunked appropriately,
    # unless the underlying data-arrays themselves are dask arrays, in which
    # case parallelization happens through broadcast chunks.
    iter_arr = xr.DataArray(range(iterations), dims="iterations")

    def _block_sample_ufunc_wrapper(iter_arr_, *arrs_, block_sampler_):
        return np.array([block_sampler_.sample_blocks_unchecked(list(arrs_)) for _ in iter_arr_])

    # ---

    # bootstrap dimensions should not be broadcast as they will be resampled as a block
    # and will not abide by vectorization/chunking rules.
    bootstrap_dims_per_arr = [arr.bootstrap_dims for arr in array_info_cln.array_info]

    arrs_bootstrapped = xr.apply_ufunc(
        iter_arr,
        *tuple(array_info_cln.arrays_ordered),
        core_input_dims=[[], *bootstrap_dims_per_arr],
        output_core_dims=bootstrap_dims_per_arr,
        kwargs={"block_sampler_": block_sampler},
        dask="parallelized",  # TODO: get from keyword args
    )

    return arrs_bootstrapped
