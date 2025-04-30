"""
Functions for performing block bootstrapping of arrays. This is inspired from
https://github.com/dougiesquire/xbootstrap with modifications to make the functions more
testable and also consistent with the scores package.
"""

import math
import os
from collections import OrderedDict
from itertools import chain, cycle, islice
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from scores.typing import XarrayLike
from scores.utils import tmp_coord_name

# When Dask is being used, this constant helps control the sizes of batches
# when bootstrapping
MAX_BATCH_SIZE_MB = 200


def _get_blocked_random_indices(
    shape: list[int], block_axis: int, block_size: int, prev_block_sizes: list[int], circular: bool = True
) -> np.ndarray:
    """
    Return indices to randomly sample an axis of an array in consecutive
    (cyclic) blocks.

    Args:
        shape: The shape of the array to sample
        block_axis: The axis along which to sample blocks
        block_size: The size of each block to sample
        prev_block_sizes: Sizes of previous blocks along other axes
        circular: whether to sample block circularly.

    Returns:
        An array of indices to use for block resampling.
    """

    def _random_blocks(length, block, circular):
        """
        Indices to randomly sample blocks in a along an axis of a specified
        length
        """
        if block == length:
            return list(range(length))
        repeats = math.ceil(length / block)
        if circular:
            indices = list(
                chain.from_iterable(
                    islice(cycle(range(length)), s, s + block) for s in np.random.randint(0, length, repeats)
                )
            )
        else:
            indices = list(
                chain.from_iterable(
                    islice(range(length), s, s + block) for s in np.random.randint(0, length - block + 1, repeats)
                )
            )
        return indices[:length]

    # Don't randomise within an outer block
    if len(prev_block_sizes) > 0:
        orig_shape = shape.copy()
        for i, b in enumerate(prev_block_sizes[::-1]):
            prev_ax = block_axis - (i + 1)
            shape[prev_ax] = math.ceil(shape[prev_ax] / b)

    if block_size == 1:
        indices = np.random.randint(
            0,
            shape[block_axis],
            shape,
        )
    else:
        non_block_shapes = [s for i, s in enumerate(shape) if i != block_axis]
        indices = np.moveaxis(
            np.stack(
                [_random_blocks(shape[block_axis], block_size, circular) for _ in range(np.prod(non_block_shapes))],
                axis=-1,
            ).reshape([shape[block_axis]] + non_block_shapes),
            0,
            block_axis,
        )

    if len(prev_block_sizes) > 0:
        for i, b in enumerate(prev_block_sizes[::-1]):
            prev_ax = block_axis - (i + 1)
            indices = np.repeat(indices, b, axis=prev_ax).take(range(orig_shape[prev_ax]), axis=prev_ax)
        return indices
    return indices


def _n_nested_blocked_random_indices(
    sizes: OrderedDict[str, Tuple[int, int]], n_iteration: int, circular: bool = True
) -> OrderedDict[str, np.ndarray]:
    """
    Returns indices to randomly resample blocks of an array (with replacement)
    in a nested manner many times. Here, "nested" resampling means to randomly
    resample the first dimension, then for each randomly sampled element along
    that dimension, randomly resample the second dimension, then for each
    randomly sampled element along that dimension, randomly resample the third
    dimension etc.

    Args:
    sizes: Dictionary with {names: (sizes, blocks)} of the dimensions to resample
    n_iteration: The number of times to repeat the random resampling
    circular: Whether or not to do circular resampling.

    Returns:
        A dictionary of arrays containing indices for nested block resampling.

    """

    shape = [s[0] for s in sizes.values()]
    indices = OrderedDict()
    prev_blocks: List[int] = []
    for ax, (key, (_, block)) in enumerate(sizes.items()):
        indices[key] = _get_blocked_random_indices(shape[: ax + 1] + [n_iteration], ax, block, prev_blocks, circular)
        prev_blocks.append(block)
    return indices


def _expand_n_nested_random_indices(indices: list[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Expand the dimensions of the nested input arrays so that they can be
    broadcast and return a tuple that can be directly indexed

    Args:
    indices:List of numpy arrays of sequentially increasing dimension as output by
        the function ``_n_nested_blocked_random_indices``. The last axis on all
        inputs is assumed to correspond to the iteration axis

    Returns:
        Expanded indices suitable for broadcasting.
    """
    broadcast_ndim = indices[-1].ndim
    broadcast_indices = []
    for i, ind in enumerate(indices):
        expand_axes = list(range(i + 1, broadcast_ndim - 1))
        broadcast_indices.append(np.expand_dims(ind, axis=expand_axes))
    return (..., *tuple(broadcast_indices))


def _bootstrap(*arrays: np.ndarray, indices: List[np.ndarray]) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Bootstrap the array(s) using the provided indices

    Args:
        arrays: list of arrays to bootstrap
        indices: list of arrays containing indices to use for bootstrapping each input array

    Returns:
        Bootstrapped arrays
    """
    bootstrapped = [array[ind] for array, ind in zip(arrays, indices)]
    if len(bootstrapped) == 1:
        return bootstrapped[0]
    return tuple(bootstrapped)


def _block_bootstrap(  # pylint: disable=too-many-locals
    array_list: List[XarrayLike],
    blocks: Dict[str, int],
    n_iteration: int,
    exclude_dims: Union[List[List[str]], None] = None,
    circular: bool = True,
) -> Tuple[xr.DataArray, ...]:
    """
    Repeatedly performs bootstrapping on provided arrays across specified dimensions, stacking
    the new arrays along a new "iteration" dimension. Bootstrapping is executed in a nested
    manner: the first provided dimension is bootstrapped, then for each bootstrapped sample
    along that dimension, the second provided dimension is bootstrapped, and so forth.

    Args:
        array_list: Data to bootstrap. Multiple arrays can be passed to be bootstrapped
            in the same way. All input arrays must have nested dimensions.
        blocks: Dictionary of dimension(s) to bootstrap and the block sizes to use
            along each dimension: ``{dim: blocksize}``. Nesting is based on the order of
            this dictionary.
        n_iteration: The number of iterations to repeat the bootstrapping process. Determines
            how many bootstrapped arrays will be generated and stacked along the iteration
            dimension.
        exclude_dims: An optional parameter indicating the dimensions to be excluded during
            bootstrapping for each arrays provided in ``arrays``. This parameter expects a list
            of lists, where each inner list corresponds to the dimensions to be excluded for
            the respective arrays. By default, the assumption is that no dimensions are
            excluded, and all arrays are bootstrapped across all specified dimensions in ``blocks``.
        circular: A boolean flag indicating whether circular block bootstrapping should be
            performed. Circular bootstrapping means that bootstrapping continues from the beginning
            when the end of the data is reached. By default, this parameter is set to True.

     Returns:
        Tuple of bootstrapped xarray DataArrays or Datasets, based on the input.

    Note:
        This function expands out the iteration dimension inside a universal function.
        However, this may generate very large chunks (multiplying chunk size by the number
        of iterations), causing issues for larger iterations. It's advisable to apply this
        function in blocks using 'block_bootstrap'.

    References:
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    # Rename exclude_dims so they are not bootstrapped
    if exclude_dims is None:
        exclude_dims = [[] for _ in range(len(array_list))]
    if not isinstance(exclude_dims, list) or not all(isinstance(x, list) for x in exclude_dims):
        raise ValueError("exclude_dims should be a list of lists")
    if len(exclude_dims) != len(array_list):
        raise ValueError(
            "exclude_dims should be a list of the same length as the number of arrays in array_list",
        )
    renames = []
    for i, (obj, exclude) in enumerate(zip(array_list, exclude_dims)):
        new_dim_list = tmp_coord_name(obj, count=len(exclude))
        if isinstance(new_dim_list, str):
            new_dim_list = [new_dim_list]
        rename_dict = {d: f"{new_dim_list[ii]}" for ii, d in enumerate(exclude)}
        array_list[i] = obj.rename(rename_dict)
        renames.append({v: k for k, v in rename_dict.items()})

    dim = list(blocks.keys())

    # Ensure bootstrapped dimensions have consistent sizes across arrays_list
    for d in blocks.keys():
        dim_sizes = [o.sizes[d] for o in array_list if d in o.dims]
        if not all(s == dim_sizes[0] for s in dim_sizes):
            raise ValueError(f"Block dimension {d} is not the same size on all input arrays")

    # Get the sizes of the bootstrap dimensions
    sizes = None
    for obj in array_list:
        try:
            sizes = OrderedDict({d: (obj.sizes[d], b) for d, b in blocks.items()})
            break
        except KeyError:
            pass
    if sizes is None:
        raise ValueError(
            "At least one input array must contain all dimensions in blocks.keys()",
        )

    # Generate random indices for bootstrapping all arrays_list
    nested_indices = _n_nested_blocked_random_indices(sizes, n_iteration, circular)

    # Expand indices for broadcasting for each array separately
    indices = []
    input_core_dims = []
    for obj in array_list:
        available_dims = [d for d in dim if d in obj.dims]
        indices_to_expand = [nested_indices[key] for key in available_dims]

        indices.append(_expand_n_nested_random_indices(indices_to_expand))
        input_core_dims.append(available_dims)

    # Process arrays_list separately to handle non-matching dimensions
    result = []
    for obj, ind, core_dims in zip(array_list, indices, input_core_dims):
        if isinstance(obj, xr.Dataset):
            # Assume all variables have the same dtype
            output_dtype = obj[list(obj.data_vars)[0]].dtype
        else:
            output_dtype = obj.dtype

        result.append(
            xr.apply_ufunc(
                _bootstrap,
                obj,
                kwargs={"indices": [ind]},
                input_core_dims=[core_dims],
                output_core_dims=[core_dims + ["iteration"]],
                dask="parallelized",
                dask_gufunc_kwargs={"output_sizes": {"iteration": n_iteration}},
                output_dtypes=[output_dtype],
            )
        )

    # Rename excluded dimensions
    return tuple(res.rename(rename) for res, rename in zip(result, renames))


def block_bootstrap(
    array_list: List[XarrayLike] | XarrayLike,
    *,  # Enforce keyword-only arguments
    blocks: Dict[str, int],
    n_iteration: int,
    exclude_dims: Union[List[List[str]], None] = None,
    circular: bool = True,
) -> Union[XarrayLike, Tuple[XarrayLike, ...]]:
    """
    Perform block bootstrapping on provided arrays. The function creates new arrays by repeatedly
    bootstrapping along specified dimensions and stacking the new arrays along a new "iteration"
    dimension. Additionally, it includes internal functions for chunk size calculation and
    handling Dask arrays for chunk size limitation.

    Args:
        array_list: The data to bootstrap, which can be a single xarray object or
            a list of multiple xarray objects. In the case where
            multiple datasets are passed, each dataset can have its own set of dimension. However,
            for successful bootstrapping, dimensions across all input arrays must be nested.
            For instance, for ``block.keys=['d1', 'd2', 'd3']``, an array with dimension 'd1' and
            'd2' is valid, but an array with only dimension 'd2' is not valid. All datasets
            are bootstrapped according to the same random samples along available dimensions.
        blocks: A dictionary specifying the dimension(s) to bootstrap and the block sizes to
            use along each dimension: ``{dimension: block_size}``. The keys represent the dimensions
            to be bootstrapped, and the values indicate the block sizes along each dimension.
            The dimension provided here should exist in the data provided in ``array_list``.
        n_iteration: The number of iterations to repeat the bootstrapping process. Determines
            how many bootstrapped arrays will be generated and stacked along the iteration
            dimension.
        exclude_dims: An optional parameter indicating the dimensions to be excluded during
            bootstrapping for each array provided in ``array_list``. This parameter expects a list
            of lists, where each inner list corresponds to the dimensions to be excluded for
            the respective array. By default, the assumption is that no dimensions are
            excluded, and all arrays are bootstrapped across all specified dimensions in ``blocks``.
        circular: A boolean flag indicating whether circular block bootstrapping should be
            performed. Circular bootstrapping means that bootstrapping continues from the beginning
            when the end of the data is reached. By default, this parameter is set to True.

    Returns:
        If a single Dataset/DataArray (XarrayLike) is provided, the functions returns a
        bootstrapped XarrayLike object along the "iteration" dimension. If multiple XarrayLike
        objects are provided, it returns a tuple of bootstrapped XarrayLike objects, each stacked
        along the "iteration" dimension.

    Raises:
        ValueError: If bootstrapped dimensions don't consistent sizes across ``arrays_list``.
        ValueError: If there is not at least one input array that contains all dimensions in blocks.keys().
        ValueError: If ``exclude_dims`` is not a list of lists.
        ValueError: If the list ``exclude_dims`` is not the same length as the number of
            as ``array_list``.

    References:
        - Gilleland, E. (2020). Bootstrap Methods for Statistical Inference. Part I:
          Comparative Forecast Verification for Continuous Variables. Journal of
          Atmospheric and Oceanic Technology, 37(11), 2117–2134. https://doi.org/10.1175/jtech-d-20-0069.1
        - Wilks, D. S. (2011). Statistical methods in the atmospheric sciences. Academic press.
          https://doi.org/10.1016/C2017-0-03921-6

    Examples:
        Bootstrap a fcst and obs dataset along the time and space dimensions with block sizes of 10
        for each dimension. The bootstrapping is repeated 1000 times.

        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.processing import block_bootstrap
        >>> obs = xr.DataArray(np.random.rand(100, 100), dims=["time", "space"])
        >>> fcst = xr.DataArray(np.random.rand(100, 100), dims=["time", "space"])
        >>> bootstrapped_obs, bootstrapped_fcst = block_bootstrap(
        ...     [obs, fcst],
        ...     blocks={"time": 10, "space": 10},
        ...     n_iteration=1000,
        ... )
    """

    # While the most efficient method involves expanding the iteration dimension withing the
    # universal function, this approach might generate excessively large chunks (resulting
    # from multiplying chunk size by iterations) leading to issues with large numbers of
    # iterations. Hence, here function loops over blocks of iterations to generate the total
    # number of iterations.
    def _max_chunk_size_mb(ds):
        """
        Get the max chunk size in a dataset
        """
        ds = ds if isinstance(ds, xr.Dataset) else ds.to_dataset(name="ds")

        chunks = []
        for var in ds.data_vars:
            da = ds[var]
            chunk = da.chunks
            itemsize = da.data.itemsize
            size_of_chunk = itemsize * np.prod([np.max(x) for x in chunk]) / (1024**2)
            chunks.append(size_of_chunk)
        return max(chunks)

    if not isinstance(array_list, List):
        array_list = [array_list]
    # Choose iteration blocks to limit chunk size on dask arrays
    if array_list[0].chunks:  # Note: This is a way to check if the array is backed by a dask.array
        # without loading data into memory.
        # See https://docs.xarray.dev/en/stable/generated/xarray.DataArray.chunks.html
        ds_max_chunk_size_mb = max(_max_chunk_size_mb(obj) for obj in array_list)
        blocksize = int(MAX_BATCH_SIZE_MB / ds_max_chunk_size_mb)
        blocksize = min(blocksize, n_iteration)
        blocksize = max(blocksize, 1)
    else:
        blocksize = n_iteration

    bootstraps = []
    for _ in range(blocksize, n_iteration + 1, blocksize):
        bootstraps.append(
            _block_bootstrap(
                array_list,
                blocks=blocks,
                n_iteration=blocksize,
                exclude_dims=exclude_dims,
                circular=circular,
            )
        )
    leftover = n_iteration % blocksize

    if leftover:
        bootstraps.append(
            _block_bootstrap(
                array_list,
                blocks=blocks,
                n_iteration=leftover,
                exclude_dims=exclude_dims,
                circular=circular,
            )
        )

    bootstraps_concat = tuple(
        xr.concat(
            bootstrap,
            dim="iteration",
            coords="minimal",
            compat="override",
        )
        for bootstrap in zip(*bootstraps)
    )

    if len(array_list) == 1:
        return bootstraps_concat[0]
    return bootstraps_concat
