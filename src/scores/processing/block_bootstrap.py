import math
from collections import OrderedDict
from itertools import chain, cycle, islice

import numpy as np
import xarray as xr


def _get_blocked_random_indices(
    shape, block_axis, block_size, prev_block_sizes, circular
):
    """
    Return indices to randomly sample an axis of an array in consecutive
    (cyclic) blocks
    """

    def _random_blocks(length, block, circular):
        """
        Indices to randomly sample blocks in a along an axis of a specified
        length
        """
        if block == length:
            return list(range(length))
        else:
            repeats = math.ceil(length / block)
            if circular:
                indices = list(
                    chain.from_iterable(
                        islice(cycle(range(length)), s, s + block)
                        for s in np.random.randint(0, length, repeats)
                    )
                )
            else:
                indices = list(
                    chain.from_iterable(
                        islice(range(length), s, s + block)
                        for s in np.random.randint(0, length - block + 1, repeats)
                    )
                )
            return indices[:length]

    # Don't randomize within an outer block
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
                [
                    _random_blocks(shape[block_axis], block_size, circular)
                    for _ in range(np.prod(non_block_shapes))
                ],
                axis=-1,
            ).reshape([shape[block_axis]] + non_block_shapes),
            0,
            block_axis,
        )

    if len(prev_block_sizes) > 0:
        for i, b in enumerate(prev_block_sizes[::-1]):
            prev_ax = block_axis - (i + 1)
            indices = np.repeat(indices, b, axis=prev_ax).take(
                range(orig_shape[prev_ax]), axis=prev_ax
            )
        return indices
    else:
        return indices


def _n_nested_blocked_random_indices(sizes, n_iteration, circular):
    """
    Returns indices to randomly resample blocks of an array (with replacement)
    in a nested manner many times. Here, "nested" resampling means to randomly
    resample the first dimension, then for each randomly sampled element along
    that dimension, randomly resample the second dimension, then for each
    randomly sampled element along that dimension, randomly resample the third
    dimension etc.

    Parameters
    ----------
    sizes : OrderedDict
        Dictionary with {names: (sizes, blocks)} of the dimensions to resample
    n_iteration : int
        The number of times to repeat the random resampling
    circular : bool
        Whether or not to do circular resampling
    """

    shape = [s[0] for s in sizes.values()]
    indices = OrderedDict()
    prev_blocks = []
    for ax, (key, (_, block)) in enumerate(sizes.items()):
        indices[key] = _get_blocked_random_indices(
            shape[: ax + 1] + [n_iteration], ax, block, prev_blocks, circular
        )
        prev_blocks.append(block)
    return indices


def _expand_n_nested_random_indices(indices):
    """
    Expand the dimensions of the nested input arrays so that they can be
    broadcast and return a tuple that can be directly indexed

    Parameters
    ----------
    indices : list of numpy arrays
        List of numpy arrays of sequentially increasing dimension as output by
        the function `_n_nested_blocked_random_indices`. The last axis on all
        inputs is assumed to correspond to the iteration axis
    """
    broadcast_ndim = indices[-1].ndim
    broadcast_indices = []
    for i, ind in enumerate(indices):
        expand_axes = list(range(i + 1, broadcast_ndim - 1))
        broadcast_indices.append(np.expand_dims(ind, axis=expand_axes))
    return (..., *tuple(broadcast_indices))


def _block_bootstrap(*objects, blocks, n_iteration, exclude_dims=None, circular=True):
    """
    Repeatedly circularly bootstrap the provided arrays across the specified
    dimension(s) and stack the new arrays along a new "iteration"
    dimension. The boostrapping is done in a nested manner. I.e. bootstrap
    the first provided dimension, then for each bootstrapped sample along
    that dimenion, bootstrap the second provided dimension, then for each
    bootstrapped sample along that dimenion etc.

    Note, this function expands out the iteration dimension inside a
    universal function. However, this can generate very large chunks (it
    multiplies chunk size by the number of iterations) and it falls over for
    large numbers of iterations for reasons I don't understand. It is thus
    best to apply this function in blocks using `block_bootstrap`

    Parameters
    ----------
    objects : xarray DataArray(s) or Dataset(s)
        The data to bootstrap. Multiple datasets can be passed to be
        bootstrapped in the same way. Where multiple datasets are passed, all
        datasets need not contain all bootstrapped dimensions. However, because
        of the bootstrapping is applied in a nested manner, the dimensions in
        all input objects must also be nested. E.g., for `blocks.keys=['d1',
        'd2','d3']` an object with dimensions 'd1' and 'd2' is valid but an
        object with only dimension 'd2' is not. All datasets are boostrapped
        according to the same random samples along available dimensions.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use
        along each dimension: `{dim: blocksize}`. Nesting is carried out according
        to the order of this dictionary.
    n_iteration : int
        The number of times to repeat the bootstrapping.
    exclude_dims : list of list
        List of the same length as the number of objects giving a list of
        dimensions specifed in `blocks` to exclude from each object. Default is
        to assume that no dimensions are excluded and all `objects` are
        bootstrapped across all (available) dimensions `blocks`.
    circular : boolean, optional
        Whether or not to do circular block bootstrapping

    References
    ----------
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """

    def _bootstrap(*arrays, indices):
        """Bootstrap the array(s) using the provided indices"""
        bootstrapped = [array[ind] for array, ind in zip(arrays, indices)]
        if len(bootstrapped) == 1:
            return bootstrapped[0]
        else:
            return tuple(bootstrapped)

    objects = list(objects)

    # Rename exclude_dims so they are not bootstrapped
    if exclude_dims is None:
        exclude_dims = [[] for _ in range(len(objects))]
    msg = (
        "exclude_dims should be a list of the same length as the number of "
        "objects containing lists of dimensions to exclude for each object"
    )
    assert isinstance(exclude_dims, list), msg
    assert len(exclude_dims) == len(objects), msg
    assert all(isinstance(x, list) for x in exclude_dims), msg
    renames = []
    for i, (obj, exclude) in enumerate(zip(objects, exclude_dims)):
        objects[i] = obj.rename(
            {d: f"dim{ii}" for ii, d in enumerate(exclude)},
        )
        renames.append({f"dim{ii}": d for ii, d in enumerate(exclude)})

    dim = list(blocks.keys())
    if isinstance(dim, str):
        dim = [dim]

    # Check that boostrapped dimensions are the same size on all objects
    for d in blocks.keys():
        dim_sizes = [o.sizes[d] for o in objects if d in o.dims]
        assert all(
            s == dim_sizes[0] for s in dim_sizes
        ), f"Block dimension {d} is not the same size on all input objects"

    # Get the sizes of the bootstrap dimensions
    sizes = None
    for obj in objects:
        try:
            sizes = OrderedDict(
                {d: (obj.sizes[d], b) for d, b in blocks.items()},
            )
            break
        except KeyError:
            pass
    if sizes is None:
        raise ValueError(
            "At least one input object must contain all dimensions in blocks.keys()",
        )

    # Generate the random indices first so that we can be sure that each
    # dask chunk uses the same indices. Note, I tried using random.seed()
    # to achieve this but it was flaky. These are the indices to bootstrap
    # all objects.
    nested_indices = _n_nested_blocked_random_indices(sizes, n_iteration, circular)

    # Need to expand the indices for broadcasting for each object separately
    # as each object may have different dimensions
    indices = []
    input_core_dims = []
    for obj in objects:
        available_dims = [d for d in dim if d in obj.dims]
        indices_to_expand = [nested_indices[key] for key in available_dims]

        # Check that dimensions are nested
        ndims = [i.ndim for i in indices_to_expand]
        # Start at 2 due to iteration dim
        if ndims != list(range(2, len(ndims) + 2)):
            raise ValueError("The dimensions of all inputs must be nested")

        indices.append(_expand_n_nested_random_indices(indices_to_expand))
        input_core_dims.append(available_dims)

    # Loop over objects because they may have non-matching dimensions and
    # we don't want to broadcast them as this will unnecessarily increase
    # chunk size for dask arrays
    result = []
    for obj, ind, core_dims in zip(objects, indices, input_core_dims):
        if isinstance(obj, xr.Dataset):
            # Assume all variables have the same dtype
            output_dtype = obj[list(obj.data_vars)[0]].dtype
        else:
            output_dtype = obj.dtype

        result.append(
            xr.apply_ufunc(
                _bootstrap,
                obj,
                kwargs=dict(
                    indices=[ind],
                ),
                input_core_dims=[core_dims],
                output_core_dims=[core_dims + ["iteration"]],
                dask="parallelized",
                dask_gufunc_kwargs=dict(
                    output_sizes={"iteration": n_iteration},
                ),
                output_dtypes=[output_dtype],
            )
        )

    # Rename excluded dimensions
    return tuple(res.rename(rename) for res, rename in zip(result, renames))


def block_bootstrap(*objects, blocks, n_iteration, exclude_dims=None, circular=True):
    """
    Repeatedly circularly bootstrap the provided arrays across the specified
    dimension(s) and stack the new arrays along a new "iteration"
    dimension. The boostrapping is done in a nested manner. I.e. bootstrap
    the first provided dimension, then for each bootstrapped sample along
    that dimenion, bootstrap the second provided dimension, then for each
    bootstrapped sample along that dimenion etc.

    Parameters
    ----------
    objects : xarray DataArray(s) or Dataset(s)
        The data to bootstrap. Multiple datasets can be passed to be
        bootstrapped in the same way. Where multiple datasets are passed, all
        datasets need not contain all bootstrapped dimensions. However, because
        of the bootstrapping is applied in a nested manner, the dimensions in
        all input objects must also be nested. E.g., for `blocks.keys=['d1',
        'd2','d3']` an object with dimensions 'd1' and 'd2' is valid but an
        object with only dimension 'd2' is not. All datasets are boostrapped
        according to the same random samples along available dimensions.
    blocks : dict
        Dictionary of the dimension(s) to bootstrap and the block sizes to use
        along each dimension: `{dim: blocksize}`. Nesting is carried out according
        to the order of this dictionary.
    n_iteration : int
        The number of times to repeat the bootstrapping.
    exclude_dims : list of list
        List of the same length as the number of objects giving a list of
        dimensions specifed in `blocks` to exclude from each object. Default is
        to assume that no dimensions are excluded and all `objects` are
        bootstrapped across all (available) dimensions `blocks`.
    circular : boolean, optional
        Whether or not to do circular block bootstrapping

    References
    ----------
    Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    # The fastest way to perform the iterations is to expand out the
    # iteration dimension inside the universal function (see
    # _iterative_bootstrap). However, this can generate very large chunks (it
    # multiplies chunk size by the number of iterations) and it falls over
    # for large numbers of iterations for reasons I don't understand. Thus
    # here we loop over blocks of iterations to generate the total number
    # of iterations.

    def _max_chunk_size_MB(ds):
        """
        Get the max chunk size in a dataset
        """

        def size_of_chunk(chunks, itemsize):
            """
            Returns size of chunk in MB given dictionary of chunk sizes
            """
            N = 1
            for value in chunks:
                if not isinstance(value, int):
                    value = max(value)
                N = N * value
            return itemsize * N / 1024**2

        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name="ds")

        chunks = []
        for var in ds.data_vars:
            da = ds[var]
            chunk = da.chunks
            itemsize = da.data.itemsize
            if chunk is None:
                # numpy array
                chunks.append((da.data.size * itemsize) / 1024**2)
            else:
                chunks.append(size_of_chunk(chunk, itemsize))
        return max(chunks)

    # Choose iteration blocks to limit chunk size on dask arrays
    if objects[
        0
    ].chunks:  # TO DO: this is not a very good check that input is dask array
        MAX_CHUNK_SIZE_MB = 200
        ds_max_chunk_size_MB = max(
            [_max_chunk_size_MB(obj) for obj in objects],
        )
        blocksize = int(MAX_CHUNK_SIZE_MB / ds_max_chunk_size_MB)
        if blocksize > n_iteration:
            blocksize = n_iteration
        if blocksize < 1:
            blocksize = 1
    else:
        blocksize = n_iteration

    bootstraps = []
    for _ in range(blocksize, n_iteration + 1, blocksize):
        bootstraps.append(
            _block_bootstrap(
                *objects,
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
                *objects,
                blocks=blocks,
                n_iteration=leftover,
                exclude_dims=exclude_dims,
                circular=circular,
            )
        )

    bootstraps_concat = tuple(
        [
            xr.concat(
                b,
                dim="iteration",
                coords="minimal",
                compat="override",
            )
            for b in zip(*bootstraps)
        ]
    )

    if len(objects) == 1:
        return bootstraps_concat[0]
    else:
        return bootstraps_concat
