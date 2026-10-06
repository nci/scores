"""
Functions for performing block bootstrapping of arrays. This is inspired from
https://github.com/dougiesquire/xbootstrap with modifications to make the functions more
testable and also consistent with the scores package.
"""

import math
from collections import OrderedDict
from itertools import chain, cycle, islice
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from scores.typing import XarrayLike
from scores.utils import tmp_coord_name

# When Dask is being used, this constant helps control the sizes of batches
# when bootstrapping
MAX_BATCH_SIZE_MB = 200


def _get_blocked_random_indices(
    shape: list[int],
    block_axis: int,
    block_size: int,
    prev_block_sizes: list[int],
    circular: bool = True,
    *,
    draw_integers: Callable[..., np.ndarray],
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
        draw_integers: Callable with the same low, high and size arguments as
            numpy.random.Generator.integers.

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
                    islice(cycle(range(length)), s, s + block) for s in draw_integers(0, length, repeats)
                )
            )
        else:
            indices = list(
                chain.from_iterable(
                    islice(range(length), s, s + block) for s in draw_integers(0, length - block + 1, repeats)
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
        indices = draw_integers(
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
    sizes: OrderedDict[str, Tuple[int, int]],
    n_iteration: int,
    circular: bool = True,
    *,
    draw_integers: Callable[..., np.ndarray],
    iteration_major: bool = False,
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
    draw_integers: Callable used to draw random integers.
    iteration_major: Draw all dimensions for one iteration before the next,
        making samples independent of the number of iterations in a batch.
        The default preserves the legacy dimension-first sampling order.

    Returns:
        A dictionary of arrays containing indices for nested block resampling.

    """

    if iteration_major:
        indices = OrderedDict()
        for iteration in range(n_iteration):
            sample = _n_nested_blocked_random_indices(sizes, 1, circular, draw_integers=draw_integers)
            for key, ind in sample.items():
                if iteration == 0:
                    indices[key] = np.empty(ind.shape[:-1] + (n_iteration,), dtype=ind.dtype)
                indices[key][..., iteration] = ind[..., 0]
        return indices

    shape = [s[0] for s in sizes.values()]
    indices = OrderedDict()
    prev_blocks: List[int] = []
    for ax, (key, (_, block)) in enumerate(sizes.items()):
        indices[key] = _get_blocked_random_indices(
            shape[: ax + 1] + [n_iteration],
            ax,
            block,
            prev_blocks,
            circular,
            draw_integers=draw_integers,
        )
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
    *,
    draw_integers: Callable[..., np.ndarray],
    iteration_major: bool = False,
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
        draw_integers: A callable function used to draw random integers, typically from a random
            number generator. This allows for custom random number generation strategies to be
            used during the bootstrapping process.
        iteration_major: Generate each iteration's indices in a fixed order, independently
            of batching. False preserves legacy sampling.

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
    # Keep renaming local so subsequent batches receive the original dimensions.
    array_list = array_list.copy()
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
    nested_indices = _n_nested_blocked_random_indices(
        sizes, n_iteration, circular, draw_integers=draw_integers, iteration_major=iteration_major
    )

    # Expand indices for broadcasting for each array separately
    indices = []
    input_core_dims = []
    for obj in array_list:
        available_dims = [d for d in dim if d in obj.dims]
        indices_to_expand = [nested_indices[key] for key in available_dims]

        indices.append(_expand_n_nested_random_indices(indices_to_expand))
        input_core_dims.append(available_dims)

    def _bootstrap_dataarray(obj, ind, core_dims):
        # A bootstrap sample can select anywhere along a core dimension.
        # Join those chunks explicitly, retaining chunks along other dimensions.
        if obj.chunks is not None:
            obj = obj.chunk({d: -1 for d in core_dims})
        return xr.apply_ufunc(
            _bootstrap,
            obj,
            kwargs={"indices": [ind]},
            input_core_dims=[core_dims],
            output_core_dims=[core_dims + ["iteration"]],
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"iteration": n_iteration}},
            output_dtypes=[obj.dtype],
        )

    # Map Dataset variables separately to preserve each variable's dtype.
    result = []
    for obj, ind, core_dims in zip(array_list, indices, input_core_dims):
        if isinstance(obj, xr.Dataset):
            result.append(obj.map(_bootstrap_dataarray, args=(ind, core_dims)))
        else:
            result.append(_bootstrap_dataarray(obj, ind, core_dims))

    # Rename excluded dimensions
    return tuple(res.rename(rename) for res, rename in zip(result, renames))


def block_bootstrap(
    array_list: List[XarrayLike] | XarrayLike,
    *,  # Enforce keyword-only arguments
    blocks: Dict[str, int],
    n_iteration: int,
    exclude_dims: Union[List[List[str]], None] = None,
    circular: bool = True,
    rng: np.random.Generator | int | None = None,
) -> Union[XarrayLike, Tuple[XarrayLike, ...]]:
    """
    Perform block bootstrapping on provided arrays. The function creates new arrays by repeatedly
    bootstrapping along specified dimensions and stacking the new arrays along a new "iteration"
    dimension. Dask inputs remain lazy, with iterations batched to limit output chunk sizes.

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
        rng: Controls the random sampling. If set to `None`, it uses NumPy's legacy global
            random state, preserving compatibility with np.random.seed(...).
            If a non-negative integer is supplied, it creates a numpy.random.Generator
            initialised with that seed. If a Generator, uses that instance
            and advances its state. Passing an integer or Generator does not use or modify NumPy's
            global random state. With an integer or Generator, samples are independent of Dask
            chunking and iteration batch sizes. Reusing a Generator continues its random stream.
            The legacy default retains its original sampling order, so its samples may depend
            on batching. Pass ``np.random.default_rng()`` for independent, unseeded sampling.

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

    Notes:
        Dask chunks along bootstrap dimensions are joined before sampling. Other dimensions
        retain their chunks. The batch-size estimate uses these joined chunks; if a single
        iteration exceeds the target size, it is processed in a batch of one.

    References:
        - Gilleland, E. (2020). Bootstrap Methods for Statistical Inference. Part I:
          Comparative Forecast Verification for Continuous Variables. Journal of
          Atmospheric and Oceanic Technology, 37(11), 2117–2134. https://doi.org/10.1175/jtech-d-20-0069.1
        - Wilks, D. S. (2011). Statistical methods in the atmospheric sciences. Academic press.
          https://doi.org/10.1016/C2017-0-03921-6

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from scores.processing import block_bootstrap

        >>> times = np.arange(6)
        >>> stations = ["S1", "S2", "S3", "S4"]

        >>> # Create synthetic observations
        >>> obs_data = np.array(
        ...     [[t + (s / 10) for s in range(4)] for t in range(6)]
        ... )
        >>> obs = xr.DataArray(
        ...     obs_data,
        ...     coords={"time": times, "station": stations},
        ...     dims=["time", "station"],
        ... )

        >>> # Create 2 synthetic forecasts, which are the observations plus bias
        >>> ecmwf = obs + 10
        >>> gfs = obs + 20

        >>> blocks = {"time": 3, "station": 2}
        >>> n_iter = 5
        >>> boot_obs, boot_ecmwf, boot_gfs = block_bootstrap(
        ...     [obs, ecmwf, gfs],
        ...     blocks=blocks,
        ...     n_iteration=n_iter,
        ...     circular=True,
        ...     rng=100,
        ... )

        >>> boot_obs
        <xarray.DataArray (time: 6, station: 4, iteration: 5)> Size: 960B
        array([[[4. , 2.2, 2.3, 4.2, 1. ],
                [4.1, 2.3, 2. , 4.3, 1.1],
                [4.2, 2.3, 2.3, 4.3, 1.2],
                [4.3, 2. , 2. , 4. , 1.3]],
        <BLANKLINE>
               [[5. , 3.2, 3.3, 5.2, 2. ],
                [5.1, 3.3, 3. , 5.3, 2.1],
                [5.2, 3.3, 3.3, 5.3, 2.2],
                [5.3, 3. , 3. , 5. , 2.3]],
        <BLANKLINE>
               [[0. , 4.2, 4.3, 0.2, 3. ],
                [0.1, 4.3, 4. , 0.3, 3.1],
                [0.2, 4.3, 4.3, 0.3, 3.2],
                [0.3, 4. , 4. , 0. , 3.3]],
        <BLANKLINE>
               [[5. , 0.3, 4. , 1.2, 3.2],
                [5.1, 0. , 4.1, 1.3, 3.3],
                [5.1, 0.2, 4.2, 1.1, 3.2],
                [5.2, 0.3, 4.3, 1.2, 3.3]],
        <BLANKLINE>
               [[0. , 1.3, 5. , 2.2, 4.2],
                [0.1, 1. , 5.1, 2.3, 4.3],
                [0.1, 1.2, 5.2, 2.1, 4.2],
                [0.2, 1.3, 5.3, 2.2, 4.3]],
        <BLANKLINE>
               [[1. , 2.3, 0. , 3.2, 5.2],
                [1.1, 2. , 0.1, 3.3, 5.3],
                [1.1, 2.2, 0.2, 3.1, 5.2],
                [1.2, 2.3, 0.3, 3.2, 5.3]]])
        Coordinates:
          * time     (time) int64 48B 0 1 2 3 4 5
          * station  (station) <U2 32B 'S1' 'S2' 'S3' 'S4'
        Dimensions without coordinates: iteration

        >>> boot_ecmwf
        <xarray.DataArray (time: 6, station: 4, iteration: 5)> Size: 960B
        array([[[14. , 12.2, 12.3, 14.2, 11. ],
                [14.1, 12.3, 12. , 14.3, 11.1],
                [14.2, 12.3, 12.3, 14.3, 11.2],
                [14.3, 12. , 12. , 14. , 11.3]],
        <BLANKLINE>
               [[15. , 13.2, 13.3, 15.2, 12. ],
                [15.1, 13.3, 13. , 15.3, 12.1],
                [15.2, 13.3, 13.3, 15.3, 12.2],
                [15.3, 13. , 13. , 15. , 12.3]],
        <BLANKLINE>
               [[10. , 14.2, 14.3, 10.2, 13. ],
                [10.1, 14.3, 14. , 10.3, 13.1],
                [10.2, 14.3, 14.3, 10.3, 13.2],
                [10.3, 14. , 14. , 10. , 13.3]],
        <BLANKLINE>
               [[15. , 10.3, 14. , 11.2, 13.2],
                [15.1, 10. , 14.1, 11.3, 13.3],
                [15.1, 10.2, 14.2, 11.1, 13.2],
                [15.2, 10.3, 14.3, 11.2, 13.3]],
        <BLANKLINE>
               [[10. , 11.3, 15. , 12.2, 14.2],
                [10.1, 11. , 15.1, 12.3, 14.3],
                [10.1, 11.2, 15.2, 12.1, 14.2],
                [10.2, 11.3, 15.3, 12.2, 14.3]],
        <BLANKLINE>
               [[11. , 12.3, 10. , 13.2, 15.2],
                [11.1, 12. , 10.1, 13.3, 15.3],
                [11.1, 12.2, 10.2, 13.1, 15.2],
                [11.2, 12.3, 10.3, 13.2, 15.3]]])
        Coordinates:
          * time     (time) int64 48B 0 1 2 3 4 5
          * station  (station) <U2 32B 'S1' 'S2' 'S3' 'S4'
        Dimensions without coordinates: iteration
    """
    if rng is None:
        draw_integers = np.random.randint
    else:
        generator = np.random.default_rng(rng)
        draw_integers = generator.integers

    # While the most efficient method involves expanding the iteration dimension withing the
    # universal function, this approach might generate excessively large chunks (resulting
    # from multiplying chunk size by iterations) leading to issues with large numbers of
    # iterations. Hence, here function loops over blocks of iterations to generate the total
    # number of iterations.
    def _max_chunk_size_mb(var):
        """
        Estimate the largest chunk after joining bootstrap dimensions.
        """
        if var.chunks is None:
            return var.nbytes / (1024**2)
        chunk_shape = [var.sizes[d] if d in blocks else max(c) for d, c in zip(var.dims, var.chunks)]
        return var.dtype.itemsize * math.prod(chunk_shape) / (1024**2)

    if not isinstance(array_list, List):
        array_list = [array_list]
    variables = [
        var for obj in array_list for var in (obj.data_vars.values() if isinstance(obj, xr.Dataset) else [obj])
    ]
    # Check every variable, including mixed eager/Dask inputs and Datasets.
    if any(var.chunks is not None for var in variables):
        ds_max_chunk_size_mb = max(_max_chunk_size_mb(var) for var in variables)
        blocksize = int(MAX_BATCH_SIZE_MB / ds_max_chunk_size_mb)
        blocksize = min(blocksize, n_iteration)
        blocksize = max(blocksize, 1)
    else:
        blocksize = n_iteration

    bootstraps = []
    for start in range(0, n_iteration, blocksize):
        bootstraps.append(
            _block_bootstrap(
                array_list,
                blocks=blocks,
                n_iteration=min(blocksize, n_iteration - start),
                exclude_dims=exclude_dims,
                circular=circular,
                draw_integers=draw_integers,
                iteration_major=rng is not None,
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
