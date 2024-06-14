"""
Helper functions for `block_bootstrap`

.. note::

    These may move to more generic utility functions in the future
"""

import xarray as xr
from typing import TypeVar, Tuple


T = TypeVar("T")


def unique_or_error(xs: list[T]):
    """
    Checks if input list has unique elements, otherwise raise an error.
    """
    if len(set(xs)) != len(xs):
        raise ValueError("List elements must be unique.")


def partial_linear_order_by_ref(xs: list[T], xs_ref: list[T]) -> Tuple[list[T], list[T]]:
    """
    Sorts elements in ``xs`` in the order that they appear in the ``xs_ref``.

    Elements in ``xs`` that are missing in ``xs_ref`` have an undefined
    ordering, and will be partitioned into a separate list.

    Args:
        xs: list of arbitrary type, with elements that can be ordered
        xs_ref: reference list, same type as xs, must have unique elements

    Returns:
        A partitioned list where,
            - The first list contains a partial linearly ordered copy of ``xs``
            - The second contains elements in ``xs`` that could not be found in
              ``xs_ref``
    Raises:
        ValueError: If list is not linearly ordered, i.e. has gaps.

    In this case, a partially linearly ordered list, ``xs`` is ordered in
    reference to a list ``xs_ref`` such that:

    .. code-block::
        if xs[i] == xs_ref[i] for all i < len(xs)

        then xs is partially linearly ordered.

        where,

        the comparision (<) reflects the order the elements appear in xs_ref:
        xs_ref[0] < xs_ref[1] < xs_ref[2] ...
    """
    # empty lists and lists with 1 element are ordered
    if len(xs) <= 1:
        return xs

    unique_or_error(xs_ref)

    # sort xs in the same order as xs_ref
    xs_sorted = [x for x in xs_ref if x in xs]

    # check for partial linear ordering as per description in docstring
    is_linord = all([xs_sorted[i] == x for i, x in enumerate(xs_ref[0 : len(xs_sorted)])])

    if not is_linord:
        xs_gaps = list(set(xs_ref[0 : len(xs_sorted)]) - set(xs_sorted))

        raise ValueError(
            f"Sorted list {xs_sorted} is not linearly ordered in comparision to"
            f" {xs_ref}, the following are gaps in the data: {xs_gaps}."
        )

    # also return elements missing in xs_ref for parent stack to handle
    xs_missing = list(set(xs) - set(xs_sorted))

    return (xs_sorted, xs_missing)


def reorder_dims(
    arr: xr.DataArray,
    dims: list[str],
    auto_order_missing: bool = True,
) -> xr.DataArray:
    """
    Reorders the dimensions of all arrays, with respect to a reference list
    of dimensions (``dims``), so that they are compatible.

    If ``dims`` does not contain a dimension in ``arr``, it cannot be
    ordered, and will cause an error to be raised. Optionally, if the
    ``auto_order_missing`` flag is ``True`` (default), then missing
    dimensions will be sent to the back (innermost axes) and ordered
    alphebatically so as to maintain consistency between all arrays.

    Args:

        arrs: data array reorder
        dims: unique list of reference dimensions that define the expected
            order.
        auto_order_missing: optionally order dimensinons in ``arrs`` that
            are not present in ``dims`` alphabetically.

    Returns:

        data array with dimensions ordered according to ``dims``.

    Raises:

        ValueError: if ``arr`` dimensions cannot be ordered due to missing
            dimensions in ``dims``
    """

    # partially ordered = (ordered dims, `arr.dims` not present in `dims`)
    (dims_ord, dims_unord) = partial_linear_order_by_ref(arr.dims, dims)

    # sort dims_unord alphebetically to resolve ordering between missing dimensions
    if auto_order_missing:
        dims_unord_sorted = sorted(dims_unord)
        dims_ord.extend(dims_unord_sorted)
    else:
        if len(dims_unord) > 0:
            raise ValueError(f"`auto_order_missing = False` unable to align unspecified dimensions: {dims_unord}")

    return arr.transpose(*tuple(dims_ord))


def reorder_all_arr_dims(
    arrs: list[xr.DataArray],
    dims: list[str],
    auto_order_missing: bool = True,
) -> list[xr.DataArray]:
    """
    Reorders the dimensions of all input arrays (``arrs``), with respect to
    a reference list of dimensions (``dims``), so that they are compatible.

    Calls py:func:`reorder_dims` for each array.

    Args:

        arrs: list of data arrays to reorder
        dims: unique list of reference dimensions that define the expected
            order.
        auto_order_missing: optionally order dimensinons in ``arrs`` that
            are not present in ``dims`` alphabetically.

    Returns:

        list of data arrays (same order as input); each array has its
        dimensions ordered according to ``dims``.

    Raises:

        DimensionError: if array dimensions are not valid.

    .. note::

        Dimensions of an array is valid if they are partially linearly
        ordered with reference to the ``dims`` argument.

        _All_ array dimensions are valid if the above holds for every array,
        _and_ at least one array has all the dimensions present in the
        ``dims`` argument.

    .. seealso::
        :py:func:`reorder_dims` for single array.

        For definition of partially linearly ordered in this context,
        see: :py:func:`partial_linear_order_by_ref`.
    """
    pass


def check_block_sizes(
    arrs: list[xr.DataArray],
    block_sizes: list[int],
    dims: list[int],
):
    """
    Checks that block sizes are valid.

    A block size is valid if its associated dimension in ``dims`` has a
    cardinality equal to or greater than the block size.

    i.e. ``len(arr.dim) >= block_size of dim``, for every array in ``arr``.
    """
    pass


def dask_chunk_iterations(
    arrs: list[xr.DataArray],
    iterations: int,
    flatten_dims: list[str],
    target_chunksize_mb: int = 200,
) -> int:
    """
    Chunks the number of iterations according to the maximum allowed size
    per chunk.

    Assumes dimensions in ``flatten_dims`` get collapsed into a single chunk.
    Which is usually the case for core dimensions in ``xr.apply_ufunc`` with the
    ``"parallelize"`` setting.

    Returns:

        Chunk size = number of iterations in a chunk

    The number of iterations per chunk is given by:

    .. code::

        data_chunk_size = n[1] * n[2] * ... * n[N] * m_1[1] * m_1[2] * ... * m_1[M]

        chunk_size = ceil(target_chunksize_mb / (data_chunk_size as mb))

        where,

            - n[1..N] are total sizes flatten_dims
            - m_1[1..M] is the size of a single chunk for dims not in flatten_dims

    .. warning::

        Unstable; for dask only. It may be safer to set ``chunk_size = 1`` and
        sacrifice speed, instead of using the output of this function. Or, let
        ``dask`` automatically decide.
    """
    raise NotImplementedError("`dask_chunk_iterations` not yet implemented")
