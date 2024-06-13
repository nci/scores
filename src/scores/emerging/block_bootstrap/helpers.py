"""
Helper functions for `block_bootstrap`

.. note::

    These may move to more generic utility functions in the future
"""

import xarray as xr
from typing import TypeVar, Tuple


T = TypeVar('T')


def partial_linear_order_by_ref(
    xs: list[T],
    xs_ref: list[T]
) -> Tuple[list[T], list[T]]:
    """
    Sorts elements in ``xs`` in the order that they appear in the ``xs_ref``.

    Elements in ``xs`` that are missing in ``xs_ref`` have an undefined
    ordering, and will be partitioned into a separate list.

    Args:
        xs: list of arbitrary type, with elements that can be ordered
        xs_ref: reference list, same type as xs

    Returns:
        A partitioned list where,
            - The first list contains a linearly ordered copy of ``xs``
            - The second contains elements in `xs` that could not be found in
              ``xs_ref``
    Raises:
        ValueError: If list is not linearly ordered, i.e. has gaps.

    In this case, a linearly ordered list, ``xs`` is ordered in reference to a
    list ``xs_ref`` such that:

    .. code-block::
        xs[i] == xs_ref[i] for all i < len(xs)

        where,

        the comparision (<) reflects the order the elements appear in xs_ref:
        xs_ref[0] < xs_ref[1] < xs_ref[2] ...
    """

    if len(xs) <= 1:
        return xs

    xs_sorted = [ x for x in xs_ref if x in xs ]
    is_linord = all([
        xs_sorted[i] == x
        for i, x in enumerate(xs_ref[0:len(xs_sorted)])
    ])

    if not is_linord:
        xs_gaps = list(set(xs_ref[0:len(xs_sorted)]) - set(xs_sorted))

        raise ValueError(
            f"Sorted list {xs_sorted} is not linearly ordered in comparision to"
            f" {xs_ref}, the following are gaps in the data: {xs_gaps}."
        )

    xs_missing = list(set(xs) - set(xs_sorted))

    return (xs_sorted, xs_missing)


def reorder_dims(
    arr: xr.DataArray,
    dims: list[str],
    auto_order_missing: bool = True,
) -> xr.DataArray:
    # (ordered dims, `arr.dims` not present in `dims`)
    (dims_ord, dims_unord) = partial_linear_order_by_ref(arr.dims, dims)

    # sort dims_unord alphebetically to ensure consistency across multiple arrays.
    if auto_order_missing:
        dims_unord_sorted = sorted(dims_unord)
        dims_ord.extend(dims_unord_sorted)
    else:
        if len(dims_unord) > 0:
            raise ValueError(
                f"`auto_order_missing = False` unable to align unspecified dimensions: {dims_unord}"
            )

    return arr.transpose(*tuple(dims_ord))
