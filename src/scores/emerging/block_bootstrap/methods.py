"""
Bootstrapping methods and configuration options go here
"""

from enum import Enum


class FitBlocksMethod(Enum):
    """
    Choice of method to fit blocks into axis, if the axis length is not a
    multiple of blocksize.

    .. note::

        ``PARTIAL`` is currently the only method that guarentees that the
        input array and sampled output array sizes will match.

        However, there may be scientific reasons for using "whole" blocks
        only, in which case ``SHRINK_TO_FIT`` or ``EXPAND_TO_FIT`` may be
        better options.
    """

    #: allows sampling of partial blocks (default)
    PARTIAL = 0
    #: shrinks axis length to fit whole blocks
    SHRINK_TO_FIT = 1
    #: expands axis tlength o fit whole blocks
    EXPAND_TO_FIT = 2


class OrderMissingDimsMethod(Enum):
    """
    Whether to prepend or append missing dimensions. Missing dimensions by
    default are sorted alphebetically.

    .. note::

        Currently by default only `ALPHABETICAL_PREPEND` is used in
        block-bootstrapping. This is because xarray's `apply_ufunc` method
        requires core dimensions to be at the end. Currently, the bootstrapping
        methods use `numpy`, so the core dimensions are essentially bootstrap
        dimensions as they will be not be broadcast, and will be re-sampled
        instead.
    """

    #: prepend missing dims alphebetically (default)
    ALPHABETICAL_PREPEND = 0
    #: append missing dims alphebetically
    ALPHABETICAL_APPEND = 1
    #: prepend missing dims without sorting
    UNSORTED_PREPEND = 2
    #: append missing dims without sorting
    UNSORTED_APPEND = 3
