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
