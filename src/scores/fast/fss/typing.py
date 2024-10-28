"""
Specific type definitions for Fss backend    
"""
from enum import Enum
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

# Note: soft keyword `type` only support from >=3.10
# "tuple" of 64 bit floats
f8x3 = np.dtype("f8, f8, f8")
# Note: `TypeAlias` on variables not available for python <3.10
if TYPE_CHECKING:  # pragma: no cover
    FssDecomposed = Union[np.ArrayLike, np.DtypeLike]
else:  # pragma: no cover
    FssDecomposed = npt.NDArray[f8x3]


class FssComputeMethod(Enum):
    """
    Choice of compute backend for FSS.
    """

    #: invalid backend
    INVALID = -1
    #: computations using numpy (default)
    NUMPY = 1
    #: computations using numba (currently unimplemented)
    NUMBA = 2
