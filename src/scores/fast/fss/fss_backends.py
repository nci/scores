"""
Convenience module to retrieve compute backend
"""
from scores.fast.fss.fss_numba import FssNumba
from scores.fast.fss.fss_numpy import FssNumpy
from scores.fast.fss.typing import FssComputeMethod


def get_compute_backend(compute_method: FssComputeMethod):  # pragma: no cover
    """
    Returns the appropriate compute backend class constructor.
    """
    # Note: compute methods other than NUMPY currently not implemented
    if compute_method == FssComputeMethod.NUMPY:
        return FssNumpy
    if compute_method == FssComputeMethod.NUMBA:
        # Note: currently not implemented fore demonstration purposes only
        # this should throw a compatiblity error.
        return FssNumba
    raise NotImplementedError(f"Compute method {compute_method} not implemented.")
