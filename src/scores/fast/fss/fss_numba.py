"""
Backend for computing Fractions Skill Score (FSS) using numba
    
"""
from dataclasses import dataclass

from scores.fast.fss.backend import FssBackend
from scores.fast.fss.typing import FssComputeMethod

_COMPATIBLE = True
try:
    import numba  # pylint: disable=unused-import
except ImportError:  # pragma: no cover
    _COMPATIBLE = False


@dataclass
class FssNumba(FssBackend):  # pragma: no cover
    """
    Implementation of numba backend for computing FSS

    This is currently a stub.
    """

    compute_method = FssComputeMethod.NUMBA

    def _check_compatibility(self):
        """
        Currently not implemented.
        """
        if _COMPATIBLE:
            return
        super()._check_compatibility()

    def _compute_integral_field(self):
        super()._compute_integral_field()

    def _compute_fss_components(self):
        super()._compute_integral_field()
