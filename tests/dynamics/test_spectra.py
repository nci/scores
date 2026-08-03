"""
Contains unit tests for scores.dynamics.spectra_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover


import numpy as np
import pytest
import xarray as xr

from scores.dynamics.budgets_utils import (
    STANDARD_CONSTANTS,
    PlanetConstants,
    _scaled_rfft,
)
from scores.dynamics.spectra_impl import (
    power_spectra,
)

