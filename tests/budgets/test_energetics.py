"""
Contains unit tests for scores.budgets.energetics_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

import operator

import numpy as np
import pytest
import xarray as xr

