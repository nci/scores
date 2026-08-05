"""
Contains unit tests for scores.dynamics.spectra_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover


import numpy as np
import xarray as xr

from scores.dynamics.budgets_utils import (
    _scaled_rfft,
)


def test_scaled_rfft():
    n = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    y = 8.0 + 5.0 * np.cos(2.0 * x) + 7.0 * np.sin(3.0 * x) + 3.0 * np.cos(5.0 * x) + 4.0 * np.sin(5.0 * x)
    yhat2 = _scaled_rfft(y, axis=-1)
    ans = np.zeros(n // 2 + 1)
    ans[0] = 64.0
    ans[2] = 25.0
    ans[3] = 49.0
    ans[5] = 25.0
    xr.testing.assert_allclose(xr.DataArray(yhat2), xr.DataArray(ans), atol=1.0e-8)
