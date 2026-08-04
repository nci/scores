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


def test_scaled_rfft():
    n = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    y = 8.0 + 5.0 * np.cos(2.0 * x) + 7.0 * np.sin(3.0 * x) + 3.0 * np.cos(5.0 * x) + 4.0 * np.sin(5.0 * x)
    kwargs = {axis: -1; norm: "forward"}
    yhat2 = _scaled_rfft(y, kwargs)
    ans = np.zeros(n//2+1)
    ans[0] = 64.0
    ans[2] = 25.0
    ans[3] = 49.0
    ans[5] = 25.0
    xr.testing.assert_allclose(xr.DataArray(yhat2), xr.DataArray(ans), atol=1.0e-8)


def ux(phi, theta, p):
    return 0.0


def uy(phi, theta, p):
    return 0.0


def omega(phi, theta, p):
    return 0.0


@pytest.mark.parametrize(
    (
        "u_velocity_func",
        "v_velocity_func",
        "custom_field_func",
        "preserve_vertical",
        "reduce_time",
        "time",
        "level",
        "longitude",
        "latitude",
        "expected",
    ),
    [
        (
            ux,
            uy,
            None,
            False,
            True,
            pd.date_range("2025-01-01", periods=1),
            np.array([200, 800, 1000]),
            np.arange(0.0, 360.0, 20, endpoint=False),
            np.array([-60.0, 0.0, 60.0]),
	    xr.DataArray(),
        ),
    ],
)
def test_spectra(
    u_velocity_func,
    v_velocity_func,
    custom_field_func,
    preserve_vertical,
    reduce_time,
    time,
    level,
    longitude,
    latitude,
    expected,
):
    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity_func(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity_func(lat3d, lon3d, lev3d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    if custom_field_func is not None:
        w = np.zeros((nt, nlev, nlat, nlon))
	w[0, :, :, :] = custom_field_func(lat3d, lon3d, lev3d)
	custom_field_name = "w"
	ds.["w"] = (["time", "level", "latitude", "longitude"], w)
    else:
	custom_field_name = "none"

    spectra = power_spectra(
	ds,
	preserve_vertical=preserve_vertical,
	reduce_time=reduce_time,
	custom_field_name,
	)

    xr.testing.assert_allclose(spectra, expected, atol=1.0e-6)
