"""
Contains unit tests for scores.dynamics.spectra_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

try:
    import pyshtools as pysh
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    pysh = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scores.dynamics.budgets_utils import (
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
    yhat2 = _scaled_rfft(y, axis=-1)
    ans = np.zeros(n // 2 + 1)
    ans[0] = 64.0
    ans[2] = 25.0
    ans[3] = 49.0
    ans[5] = 25.0
    xr.testing.assert_allclose(xr.DataArray(yhat2), xr.DataArray(ans), atol=1.0e-8)


def ux(phi, theta, p, phi1d):
    _u = np.zeros(phi.shape)

    mask1 = phi1d < -1.0e-6
    _u[:, mask1, :] = 4.0 + 3.0 * np.cos(2.0 * np.pi * theta[:, mask1, :] / 180.0)

    mask2 = phi1d > -1.0e-6
    mask3 = phi1d < +1.0e-6
    mask2 = np.logical_and(mask2, mask3)
    _u[:, mask2, :] = 3.0 * np.cos(3.0 * np.pi * theta[:, mask2, :] / 180.0) + 4.0 * np.sin(
        3.0 * np.pi * theta[:, mask2, :] / 180.0
    )

    return _u


def uy(phi, theta, p, phi1d):
    _v = np.zeros(phi.shape)

    mask1 = phi1d < -1.0e-6
    _v[:, mask1, :] = 4.0 * np.sin(2.0 * np.pi * theta[:, mask1, :] / 180.0)

    return _v


def omega(phi, theta, p, phi1d):
    _w = np.zeros(phi.shape)

    mask1 = phi1d < -1.0e-6
    _w[:, mask1, :] = 4.0 + 3.0 * np.cos(2.0 * np.pi * theta[:, mask1, :] / 90.0)

    mask2 = phi1d > -1.0e-6
    mask3 = phi1d < +1.0e-6
    mask2 = np.logical_and(mask2, mask3)
    _w[:, mask2, :] = 3.0 * np.cos(3.0 * np.pi * theta[:, mask2, :] / 90.0) + 4.0 * np.sin(
        3.0 * np.pi * theta[:, mask2, :] / 90.0
    )

    return _w


@pytest.mark.parametrize(
    (
        "u_velocity_func",
        "v_velocity_func",
        "custom_field_func",
        "reduce_vertical",
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
            True,
            True,
            pd.date_range("2025-01-01", periods=1),
            np.array([200, 800, 1000]),
            np.arange(0.0, 360.0, 20),
            np.array([-60.0, 0.0, 60.0]),
            xr.DataArray(
                np.array(
                    [
                        [
                            8.00000000e00,
                            0.00000000e00,
                            1.25000000e1,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            1.25000000e1,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ]
                )
            ),
        ),
        (
            ux,
            uy,
            omega,
            True,
            True,
            pd.date_range("2025-01-01", periods=1),
            np.array([200, 800, 1000]),
            np.arange(0.0, 180.0, 10),
            np.array([-60.0, 0.0, 60.0]),
            xr.DataArray(
                np.array(
                    [
                        [
                            1.014619e01,
                            2.473585e00,
                            2.521114e00,
                            1.376600e00,
                            7.955386e-01,
                            3.680748e-01,
                            1.238868e-01,
                            2.367847e-02,
                            8.529394e-04,
                            1.752576e-04,
                        ],
                        [
                            6.731174e-02,
                            2.882387e-01,
                            3.298907e-01,
                            1.945190e01,
                            3.469772e-01,
                            2.862770e-01,
                            1.955920e-01,
                            1.045931e-01,
                            3.959687e-02,
                            1.630067e-02,
                        ],
                        [
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                        ],
                    ]
                )
            ),
        ),
        (
            ux,
            uy,
            None,
            False,
            False,
            pd.date_range("2025-01-01", periods=3),
            np.array([500, 1000]),
            np.arange(0.0, 360.0, 20),
            np.array([-60.0]),
            xr.DataArray(
                np.array(
                    [
                        [
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                        ],
                        [
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                        ],
                        [
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                            [
                                [
                                    8.000000e00,
                                    0.000000e00,
                                    1.250000e01,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                    0.000000e00,
                                ]
                            ],
                        ],
                    ]
                )
            ),
        ),
    ],
)
def test_spectra(
    u_velocity_func,
    v_velocity_func,
    custom_field_func,
    reduce_vertical,
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

    u[:, :, :, :] = u_velocity_func(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = v_velocity_func(lat3d, lon3d, lev3d, latitude)

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
        w[:, :, :, :] = custom_field_func(lat3d, lon3d, lev3d, latitude)
        custom_field_name = "w"
        ds["w"] = (["time", "level", "latitude", "longitude"], w)
    else:
        custom_field_name = "none"

    spectra = power_spectra(
        ds,
        reduce_vertical=reduce_vertical,
        reduce_time=reduce_time,
        custom_field_name=custom_field_name,
    )

    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)


# test that the energy budget computation is compatible with Dask
def test_spectra_dask():
    if dask == "Unavailable":
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(0.0, 360.0, 40)
    latitude = np.array([-60.0, 0.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)

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

    spectra = power_spectra(ds.chunk(), reduce_vertical=True, reduce_time=True).chunk()
    assert isinstance(spectra["amplitude_squared"].data, dask.array.Array)
    spectra = spectra.compute()
    assert isinstance(spectra["amplitude_squared"].data, (np.ndarray, np.generic))
    expected = xr.DataArray(np.array([[8.0, 0.0, 12.5, 0.0, 0.0], [0.0, 0.0, 0.0, 12.5, 0.0]]))
    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)


def test_spectra_filter_width_error():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(90.0, 270.0, 10)
    latitude = np.array([-60.0, 0.0, +60.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)

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

    with pytest.raises(ValueError, match="The zonal_filter_width is: 0.55, must be < 0.5."):
        power_spectra(ds, zonal_filter_width=0.55)


def test_spectra_spherical_harmonic():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(90.0, 270.0, 10)
    latitude = np.array([-60.0, 0.0, +60.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)

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

    if pysh == "Unavailable":
        with pytest.raises(
            ImportError,
            match="The 'pyshtools' package is not installed, cannot perform a spherical harmonic transform.",
        ):
            power_spectra(ds, spherical_harmonic=True)
