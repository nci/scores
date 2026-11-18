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
    STANDARD_CONSTANTS,
    _scaled_rfft_squared,
)
from scores.dynamics.spectra_impl import (
    power_spectra_scalar,
    power_spectra_vector,
)


def test_scaled_rfft_squared():
    n = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    y = 8.0 + 5.0 * np.cos(2.0 * x) + 7.0 * np.sin(3.0 * x) + 3.0 * np.cos(5.0 * x) + 4.0 * np.sin(5.0 * x)
    yhat2 = _scaled_rfft_squared(y, axis=-1)
    ans = np.zeros(n // 2 + 1)
    ans[0] = 64.0
    ans[2] = 25.0
    ans[3] = 49.0
    ans[5] = 25.0
    xr.testing.assert_allclose(xr.DataArray(yhat2), xr.DataArray(ans), atol=1.0e-8)


def ux(phi, theta, p, phi1d):
    _u = np.zeros(phi.shape)

    # southern hemisphere only
    mask1 = phi1d < -1.0e-6
    _u[:, mask1, :] = 4.0 + 3.0 * np.cos(2.0 * np.pi * theta[:, mask1, :] / 180.0)

    # equator only
    mask2 = phi1d > -1.0e-6
    mask3 = phi1d < +1.0e-6
    mask2 = np.logical_and(mask2, mask3)
    _u[:, mask2, :] = 3.0 * np.cos(3.0 * np.pi * theta[:, mask2, :] / 180.0) + 4.0 * np.sin(
        3.0 * np.pi * theta[:, mask2, :] / 180.0
    )

    return _u


def uy(phi, theta, p, phi1d):
    _v = np.zeros(phi.shape)

    # southern hemisphere only
    mask1 = phi1d < -1.0e-6
    _v[:, mask1, :] = 4.0 * np.sin(2.0 * np.pi * theta[:, mask1, :] / 180.0)

    return _v


def omega(phi, theta, p, phi1d):
    _w = np.zeros(phi.shape)

    # southern hemisphere only
    mask1 = phi1d < -1.0e-6
    _w[:, mask1, :] = 4.0 + 3.0 * np.cos(2.0 * np.pi * theta[:, mask1, :] / 90.0)

    # equator only
    mask2 = phi1d > -1.0e-6
    mask3 = phi1d < +1.0e-6
    mask2 = np.logical_and(mask2, mask3)
    np.savetxt("shape.txt", phi.shape)
    _w[:, mask2, :] = 3.0 * np.cos(3.0 * np.pi * theta[:, mask2, :] / 90.0) + 4.0 * np.sin(
        3.0 * np.pi * theta[:, mask2, :] / 90.0
    )

    return _w


def omega_variable_levels(phi, theta, p, phi1d):
    _w = np.zeros(phi.shape)

    # southern hemisphere only
    mask1 = phi1d < -1.0e-6
    _w[0, mask1, :] = 1.0 + 2.0 * np.cos(2.0 * np.pi * theta[0, mask1, :] / 90.0)
    _w[1, mask1, :] = 3.0 + 5.0 * np.cos(2.0 * np.pi * theta[1, mask1, :] / 90.0)

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
                            8.582760e00,
                            4.117347e00,
                            2.035150e00,
                            1.644566e00,
                            6.562332e-01,
                            1.561022e-01,
                            1.166555e-02,
                            3.572741e-03,
                            4.670897e-03,
                            3.904822e-04,
                        ],
                        [
                            3.255839e-03,
                            1.731906e-01,
                            5.229832e-01,
                            1.692262e01,
                            7.730010e-01,
                            5.229759e-01,
                            2.275568e-01,
                            4.824236e-02,
                            9.683661e-04,
                            1.067055e-03,
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
            omega,
            False,
            False,
            pd.date_range("2025-01-01", periods=2),
            np.array([500, 1000]),
            np.arange(0.0, 180.0, 10),
            np.array([-60.0, 0.0, 60.0]),
            xr.DataArray(
                np.array(
                    [
                        [
                            [
                                [
                                    8.582760e00,
                                    4.117347e00,
                                    2.035150e00,
                                    1.644566e00,
                                    6.562332e-01,
                                    1.561022e-01,
                                    1.166555e-02,
                                    3.572741e-03,
                                    4.670897e-03,
                                    3.904822e-04,
                                ],
                                [
                                    3.255839e-03,
                                    1.731906e-01,
                                    5.229832e-01,
                                    1.692262e01,
                                    7.730010e-01,
                                    5.229759e-01,
                                    2.275568e-01,
                                    4.824236e-02,
                                    9.683661e-04,
                                    1.067055e-03,
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
                            ],
                            [
                                [
                                    8.582760e00,
                                    4.117347e00,
                                    2.035150e00,
                                    1.644566e00,
                                    6.562332e-01,
                                    1.561022e-01,
                                    1.166555e-02,
                                    3.572741e-03,
                                    4.670897e-03,
                                    3.904822e-04,
                                ],
                                [
                                    3.255839e-03,
                                    1.731906e-01,
                                    5.229832e-01,
                                    1.692262e01,
                                    7.730010e-01,
                                    5.229759e-01,
                                    2.275568e-01,
                                    4.824236e-02,
                                    9.683661e-04,
                                    1.067055e-03,
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
                            ],
                        ],
                        [
                            [
                                [
                                    8.582760e00,
                                    4.117347e00,
                                    2.035150e00,
                                    1.644566e00,
                                    6.562332e-01,
                                    1.561022e-01,
                                    1.166555e-02,
                                    3.572741e-03,
                                    4.670897e-03,
                                    3.904822e-04,
                                ],
                                [
                                    3.255839e-03,
                                    1.731906e-01,
                                    5.229832e-01,
                                    1.692262e01,
                                    7.730010e-01,
                                    5.229759e-01,
                                    2.275568e-01,
                                    4.824236e-02,
                                    9.683661e-04,
                                    1.067055e-03,
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
                            ],
                            [
                                [
                                    8.582760e00,
                                    4.117347e00,
                                    2.035150e00,
                                    1.644566e00,
                                    6.562332e-01,
                                    1.561022e-01,
                                    1.166555e-02,
                                    3.572741e-03,
                                    4.670897e-03,
                                    3.904822e-04,
                                ],
                                [
                                    3.255839e-03,
                                    1.731906e-01,
                                    5.229832e-01,
                                    1.692262e01,
                                    7.730010e-01,
                                    5.229759e-01,
                                    2.275568e-01,
                                    4.824236e-02,
                                    9.683661e-04,
                                    1.067055e-03,
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
                            ],
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
        (
            ux,
            uy,
            None,
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
                            4.007167e00,
                            7.697420e00,
                            1.742463e00,
                            9.312828e-01,
                            3.444901e-01,
                            7.080315e-02,
                            8.429491e-03,
                            1.204080e-02,
                            9.049734e-03,
                            1.682182e-03,
                        ],
                        [
                            2.172919e-01,
                            4.961559e00,
                            4.966927e00,
                            3.829931e-01,
                            4.310792e-02,
                            4.180241e-03,
                            1.748840e-02,
                            2.005637e-02,
                            7.987944e-03,
                            1.771520e-04,
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
            omega_variable_levels,
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
                            4.875000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            1.400000e01,
                            0.000000e-01,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
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
        ds["w"] = (["time", "level", "latitude", "longitude"], w)
        spectra = power_spectra_scalar(
            ds,
            reduce_vertical=reduce_vertical,
            reduce_time=reduce_time,
            scalar_field_name="w",
        )
    else:
        spectra = power_spectra_vector(
            ds,
            reduce_vertical=reduce_vertical,
            reduce_time=reduce_time,
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

    spectra = power_spectra_vector(ds.chunk(), reduce_vertical=True, reduce_time=True).chunk()
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
        power_spectra_vector(ds, zonal_filter_width=0.55)


def test_spectra_vector_with_vertical_weights():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 360.0, 20)
    latitude = np.array([-60.0, 0.0, 60.0])

    expected = xr.DataArray(
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
    )

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

    dp = np.zeros(nlev)
    dp[1:-1] = 0.5 * (level[2:] - level[:-2])
    dp[0] = 0.5 * (level[1] - level[0])
    dp[-1] = 0.5 * (level[-1] - level[-2])
    dp = 100.0 * dp / STANDARD_CONSTANTS.GRAVITY
    dp = dp / np.sum(dp)

    spectra = power_spectra_vector(ds, reduce_vertical=True, reduce_time=True, vertical_weights=dp)

    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)


def test_spectra_vector_with_nan_u():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 360.0, 20)
    latitude = np.array([-60.0, 0.0, 60.0])

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
    u[0, 1, 2, 3] = np.nan

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

    with pytest.raises(ValueError, match="NaN value found in field: u."):
        power_spectra_vector(ds, reduce_vertical=True, reduce_time=True)


def test_spectra_vector_with_nan_v():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 360.0, 20)
    latitude = np.array([-60.0, 0.0, 60.0])

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
    v[0, 1, 2, 3] = np.nan

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

    with pytest.raises(ValueError, match="NaN value found in field: v."):
        power_spectra_vector(ds, reduce_vertical=True, reduce_time=True)


def test_spectra_vector_with_vertical_weights_error():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 360.0, 20)
    latitude = np.array([-60.0, 0.0, 60.0])

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

    dp = np.ones(nlev + 1)

    with pytest.raises(
        IndexError, match="Length of supplied vertical weights array: 4 does not match length of level: 3."
    ):
        power_spectra_vector(ds, reduce_vertical=True, reduce_time=True, vertical_weights=dp)


def test_spectra_scalar_with_vertical_weights():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 180.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    expected = xr.DataArray(
        np.array(
            [
                [
                    8.582760e00,
                    4.117347e00,
                    2.035150e00,
                    1.644566e00,
                    6.562332e-01,
                    1.561022e-01,
                    1.166555e-02,
                    3.572741e-03,
                    4.670897e-03,
                    3.904822e-04,
                ],
                [
                    3.255839e-03,
                    1.731906e-01,
                    5.229832e-01,
                    1.692262e01,
                    7.730010e-01,
                    5.229759e-01,
                    2.275568e-01,
                    4.824236e-02,
                    9.683661e-04,
                    1.067055e-03,
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
    )

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[:, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    dp = np.zeros(nlev)
    dp[1:-1] = 0.5 * (level[2:] - level[:-2])
    dp[0] = 0.5 * (level[1] - level[0])
    dp[-1] = 0.5 * (level[-1] - level[-2])
    dp = 100.0 * dp / STANDARD_CONSTANTS.GRAVITY
    dp = dp / np.sum(dp)

    spectra = power_spectra_scalar(
        ds, reduce_vertical=True, reduce_time=True, scalar_field_name="w", vertical_weights=dp
    )

    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)


def test_spectra_scalar_with_vertical_weights_error():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 180.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[:, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    dp = np.ones(nlev + 1)

    with pytest.raises(
        IndexError, match="Length of supplied vertical weights array: 4 does not match length of level: 3."
    ):
        power_spectra_scalar(ds, reduce_vertical=True, reduce_time=True, scalar_field_name="w", vertical_weights=dp)


def test_spectra_scalar_with_nan():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 180.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[:, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)
    w[0, 1, 2, 3] = np.nan

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    with pytest.raises(ValueError, match="NaN value found in field: w."):
        power_spectra_scalar(ds, reduce_vertical=True, reduce_time=True, scalar_field_name="w")


def test_spectra_scalar_with_domain_error():
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 400.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[:, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    with pytest.raises(ValueError, match="Zonal domain size: 400.0 > 360.0 degrees, must be <= 360.0 degrees."):
        power_spectra_scalar(ds, reduce_vertical=True, reduce_time=True, scalar_field_name="w")


def test_spectra_spherical_harmonic_vector():
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

    with pytest.raises(
        NotImplementedError, match="Support for spherical harmonic transform spectra has not been implemented yet."
    ):
        power_spectra_vector(ds, spherical_harmonic=True)


def test_spectra_spherical_harmonic_scalar():
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
    w = np.zeros((nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[:, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[:, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[:, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    with pytest.raises(
        NotImplementedError, match="Support for spherical harmonic transform spectra has not been implemented yet."
    ):
        power_spectra_scalar(ds, spherical_harmonic=True)


def test_spectra_scalar_with_ensemble_averaging():
    ensemble = np.arange(2) + 1
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 180.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    expected = xr.DataArray(
        np.array(
            [
                [
                    5.517972e01,
                    1.205473e01,
                    3.450323e01,
                    1.163004e01,
                    3.892134e00,
                    5.558260e-01,
                    1.690237e-02,
                    1.791440e-01,
                    1.103715e-01,
                    3.736703e-04,
                ],
                [
                    1.568831e01,
                    8.450467e-01,
                    8.288169e-01,
                    2.702585e01,
                    6.763590e-01,
                    5.117548e-01,
                    3.183059e-01,
                    1.477136e-01,
                    3.832006e-02,
                    9.378005e-04,
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
    )

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((2, nt, nlev, nlat, nlon))
    v = np.zeros((2, nt, nlev, nlat, nlon))
    w = np.zeros((2, nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[0, :, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[0, :, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    u[1, :, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[1, :, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[1, :, :, :, :] = 0.5 * (u[1, :, :, :] ** 2 + v[1, :, :, :] ** 2)

    ds = xr.Dataset(
        data_vars={
            "u": (["ensemble_index", "time", "level", "latitude", "longitude"], u),
            "v": (["ensemble_index", "time", "level", "latitude", "longitude"], v),
            "w": (["ensemble_index", "time", "level", "latitude", "longitude"], w),
        },
        coords={
            "ensemble_index": ensemble,
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    spectra = power_spectra_scalar(
        ds, reduce_vertical=True, reduce_time=True, reduce_ensemble=True, scalar_field_name="w"
    )

    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)


def test_spectra_scalar_with_ensemble_no_averaging():
    ensemble = np.arange(2) + 1
    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([200, 800, 1000])
    longitude = np.arange(0.0, 180.0, 10)
    latitude = np.array([-60.0, 0.0, 60.0])

    expected = xr.DataArray(
        np.array(
            [
                [
                    [
                        8.582760e00,
                        4.117347e00,
                        2.035150e00,
                        1.644566e00,
                        6.562332e-01,
                        1.561022e-01,
                        1.166555e-02,
                        3.572741e-03,
                        4.670897e-03,
                        3.904822e-04,
                    ],
                    [
                        3.255839e-03,
                        1.731906e-01,
                        5.229832e-01,
                        1.692262e01,
                        7.730010e-01,
                        5.229759e-01,
                        2.275568e-01,
                        4.824236e-02,
                        9.683661e-04,
                        1.067055e-03,
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
                ],
                [
                    [
                        1.017767e02,
                        1.999212e01,
                        6.697131e01,
                        2.161551e01,
                        7.128035e00,
                        9.555498e-01,
                        2.213920e-02,
                        3.547153e-01,
                        2.160720e-01,
                        3.568584e-04,
                    ],
                    [
                        3.137336e01,
                        1.516903e00,
                        1.134651e00,
                        3.712908e01,
                        5.797169e-01,
                        5.005338e-01,
                        4.090550e-01,
                        2.471849e-01,
                        7.567176e-02,
                        8.085459e-04,
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
                ],
            ]
        )
    )

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((2, nt, nlev, nlat, nlon))
    v = np.zeros((2, nt, nlev, nlat, nlon))
    w = np.zeros((2, nt, nlev, nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[0, :, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[0, :, :, :, :] = omega(lat3d, lon3d, lev3d, latitude)

    u[1, :, :, :, :] = ux(lat3d, lon3d, lev3d, latitude)
    v[1, :, :, :, :] = uy(lat3d, lon3d, lev3d, latitude)
    w[1, :, :, :, :] = 0.5 * (u[1, :, :, :] ** 2 + v[1, :, :, :] ** 2)

    ds = xr.Dataset(
        data_vars={
            "u": (["ensemble_index", "time", "level", "latitude", "longitude"], u),
            "v": (["ensemble_index", "time", "level", "latitude", "longitude"], v),
            "w": (["ensemble_index", "time", "level", "latitude", "longitude"], w),
        },
        coords={
            "ensemble_index": ensemble,
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    spectra = power_spectra_scalar(
        ds, reduce_vertical=True, reduce_time=True, reduce_ensemble=False, scalar_field_name="w"
    )

    xr.testing.assert_allclose(xr.DataArray(spectra["amplitude_squared"].data), expected, atol=1.0e-6)
