"""
Contains unit tests for scores.dynamics.energetics_impl
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scores.dynamics.budgets_utils import (
    STANDARD_CONSTANTS,
    PlanetConstants,
    _integrate_energy_exchange,
    _integrate_horizontal,
    _integration_weights,
    _trig_fields,
)
from scores.dynamics.energetics_impl import (
    energy_components_lat_lon,
    energy_exchanges_lat_lon,
)

# Williamson 5 test case initial condition stream function
# Reference:
#   Williamson, et al. (1992) JCP vol. 102 pp 211--224, eqn. (92)
#
# phi: azimuthal angle
# theta: polar angle (from the equator)

area_earth = 4.0 * np.pi * STANDARD_CONSTANTS.RAD_EARTH * STANDARD_CONSTANTS.RAD_EARTH
u_0 = 2.0 * np.pi * STANDARD_CONSTANTS.RAD_EARTH / (12.0 * 24.0 * 60.0 * 60.0)


def stream_function(phi, theta, alpha):
    _phi = np.deg2rad(phi)
    _theta = np.deg2rad(theta)

    return (
        -STANDARD_CONSTANTS.RAD_EARTH
        * u_0
        * (np.sin(_theta) * np.cos(alpha) - np.cos(_phi) * np.cos(_theta) * np.sin(alpha))
    )


# derived from the stream function, \psi as:
#   1/(r cos(theta)) d\psi/d\phi
# see eqn. (91) of Williamson et al. (1992)
def d_psi_d_phi(phi, theta, alpha):
    _phi = np.deg2rad(phi)
    _theta = np.deg2rad(theta)

    return -u_0 * np.sin(_phi) * np.sin(alpha)


# derived from the stream function, \psi as:
#   1/r d\psi/d\theta
# see eqn. (90) of Williamson et al. (1992)
def d_psi_d_theta(phi, theta, alpha):
    _phi = np.deg2rad(phi)
    _theta = np.deg2rad(theta)

    return -u_0 * (np.cos(_theta) * np.cos(alpha) + np.cos(_phi) * np.sin(_theta) * np.sin(alpha))


# spherical Laplacian of the stream function
# see eqn. (94) of Williamson et al. (1992)
def vorticity(phi, theta, alpha):
    _phi = np.deg2rad(phi)
    _theta = np.deg2rad(theta)

    return 2.0 * u_0 * (-np.cos(_phi) * np.cos(_theta) * np.sin(alpha) + np.sin(_theta) * np.cos(alpha))


# init test for the spherical integration via convergence of errors with resolution
@pytest.mark.parametrize(
    (
        "field",
        "alpha",
        "offset",
        "low_resolution",
        "num_resolutions",
        "sub_domain_longitude",
        "sub_domain_latitude",
        "analytic_solution",
        "convergence_rate",
        "tolerance",
    ),
    [
        # Global integral - test latitude/longitude integration for full domain
        (
            vorticity,
            0.25 * np.pi,
            10.0,
            np.array([30, 60], dtype=np.int64),
            4,
            None,
            None,
            area_earth * 10.0,
            4.0,
            1.0e-3,
        ),
        # Northern hemisphere - test latitude/longitude integration for latitudinal sub-domain
        (
            vorticity,
            0.5 * np.pi,
            10.0,
            np.array([30, 60], dtype=np.int64),
            4,
            None,
            np.array([0.0, 90.0]),
            0.5 * area_earth * 10.0,
            2.0,
            2.0e-2,
        ),
        # Longitudinal sub-domain
        (
            vorticity,
            0.0,
            10.0,
            np.array([30, 60], dtype=np.int64),
            4,
            np.array([90.0, 180.0]),
            None,
            0.25 * area_earth * 10.0,
            4.0,
            1.0e-3,
        ),
        # Sub-domain in latitude and longitude
        (
            vorticity,
            0.5 * np.pi,
            10.0,
            np.array([60, 120], dtype=np.int64),
            4,
            np.array([45.0, 135.0]),
            np.array([-60.0, +60.0]),
            STANDARD_CONSTANTS.RAD_EARTH
            * STANDARD_CONSTANTS.RAD_EARTH
            * np.pi
            * (135.0 - 45.0)
            / 180.0
            * (np.sin(np.pi / 3.0) + np.sin(np.pi / 3.0))
            * 10.0,
            2.0,
            1.0e-2,
        ),
    ],
)
def test_budgets_integral(
    field,
    alpha,
    offset,
    low_resolution,
    num_resolutions,
    sub_domain_longitude,
    sub_domain_latitude,
    analytic_solution,
    convergence_rate,
    tolerance,
):
    error_at_res = np.zeros(num_resolutions)

    for res in np.arange(num_resolutions):
        nlon = low_resolution[1] * np.power(2, res)
        nlat = low_resolution[0] * np.power(2, res)

        longitude = np.linspace(-180.0, +180.0, nlon, endpoint=False)
        if sub_domain_longitude is not None:
            longitude = longitude[
                (longitude > sub_domain_longitude[0] - 1.0e-6) & (longitude < sub_domain_longitude[1] + 1.0e-6)
            ]

        latitude = np.linspace(-90.0, +90.0, nlat, endpoint=False)
        if sub_domain_latitude is not None:
            latitude = latitude[
                (latitude > sub_domain_latitude[0] - 1.0e-6) & (latitude < sub_domain_latitude[1] + 1.0e-6)
            ]

        dlon, dlat = _integration_weights(longitude, latitude, "longitude", "latitude", STANDARD_CONSTANTS)
        nlon = len(dlon)
        nlat = len(dlat)

        lon2d, lat2d = np.meshgrid(dlon.longitude, dlat.latitude)
        _psi = field(lon2d, lat2d, alpha) + offset

        psi = xr.DataArray(
            _psi, dims=["latitude", "longitude"], coords={"latitude": dlat.latitude, "longitude": dlon.longitude}
        )

        int_psi = _integrate_horizontal(psi, dlon, dlat, False)

        error_at_res[res] = (analytic_solution - int_psi.data) / analytic_solution

    convergence = np.zeros(num_resolutions - 1)
    for res in np.arange(num_resolutions - 1):
        convergence[res] = error_at_res[res] / error_at_res[res + 1]

    # for second order accuracy, we anticipate the integration error should decrease by a factor of 4 for a
    # doubling of the spatial resolution (to within some tolerance)
    xr.testing.assert_allclose(
        xr.DataArray(convergence), xr.DataArray(convergence_rate * np.ones(num_resolutions - 1)), atol=tolerance
    )


# unit test for the energy exchanges
@pytest.mark.parametrize(
    (
        "field",
        "zonal_gradient",
        "meridional_gradient",
        "div_vec",
        "alpha",
        "low_resolution",
        "num_resolutions",
        "sub_domain_longitude",
        "sub_domain_latitude",
        "convergence_rate",
        "tolerance",
    ),
    [
        # Global integral
        (
            stream_function,
            d_psi_d_phi,
            d_psi_d_theta,
            vorticity,
            0.25 * np.pi,
            np.array([60, 120], dtype=np.int64),
            3,
            None,
            None,
            2.0,
            2.0e-2,
        ),
    ],
)
def test_budgets_gradient(
    field,
    zonal_gradient,
    meridional_gradient,
    div_vec,
    alpha,
    low_resolution,
    num_resolutions,
    sub_domain_longitude,
    sub_domain_latitude,
    convergence_rate,
    tolerance,
):
    error_at_res_grad_f_dot_u = np.zeros(num_resolutions)
    error_at_res_f_div_u = np.zeros(num_resolutions)
    balance_error_at_res = np.zeros(num_resolutions)

    for res in np.arange(num_resolutions):
        nlon = low_resolution[1] * np.power(2, res)
        nlat = low_resolution[0] * np.power(2, res)

        longitude = np.linspace(-180.0, +180.0, nlon)
        if sub_domain_longitude is not None:
            longitude = longitude[
                (longitude > sub_domain_longitude[0] - 1.0e-6) & (longitude < sub_domain_longitude[1] + 1.0e-6)
            ]
        latitude = np.linspace(-90.0, +90.0, nlat)
        if sub_domain_latitude is not None:
            latitude = latitude[
                (latitude > sub_domain_latitude[0] - 1.0e-6) & (latitude < sub_domain_latitude[1] + 1.0e-6)
            ]

        dlon, dlat = _integration_weights(longitude, latitude, "longitude", "latitude", STANDARD_CONSTANTS)
        nlon = len(dlon)
        nlat = len(dlat)
        cos_theta, sin_theta, cos_theta_inv = _trig_fields(dlon.longitude, dlat.latitude, "longitude", "latitude")

        lon2d, lat2d = np.meshgrid(dlon.longitude, dlat.latitude)
        _psi = field(lon2d, lat2d, alpha)
        _dPsiDx = zonal_gradient(lon2d, lat2d, alpha)
        _dPsiDy = meridional_gradient(lon2d, lat2d, alpha)
        _lapPsi = div_vec(lon2d, lat2d, alpha)

        psi = xr.DataArray(
            _psi, dims=["latitude", "longitude"], coords={"latitude": dlat.latitude, "longitude": dlon.longitude}
        )
        dPsiDx = xr.DataArray(
            _dPsiDx, dims=["latitude", "longitude"], coords={"latitude": dlat.latitude, "longitude": dlon.longitude}
        )
        dPsiDy = xr.DataArray(
            _dPsiDy, dims=["latitude", "longitude"], coords={"latitude": dlat.latitude, "longitude": dlon.longitude}
        )
        lapPsi = xr.DataArray(
            _lapPsi, dims=["latitude", "longitude"], coords={"latitude": dlat.latitude, "longitude": dlon.longitude}
        )

        grad_f_dot_u, f_div_u = _integrate_energy_exchange(
            psi,
            dPsiDx,
            dPsiDy,
            dlon,
            dlat,
            cos_theta,
            sin_theta,
            cos_theta_inv,
            "longitude",
            "latitude",
            STANDARD_CONSTANTS,
        )

        err1 = grad_f_dot_u - (dPsiDx * dPsiDx + dPsiDy * dPsiDy)
        err2 = f_div_u - psi * lapPsi
        error_at_res_grad_f_dot_u[res] = _integrate_horizontal(err1, dlon, dlat, False)
        error_at_res_f_div_u[res] = _integrate_horizontal(err2, dlon, dlat, False)
        balance_error_at_res[res] = np.abs(error_at_res_grad_f_dot_u[res] + error_at_res_f_div_u[res]) / np.abs(
            error_at_res_grad_f_dot_u[res]
        )

    convergence = np.zeros(num_resolutions - 1)
    for res in np.arange(num_resolutions - 1):
        convergence[res] = balance_error_at_res[res] / balance_error_at_res[res + 1]

    xr.testing.assert_allclose(
        xr.DataArray(convergence), xr.DataArray(convergence_rate * np.ones(num_resolutions - 1)), atol=tolerance
    )
    xr.testing.assert_allclose(xr.DataArray(balance_error_at_res), xr.DataArray(np.zeros(num_resolutions)), atol=1.0e-2)


def u_velocity(phi, theta, p):
    u = -1.0e-6 * p * p * d_psi_d_theta(phi, theta, 0.25 * np.pi)
    return u


def v_velocity(phi, theta, p):
    v = +1.0e-6 * p * p * d_psi_d_phi(phi, theta, 0.25 * np.pi)
    return v


def w_velocity(phi, theta, p):
    w = 0.0
    return w


def temperature(phi, theta, p):
    t = 1.0e-3 * p * (90.0 - phi) * (90.0 - phi)
    return t


def humidity(phi, theta, p):
    q = 1.0e-10 * p
    return q


def geopotential(phi, theta, p):
    z = 1.0e-9 * p * p * p * vorticity(phi, theta, 0.25 * np.pi)
    return z


def surface_pressure(phi, theta):
    sp = 1.0e-16 * stream_function(phi, theta, 0.25 * np.pi) * stream_function(phi, theta, 0.25 * np.pi)
    return sp


# test the energy budget against a previously computed solution
@pytest.mark.parametrize(
    (
        "time",
        "level",
        "longitude",
        "latitude",
        "sub_domain_longitude",
        "sub_domain_latitude",
        "preserve_horizontal",
        "preserve_vertical",
        "u_velocity_func",
        "v_velocity_func",
        "w_velocity_func",
        "temperature_func",
        "humidity_func",
        "geopotential_func",
        "surface_pressure_func",
        "expected",
    ),
    [
        # test energy budget for full domain
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            None,
            None,
            False,
            False,
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray([[2.505574e25], [6.481852e17], [1.285482e20], [4.824884e20], [0.000000e00]]),
        ),
        # test energy budget for full domain with shifted longitudinal coordinate
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(-180.0, 180.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            None,
            None,
            False,
            False,
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray([[2.505574e25], [6.481852e17], [1.285482e20], [4.824884e20], [0.000000e00]]),
        ),
        # test energy budget for sub domain in latitude and longitude
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            np.array([90.0, 180.0]),
            np.array([-45.0, 45.0]),
            False,
            False,
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray([[4.281888e24], [1.223907e17], [4.243540e19], [4.237897e19], [0.000000e00]]),
        ),
        # test energy budget while preserving the vertical dimension
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            None,
            None,
            False,
            True,
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray(
                [
                    [1.255927e23, 7.535562e23, 1.569909e24, 3.516596e24, 6.782006e24, 8.540304e24, 3.767781e24],
                    [3.249049e15, 1.949429e16, 4.061311e16, 9.097336e16, 1.754486e17, 2.209353e17, 9.747146e16],
                    [0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00],
                    [7.059522e14, 1.143643e17, 1.103050e18, 1.012053e19, 6.587382e19, 2.358473e20, 1.694285e20],
                    [0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00],
                ]
            ),
        ),
        # test energy budget while preserving the horizontal dimensions
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 30),
            np.linspace(-90.0, 90.0, 7, endpoint=True),
            None,
            None,
            True,
            False,
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray(
                [
                    [
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                            6.40553207e23,
                        ],
                        [
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                            7.10061247e23,
                        ],
                        [
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                            4.61198309e23,
                        ],
                        [
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                            1.77515312e23,
                        ],
                        [
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
                            2.56221283e22,
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                        ],
                        [
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                        ],
                        [
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                            1.41541450e16,
                        ],
                        [
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                            1.22578492e16,
                        ],
                        [
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
                            7.07707252e15,
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            4.20903592e17,
                            7.55487956e15,
                            6.38910267e17,
                            1.68361437e18,
                            2.09696308e18,
                            1.46560769e18,
                            4.20903592e17,
                            7.55487956e15,
                            6.38910267e17,
                            1.68361437e18,
                            2.09696308e18,
                            1.46560769e18,
                        ],
                        [
                            2.18707922e18,
                            1.82256601e17,
                            5.46769804e17,
                            2.91610562e18,
                            4.92092824e18,
                            4.55641503e18,
                            2.18707922e18,
                            1.82256601e17,
                            5.46769804e17,
                            2.91610562e18,
                            4.92092824e18,
                            4.55641503e18,
                        ],
                        [
                            3.36722873e18,
                            4.51123110e17,
                            4.51123110e17,
                            3.36722873e18,
                            6.28333435e18,
                            6.28333435e18,
                            3.36722873e18,
                            4.51123110e17,
                            4.51123110e17,
                            3.36722873e18,
                            6.28333435e18,
                            6.28333435e18,
                        ],
                        [
                            2.18707922e18,
                            1.82256601e17,
                            5.46769804e17,
                            2.91610562e18,
                            4.92092824e18,
                            4.55641503e18,
                            2.18707922e18,
                            1.82256601e17,
                            5.46769804e17,
                            2.91610562e18,
                            4.92092824e18,
                            4.55641503e18,
                        ],
                        [
                            4.20903592e17,
                            7.55487956e15,
                            6.38910267e17,
                            1.68361437e18,
                            2.09696308e18,
                            1.46560769e18,
                            4.20903592e17,
                            7.55487956e15,
                            6.38910267e17,
                            1.68361437e18,
                            2.09696308e18,
                            1.46560769e18,
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                        [
                            7.90398909e18,
                            9.01286322e18,
                            7.31915127e18,
                            4.51656520e18,
                            3.40769107e18,
                            5.10140302e18,
                            7.90398909e18,
                            9.01286322e18,
                            7.31915127e18,
                            4.51656520e18,
                            3.40769107e18,
                            5.10140302e18,
                        ],
                        [
                            9.77865050e18,
                            1.51569083e19,
                            1.41790432e19,
                            7.82292040e18,
                            2.44466262e18,
                            3.42252767e18,
                            9.77865050e18,
                            1.51569083e19,
                            1.41790432e19,
                            7.82292040e18,
                            2.44466262e18,
                            3.42252767e18,
                        ],
                        [
                            9.03313039e18,
                            1.68560508e19,
                            1.68560508e19,
                            9.03313039e18,
                            1.21021000e18,
                            1.21021000e18,
                            9.03313039e18,
                            1.68560508e19,
                            1.68560508e19,
                            9.03313039e18,
                            1.21021000e18,
                            1.21021000e18,
                        ],
                        [
                            9.77865050e18,
                            1.51569083e19,
                            1.41790432e19,
                            7.82292040e18,
                            2.44466262e18,
                            3.42252767e18,
                            9.77865050e18,
                            1.51569083e19,
                            1.41790432e19,
                            7.82292040e18,
                            2.44466262e18,
                            3.42252767e18,
                        ],
                        [
                            7.90398909e18,
                            9.01286322e18,
                            7.31915127e18,
                            4.51656520e18,
                            3.40769107e18,
                            5.10140302e18,
                            7.90398909e18,
                            9.01286322e18,
                            7.31915127e18,
                            4.51656520e18,
                            3.40769107e18,
                            5.10140302e18,
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
                            0.00000000e00,
                            0.00000000e00,
                        ],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ),
    ],
)
def test_budget(
    time,
    level,
    longitude,
    latitude,
    sub_domain_longitude,
    sub_domain_latitude,
    preserve_horizontal,
    preserve_vertical,
    u_velocity_func,
    v_velocity_func,
    w_velocity_func,
    temperature_func,
    humidity_func,
    geopotential_func,
    surface_pressure_func,
    expected,
):
    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = 1.0e5 * np.ones((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity_func(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity_func(lat3d, lon3d, lev3d)
    w[0, :, :, :] = w_velocity_func(lat3d, lon3d, lev3d)
    t[0, :, :, :] = temperature_func(lat3d, lon3d, lev3d)
    q[0, :, :, :] = humidity_func(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential_func(lat3d, lon3d, lev3d)
    sp[0, :, :] = surface_pressure_func(lat2d, lon2d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    if sub_domain_longitude is not None:
        sub_longitude = longitude[
            (longitude > sub_domain_longitude[0] - 1.0e-6) & (longitude < sub_domain_longitude[1] + 1.0e-6)
        ]
        ds = ds.sel(longitude=sub_longitude)

    if sub_domain_latitude is not None:
        sub_latitude = latitude[
            (latitude > sub_domain_latitude[0] - 1.0e-6) & (latitude < sub_domain_latitude[1] + 1.0e-6)
        ]
        ds = ds.sel(latitude=sub_latitude)

    # test with some alternative thermodynamic constants
    if preserve_horizontal:
        _constants = PlanetConstants(C_PD=1006.0, C_PV=1872.0)

        E = energy_components_lat_lon(
            ds,
            preserve_horizontal=preserve_horizontal,
            preserve_vertical=preserve_vertical,
            constants=_constants,
        )
    else:
        E = energy_components_lat_lon(
            ds,
            preserve_horizontal=preserve_horizontal,
            preserve_vertical=preserve_vertical,
        )
    if not preserve_horizontal and not preserve_vertical:
        _E = xr.DataArray(
            [
                E["Internal"].as_numpy(),
                E["Latent"].as_numpy(),
                E["Potential"].as_numpy(),
                E["HorizontalKinetic"].as_numpy(),
                E["VerticalKinetic"].as_numpy(),
            ]
        )
    elif preserve_horizontal:
        _E = xr.DataArray(
            [
                E["Internal"].as_numpy()[0, :, :],
                E["Latent"].as_numpy()[0, :, :],
                E["Potential"].as_numpy()[0, :, :],
                E["HorizontalKinetic"].as_numpy()[0, :, :],
                E["VerticalKinetic"].as_numpy()[0, :, :],
            ]
        )
    else:
        _E = xr.DataArray(
            [
                E["Internal"].as_numpy()[0],
                E["Latent"].as_numpy()[0],
                np.zeros(E["Internal"].as_numpy().shape[1:]),
                E["HorizontalKinetic"].as_numpy()[0],
                E["VerticalKinetic"].as_numpy()[0],
            ]
        )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)


# test that the energy budget computation is compatible with Dask
def test_budgets_dask():
    if dask == "Unavailable":
        pytest.skip("Dask unavailable, could not run test")

    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(0.0, 360.0, 6)
    latitude = np.linspace(-90.0, 90.0, 31, endpoint=True)
    expected = xr.DataArray([[2.505574e25], [6.481852e17], [1.285482e20], [4.824884e20], [0.000000e00]])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = 1.0e5 * np.ones((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity(lat3d, lon3d, lev3d)
    w[0, :, :, :] = w_velocity(lat3d, lon3d, lev3d)
    t[0, :, :, :] = temperature(lat3d, lon3d, lev3d)
    q[0, :, :, :] = humidity(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential(lat3d, lon3d, lev3d)
    sp[0, :, :] = surface_pressure(lat2d, lon2d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    E = energy_components_lat_lon(ds.chunk())
    assert isinstance(E["Internal"].data, dask.array.Array)
    assert isinstance(E["Latent"].data, dask.array.Array)
    assert isinstance(E["Potential"].data, dask.array.Array)
    assert isinstance(E["HorizontalKinetic"].data, dask.array.Array)
    assert isinstance(E["VerticalKinetic"].data, dask.array.Array)
    E = E.compute()
    assert isinstance(E["Internal"].data, (np.ndarray, np.generic))
    assert isinstance(E["Latent"].data, (np.ndarray, np.generic))
    assert isinstance(E["Potential"].data, (np.ndarray, np.generic))
    assert isinstance(E["HorizontalKinetic"].data, (np.ndarray, np.generic))
    assert isinstance(E["VerticalKinetic"].data, (np.ndarray, np.generic))
    _E = xr.DataArray(
        [
            E["Internal"].as_numpy(),
            E["Latent"].as_numpy(),
            E["Potential"].as_numpy(),
            E["HorizontalKinetic"].as_numpy(),
            E["VerticalKinetic"].as_numpy(),
        ]
    )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)


# test for internal and latent energy with multipltle phases
def test_budgets_multiphase_moisture():
    if dask == "Unavailable":
        pytest.skip("Dask unavailable, could not run test")

    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(0.0, 360.0, 6)
    latitude = np.linspace(-90.0, 90.0, 31, endpoint=True)
    expected = xr.DataArray([[2.505575e25], [7.518202e17], [1.285482e20], [4.824884e20], [0.000000e00]])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    ql = np.zeros((nt, nlev, nlat, nlon))
    qi = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = 1.0e5 * np.ones((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity(lat3d, lon3d, lev3d)
    w[0, :, :, :] = w_velocity(lat3d, lon3d, lev3d)
    t[0, :, :, :] = temperature(lat3d, lon3d, lev3d)
    q[0, :, :, :] = humidity(lat3d, lon3d, lev3d)
    ql[0, :, :, :] = 0.2 * humidity(lat3d, lon3d, lev3d)
    qi[0, :, :, :] = 0.1 * humidity(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential(lat3d, lon3d, lev3d)
    sp[0, :, :] = surface_pressure(lat2d, lon2d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "ql": (["time", "level", "latitude", "longitude"], ql),
            "qi": (["time", "level", "latitude", "longitude"], qi),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    E = energy_components_lat_lon(ds, liquid_mass_fraction_name="ql", ice_mass_fraction_name="qi")
    _E = xr.DataArray(
        [
            E["Internal"].as_numpy(),
            E["Latent"].as_numpy(),
            E["Potential"].as_numpy(),
            E["HorizontalKinetic"].as_numpy(),
            E["VerticalKinetic"].as_numpy(),
        ]
    )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)


def test_budgets_multiphase_moisture_vertical():
    if dask == "Unavailable":
        pytest.skip("Dask unavailable, could not run test")

    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(0.0, 360.0, 6)
    latitude = np.linspace(-90.0, 90.0, 31, endpoint=True)
    expected = xr.DataArray(
        [
            [1.255927e23, 7.535562e23, 1.569909e24, 3.516596e24, 6.782006e24, 8.540304e24, 3.767781e24],
            [3.768522e15, 2.261113e16, 4.710653e16, 1.055186e17, 2.035002e17, 2.562595e17, 1.130557e17],
            [0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00],
            [7.059522e14, 1.143643e17, 1.103050e18, 1.012053e19, 6.587382e19, 2.358473e20, 1.694285e20],
            [0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00],
        ]
    )

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    ql = np.zeros((nt, nlev, nlat, nlon))
    qi = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = 1.0e5 * np.ones((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity(lat3d, lon3d, lev3d)
    w[0, :, :, :] = w_velocity(lat3d, lon3d, lev3d)
    t[0, :, :, :] = temperature(lat3d, lon3d, lev3d)
    q[0, :, :, :] = humidity(lat3d, lon3d, lev3d)
    ql[0, :, :, :] = 0.2 * humidity(lat3d, lon3d, lev3d)
    qi[0, :, :, :] = 0.1 * humidity(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential(lat3d, lon3d, lev3d)
    sp[0, :, :] = surface_pressure(lat2d, lon2d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "ql": (["time", "level", "latitude", "longitude"], ql),
            "qi": (["time", "level", "latitude", "longitude"], qi),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    E = energy_components_lat_lon(
        ds, preserve_vertical=True, liquid_mass_fraction_name="ql", ice_mass_fraction_name="qi"
    )
    _E = xr.DataArray(
        [
            E["Internal"].as_numpy()[0],
            E["Latent"].as_numpy()[0],
            np.zeros(E["Internal"].as_numpy().shape[1:]),
            E["HorizontalKinetic"].as_numpy()[0],
            E["VerticalKinetic"].as_numpy()[0],
        ]
    )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)


def surface_geopotential(phi, theta):
    w = vorticity(phi, theta, 0.25 * np.pi)
    zs = 10.0 + w * w
    return zs


# test the energy exchanges against a previously computed solution
@pytest.mark.parametrize(
    (
        "time",
        "level",
        "longitude",
        "latitude",
        "u_velocity_func",
        "v_velocity_func",
        "geopotential_func",
        "surface_geopotential_func",
        "reduce_time",
        "preserve_horizontal",
        "preserve_vertical",
        "expected",
    ),
    [
        # global domain over single time period
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            u_velocity,
            v_velocity,
            geopotential,
            surface_geopotential,
            False,
            False,
            False,
            xr.DataArray([[-4.147951e15], [4.725676e15], [4.404260e15], [-4.982006e15]]),
        ),
        # global domain over single time period on shifted longitudinal domain
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(-180, 180.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            u_velocity,
            v_velocity,
            geopotential,
            surface_geopotential,
            False,
            False,
            False,
            xr.DataArray([[-4.114057e15], [4.730493e15], [4.370366e15], [-4.986823e15]]),
        ),
        # preserve the vertical dimension for horizontal energy exchanges in each
        # vertical column
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            u_velocity,
            v_velocity,
            geopotential,
            surface_geopotential,
            False,
            False,
            True,
            xr.DataArray(
                [
                    [-1.625466e12, -2.924816e13, -1.014222e14, -3.616016e14, -1.028833e15, -1.754881e15, -8.703392e14],
                    [1.838696e12, 3.308629e13, 1.147490e14, 4.093649e14, 1.167004e15, 2.001365e15, 9.982684e14],
                    [1.625488e12, 2.925878e13, 1.015930e14, 3.641093e14, 1.053316e15, 1.879064e15, 9.752928e14],
                    [-1.838718e12, -3.309692e13, -1.149199e14, -4.118728e14, -1.191489e15, -2.125558e15, -1.103231e15],
                ]
            ),
        ),
        # integrate energy exchanges in time over several time periods
        (
            pd.date_range("2025-01-01", periods=3),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.arange(0.0, 360.0, 6),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            u_velocity,
            v_velocity,
            geopotential,
            surface_geopotential,
            True,
            False,
            False,
            xr.DataArray(3 * 24 * 60 * 60 * np.array([-4.147951e15, 4.725676e15, 4.404260e15, -4.982006e15])),
        ),
    ],
)
def test_exchanges(
    time,
    level,
    longitude,
    latitude,
    u_velocity_func,
    v_velocity_func,
    geopotential_func,
    surface_geopotential_func,
    reduce_time,
    preserve_horizontal,
    preserve_vertical,
    expected,
):
    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = np.zeros((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity_func(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity_func(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential_func(lat3d, lon3d, lev3d)
    zs[:, :] = surface_geopotential_func(lat2d, lon2d)
    if reduce_time is True:
        for ii in np.arange(nt - 1):
            u[ii + 1, :, :, :] = u_velocity_func(lat3d, lon3d, lev3d)
            v[ii + 1, :, :, :] = v_velocity_func(lat3d, lon3d, lev3d)
            z[ii + 1, :, :, :] = geopotential_func(lat3d, lon3d, lev3d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    E = energy_exchanges_lat_lon(
        ds, reduce_time=reduce_time, preserve_horizontal=preserve_horizontal, preserve_vertical=preserve_vertical
    )
    if not preserve_horizontal and not preserve_vertical:
        _E = xr.DataArray(
            [
                E["KineticToInternal"].as_numpy(),
                E["InternalToKinetic"].as_numpy(),
                E["KineticToPotential"].as_numpy(),
                E["PotentialToKinetic"].as_numpy(),
            ]
        )
    else:
        _E = xr.DataArray(
            [
                E["KineticToInternal"].as_numpy()[0, :],
                E["InternalToKinetic"].as_numpy()[0, :],
                E["KineticToPotential"].as_numpy()[0, :],
                E["PotentialToKinetic"].as_numpy()[0, :],
            ]
        )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)


# test that the energy exchanges computation is compatible with Dask
def test_exchanges_dask():
    if dask == "Unavailable":
        pytest.skip("Dask unavailable, could not run test")

    time = pd.date_range("2025-01-01", periods=1)
    level = np.array([50, 150, 250, 400, 600, 850, 1000])
    longitude = np.arange(0.0, 360.0, 6)
    latitude = np.linspace(-90.0, 90.0, 31, endpoint=True)
    expected = xr.DataArray([[-4.147951e15], [4.725676e15], [4.404260e15], [-4.982006e15]])

    nt = len(time)
    nlev = len(level)
    nlat = len(latitude)
    nlon = len(longitude)

    u = np.zeros((nt, nlev, nlat, nlon))
    v = np.zeros((nt, nlev, nlat, nlon))
    w = np.zeros((nt, nlev, nlat, nlon))
    t = np.zeros((nt, nlev, nlat, nlon))
    q = np.zeros((nt, nlev, nlat, nlon))
    z = np.zeros((nt, nlev, nlat, nlon))
    sp = np.zeros((nt, nlat, nlon))
    zs = np.zeros((nlat, nlon))

    lon2d, lat2d = np.meshgrid(longitude, latitude)
    lev3d, lat3d, lon3d = np.meshgrid(level, latitude, longitude, indexing="ij")

    u[0, :, :, :] = u_velocity(lat3d, lon3d, lev3d)
    v[0, :, :, :] = v_velocity(lat3d, lon3d, lev3d)
    z[0, :, :, :] = geopotential(lat3d, lon3d, lev3d)
    zs[:, :] = surface_geopotential(lat2d, lon2d)

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "level", "latitude", "longitude"], u),
            "v": (["time", "level", "latitude", "longitude"], v),
            "w": (["time", "level", "latitude", "longitude"], w),
            "t": (["time", "level", "latitude", "longitude"], t),
            "q": (["time", "level", "latitude", "longitude"], q),
            "z": (["time", "level", "latitude", "longitude"], z),
            "sp": (["time", "latitude", "longitude"], sp),
            "zs": (["latitude", "longitude"], zs),
        },
        coords={
            "time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    E = energy_exchanges_lat_lon(ds.chunk())
    assert isinstance(E["KineticToInternal"].data, dask.array.Array)
    assert isinstance(E["InternalToKinetic"].data, dask.array.Array)
    assert isinstance(E["KineticToPotential"].data, dask.array.Array)
    assert isinstance(E["PotentialToKinetic"].data, dask.array.Array)
    E = E.compute()
    assert isinstance(E["KineticToInternal"].data, (np.ndarray, np.generic))
    assert isinstance(E["InternalToKinetic"].data, (np.ndarray, np.generic))
    assert isinstance(E["KineticToPotential"].data, (np.ndarray, np.generic))
    assert isinstance(E["PotentialToKinetic"].data, (np.ndarray, np.generic))
    _E = xr.DataArray(
        [
            E["KineticToInternal"].as_numpy(),
            E["InternalToKinetic"].as_numpy(),
            E["KineticToPotential"].as_numpy(),
            E["PotentialToKinetic"].as_numpy(),
        ]
    )
    xr.testing.assert_allclose(_E, expected, atol=1.0e-2)
