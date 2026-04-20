"""
Contains unit tests for scores.budgets.energetics_impl
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

from scores.budgets.budgets_utils import (
    integrate_energy_exchange,
    integrate_horizontal,
    integration_weights,
    trig_fields,
)
from scores.budgets.energetics_impl import energy_components

# Williamson 5 test case initial condition stream function
# Reference:
#   Williamson, et al. (1992) JCP vol. 102 pp 211--224, eqn. (92)
#
# phi: azimuthal angle
# theta: polar angle (from the equator)

rad_earth = 6371220.0
area_earth = 4.0 * np.pi * rad_earth * rad_earth
u_0 = 2.0 * np.pi * rad_earth / (12.0 * 24.0 * 60.0 * 60.0)


def stream_function(phi, theta, alpha):
    _phi = np.deg2rad(phi)
    _theta = np.deg2rad(theta)

    return -rad_earth * u_0 * (np.sin(_theta) * np.cos(alpha) - np.cos(_phi) * np.cos(_theta) * np.sin(alpha))


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
        # Global integral
        (
            vorticity,
            0.25 * np.pi,
            10.0,
            np.array([30, 60], dtype=np.int64),
            4,
            np.array([None]),
            np.array([None]),
            area_earth * 10.0,
            4.0,
            1.0e-3,
        ),
        # Northern hemisphere
        (
            vorticity,
            0.5 * np.pi,
            10.0,
            np.array([30, 60], dtype=np.int64),
            4,
            np.array([None]),
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
            np.array([None]),
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
            rad_earth * rad_earth * np.pi * (135.0 - 45.0) / 180.0 * (np.sin(np.pi / 3.0) + np.sin(np.pi / 3.0)) * 10.0,
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
        latitude = np.linspace(-90.0, +90.0, nlat, endpoint=False)

        dlon, dlat, lon, lat = integration_weights(longitude, latitude, sub_domain_longitude, sub_domain_latitude)
        nlon = len(lon)
        nlat = len(lat)

        psi = np.zeros((nlat, nlon))
        for ii in np.arange(nlat):
            for jj in np.arange(nlon):
                psi[ii, jj] = field(lon[jj], lat[ii], alpha) + offset

        int_psi = integrate_horizontal(psi, dlon, dlat)

        error_at_res[res] = (analytic_solution - int_psi) / analytic_solution

        print(
            str(res)
            + ":\t{:.4e}".format(analytic_solution)
            + "\t{:.4e}".format(int_psi)
            + "\t{:.4e}".format(error_at_res[res])
        )

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
            np.array([None]),
            np.array([None]),
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
        latitude = np.linspace(-90.0, +90.0, nlat)

        dlon, dlat, lon, lat = integration_weights(longitude, latitude, sub_domain_longitude, sub_domain_latitude)
        nlon = len(lon)
        nlat = len(lat)
        cos_theta, sin_theta, cos_theta_inv = trig_fields(lon, lat)

        psi = np.zeros((nlat, nlon))
        dPsiDx = np.zeros((nlat, nlon))
        dPsiDy = np.zeros((nlat, nlon))
        lap_psi = np.zeros((nlat, nlon))
        for ii in np.arange(nlat):
            for jj in np.arange(nlon):
                psi[ii, jj] = field(lon[jj], lat[ii], alpha)
                dPsiDx[ii, jj] = zonal_gradient(lon[jj], lat[ii], alpha)
                dPsiDy[ii, jj] = meridional_gradient(lon[jj], lat[ii], alpha)
                lap_psi[ii, jj] = div_vec(lon[jj], lat[ii], alpha)

        grad_f_dot_u, f_div_u = integrate_energy_exchange(
            psi, dPsiDx, dPsiDy, lon, lat, dlon, dlat, cos_theta, sin_theta, cos_theta_inv
        )

        err1 = grad_f_dot_u - (dPsiDx * dPsiDx + dPsiDy * dPsiDy)
        err2 = f_div_u - psi * lap_psi
        error_at_res_grad_f_dot_u[res] = integrate_horizontal(err1, dlon, dlat)
        error_at_res_f_div_u[res] = integrate_horizontal(err2, dlon, dlat)
        balance_error_at_res[res] = np.abs(error_at_res_grad_f_dot_u[res] + error_at_res_f_div_u[res]) / np.abs(
            error_at_res_grad_f_dot_u[res]
        )

        print(
            str(res)
            + "\t{:.4e}".format(error_at_res_grad_f_dot_u[res])
            + "\t{:.4e}".format(error_at_res_f_div_u[res])
            + "\t{:.4e}".format(balance_error_at_res[res])
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
        "latitude",
        "longitude",
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
        (
            pd.date_range("2025-01-01", periods=1),
            np.array([50, 150, 250, 400, 600, 850, 1000]),
            np.linspace(-90.0, 90.0, 31, endpoint=True),
            np.arange(0.0, 360.0, 6),
            u_velocity,
            v_velocity,
            w_velocity,
            temperature,
            humidity,
            geopotential,
            surface_pressure,
            xr.DataArray([[2.510566e25], [6.481852e17], [1.285482e20], [4.824884e20], [0.0]]),
        ),
    ],
)
def test_budget(
    time,
    level,
    latitude,
    longitude,
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
    for ii in np.arange(nlev):
        for jj in np.arange(nlat):
            for kk in np.arange(nlon):
                u[0, ii, jj, kk] = u_velocity(latitude[jj], longitude[kk], level[ii])
                v[0, ii, jj, kk] = v_velocity(latitude[jj], longitude[kk], level[ii])
                w[0, ii, jj, kk] = w_velocity(latitude[jj], longitude[kk], level[ii])
                t[0, ii, jj, kk] = temperature(latitude[jj], longitude[kk], level[ii])
                q[0, ii, jj, kk] = humidity(latitude[jj], longitude[kk], level[ii])
                z[0, ii, jj, kk] = geopotential(latitude[jj], longitude[kk], level[ii])
    for jj in np.arange(nlat):
        for kk in np.arange(nlon):
            sp[0, jj, kk] = surface_pressure(latitude[jj], longitude[kk])

    field_names = ["u", "v", "w", "t", "q", "z", "sp", "zs"]
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

    E = energy_components(ds, field_names)
    xr.testing.assert_allclose(E, expected, atol=1.0e-2)
