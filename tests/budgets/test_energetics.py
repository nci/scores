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

from scores.budgets.budgets_utils import *

# Williamson 5 test case initial condition stream function
# Refereince: 
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

    #return 2.0 * u_0 / rad_earth * (-np.cos(_phi) * np.cos(_theta) * np.sin(alpha) + np.sin(_theta) * np.cos(alpha))
    return 2.0 * u_0 * (-np.cos(_phi) * np.cos(_theta) * np.sin(alpha) + np.sin(_theta) * np.cos(alpha))

# init test for the spherical integration via convergence of errors with resolution
@pytest.mark.parametrize(
    ("field", "alpha", "offset", "low_resolution", "num_resolutions", "sub_domain_longitude", "sub_domain_latitude", \
            "analytic_solution", "convergence_rate", "tolerance"),
    [
        # Global integral
        (
            vorticity,
            0.25*np.pi,
            10.0,
            np.array([30,60], dtype=np.int64),
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
            0.5*np.pi,
            10.0,
            np.array([30,60], dtype=np.int64),
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
            np.array([30,60], dtype=np.int64),
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
            0.5*np.pi,
            10.0,
            np.array([60,120], dtype=np.int64),
            4,
            np.array([45.0, 135.0]),
            np.array([-60.0, +60.0]),
            rad_earth * rad_earth * np.pi * (135.0 - 45.0) / 180.0 * (np.sin(np.pi/3.0) + np.sin(np.pi/3.0)) * 10.0,
            2.0,
            1.0e-2,
        ),
    ],
)

def test_budgets_integral(field, alpha, offset, low_resolution, num_resolutions, sub_domain_longitude, sub_domain_latitude, \
        analytic_solution, convergence_rate, tolerance):

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
                psi[ii,jj] = field(lon[jj], lat[ii], alpha) + offset

        int_psi = integrate_horizontal(psi, dlon, dlat)

        error_at_res[res] = (analytic_solution - int_psi)/analytic_solution

        print(str(res) + ':\t{:.4e}'.format(analytic_solution) + '\t{:.4e}'.format(int_psi) + '\t{:.4e}'.format(error_at_res[res]))

    convergence = np.zeros(num_resolutions-1)
    for res in np.arange(num_resolutions-1):
        convergence[res] = error_at_res[res] / error_at_res[res+1]

    # for second order accuracy, we anticipate the integration error should decrease by a factor of 4 for a
    # doubling of the spatial resolution (to within some tolerance)
    xr.testing.assert_allclose(xr.DataArray(convergence), xr.DataArray(convergence_rate*np.ones(num_resolutions-1)), atol=tolerance)

# unit test for the energy exchanges
@pytest.mark.parametrize(
    ("field", "zonal_gradient", "meridional_gradient", "div_vec", "alpha", "low_resolution", "num_resolutions", \
            "sub_domain_longitude", "sub_domain_latitude", "convergence_rate", "tolerance"),
    [
        # Global integral
        (
            stream_function,
            d_psi_d_phi,
            d_psi_d_theta,
            vorticity,
            0.25*np.pi,
            np.array([60,120], dtype=np.int64),
            3,
            np.array([None]),
            np.array([None]),
            2.0,
            2.0e-2,
        ),
    ],
)

def test_budgets_gradient(field, zonal_gradient, meridional_gradient, div_vec, alpha, low_resolution, num_resolutions, \
        sub_domain_longitude, sub_domain_latitude, convergence_rate, tolerance):

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
                psi[ii,jj] = field(lon[jj], lat[ii], alpha)
                dPsiDx[ii,jj] = zonal_gradient(lon[jj], lat[ii], alpha)
                dPsiDy[ii,jj] = meridional_gradient(lon[jj], lat[ii], alpha)
                lap_psi[ii,jj] = div_vec(lon[jj], lat[ii], alpha)

        grad_f_dot_u, f_div_u = integrate_energy_exchange(psi, dPsiDx, dPsiDy, lon, lat, dlon, dlat, \
                cos_theta, sin_theta, cos_theta_inv)

        err1 = grad_f_dot_u - (dPsiDx*dPsiDx + dPsiDy*dPsiDy)
        err2 = f_div_u - psi*lap_psi
        error_at_res_grad_f_dot_u[res] = integrate_horizontal(err1, dlon, dlat)
        error_at_res_f_div_u[res] = integrate_horizontal(err2, dlon, dlat)
        balance_error_at_res[res] = np.abs(error_at_res_grad_f_dot_u[res] + error_at_res_f_div_u[res])/np.abs(error_at_res_grad_f_dot_u[res])

        print(str(res) + '\t{:.4e}'.format(error_at_res_grad_f_dot_u[res]) + '\t{:.4e}'.format(error_at_res_f_div_u[res]) + \
                '\t{:.4e}'.format(balance_error_at_res[res]))

    convergence = np.zeros(num_resolutions-1)
    for res in np.arange(num_resolutions-1):
        convergence[res] = balance_error_at_res[res] / balance_error_at_res[res+1]

    xr.testing.assert_allclose(xr.DataArray(convergence), xr.DataArray(convergence_rate*np.ones(num_resolutions-1)), atol=tolerance)
    xr.testing.assert_allclose(xr.DataArray(balance_error_at_res), xr.DataArray(np.zeros(num_resolutions)), atol=1.0e-2)
