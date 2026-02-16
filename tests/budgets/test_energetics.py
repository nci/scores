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
# lambda: azimuthal angle
# theta: polar angle (from the equator)
rad_earth = 6371220.0
area_earth = 4.0 * np.pi * rad_earth * rad_earth
u_0 = 2.0 * np.pi * rad_earth / (12.0 * 24.0 * 60.0 * 60.0)
def stream_function(lambda, theta, alpha):
    _lambda = np.deg2rad(lambda)
    _theta = np.deg2rad(theta)

    return -rad_earth * u_0 * (np.sin(_theta) * np.cos(alpha) - np.cos(_lambda) * np.cos(_theta) * np.sin(alpha))

def zero_function(lambda, theta, alpha):
    return 0.0

# derived from the stream function, \phi as:
#   -1/r d\phi/d\theta
# see eqn. (90) of Williamson et al. (1992)
def zonal_velocity(lambda, theta, alpha):
    _lambda = np.deg2rad(lambda)
    _theta = np.deg2rad(theta)

    return u_0 * (np.cos(_theta) * np.cos(alpha) + np.cos(_lambda) * np.sin(_theta) * np.sin(alpha))

# derived from the stream function, \phi as:
#   1/(r cos(theta)) d\phi/d\lambda
# see eqn. (91) of Williamson et al. (1992)
def meridional_velocity(lambda, theta, alpha):
    _lambda = np.deg2rad(lambda)
    _theta = np.deg2rad(theta)

    return -u_0 * np.sin(_lambda) * np.sin(alpha)

# init test for the spherical integration via convergence of errors with resolution
@pytest.mark.parametrize(
    ("field", "alpha", "offset", "low_resolution", "num_resolutions", "sub_domain_latitude", "sub_domain_longitude", "analytic_solution"),
    [
        # Global integral
        (
            stream_function,
            0.25*np.pi + 0.05
            10.0,
            np.array([9,18], dtype=np.int64),
            3,
            np.array([None]),
            np.array([None]),
            area_earth * 10.0,
        ),
        # Northern hemisphere
        (
            stream_function,
            0.5*np.pi,
            10.0,
            np.array([9,18], dtype=np.int64),
            3,
            np.array([None]),
            np.array([0.0, 90.0]),
            0.5 * area_earth * 10.0,
        ),
        # Longitudinal sub-domain
        (
            stream_function,
            0.0,
            10.0,
            np.array([9,18], dtype=np.int64),
            3,
            np.array([-90.0, 0.0]),
            np.array([None]),
            0.25 * area_earth * 10.0,
        ),
        # Sub-domain in latitude and longitude
        (
            zero_function,
            0.0,
            10.0,
            np.array([9,18], dtype=np.int64),
            3,
            np.array([-45.0, +45.0]),
            np.array([-90.0, 0.0]),
            0.125 * area_earth * 10.0,
        ),
    ],
)

def test_budgets_integral(field, alpha, offset, low_resolution, num_resolutions, sub_domain_longitude, sub_domain_latitude, analytic_solution):

    error_at_res = np.zeros(num_resolutions)

    for res in np.arange(num_resolutions):
        nlon = low_resolution[1] * np.power(2, res)
        nlat = low_resolution[0] * np.power(2, res)

        longitude = np.linspace(-180.0, +180.0, nlon)
        latitude = np.linspace(-90.0, +90.0, nlat)

        dlon, dlat, lon, lat = integration_weights(longitude, latitude, sub_domain_longitude, sub_domain_latitude)
        cos_theta, sin_theta, cos_theta_inv = trig_fields(lon, lat)

        psi = np.zeros((nlat, nlon))
        for ii in np.arange(nlat):
            for jj in np.arange(nlon):
                psi[ii,jj] = field(lon[jj], lat[ii], alpha) + offset

        int_psi = integrate_horizontal(psi, dlon, dlat)

        error_at_res[res] = (analytic_solution - int_psi)/analytic_solution

    convergence = np.zeros(num_resolutions-1)
    for res in np.arange(num_resolutions-1):
        convergence[res] = error_at_res[res] / error_at_res[res+1]

    xr.testing.assert_allclose(xr.DataArray(convergence), xr.DataArray(4.0*np.ones(num_resolutions-1)), atol=1.0e-3)

# unit test for the energy exchanges
@pytest.mark.parametrize(
    ("field", "zonal_gradient", "meridional_gradient", "longitude", "latitude", "sub_domain_latitude", "sub_domain_longitude"),
    [
        # Global integral
        (
            stream_function,
            zonal_velocity,
            meridional_velocity,
            np.arange(-180.0,+180.0,36),
            np.arange(-90.0,+90.0,18),
            None,
            None,
        ),
    ],
)

def test_budgets_gradient(field, zonal_gradient, meridional_gradient, longitude, latitude, sub_domain_longitude, sub_domain_latitude):
    dlon, dlat, lon, lat = integration_weights(longitude, latitude, sub_domain_longitude, sub_domain_latitude)
    cos_theta, sin_theta, cos_theta_inv = trig_fields(lon, lat)

    nlon = len(lon)
    nlat = len(lat)

    psi = np.zeros((nlat, nlon))
    dpsidlambda_analytic = np.zeros((nlat, nlon))
    dpsidtheta_analytic = np.zeros((nlat, nlon))
    for ii in np.arange(nlat):
        for jj in np.arange(nlon):
            psi[ii,jj] = field(lon[jj], lat[ii])
            dpsidlambda_analytic[ii,jj] = meridional_velocity(lon[jj], lat[ii])
            dpsidtheta_analytic[ii,jj] = -1.0*zonal_velocity(lon[jj], lat[ii])

    return
