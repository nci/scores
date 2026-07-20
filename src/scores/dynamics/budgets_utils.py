"""
Common backend functionality required for the energy budget diagnosics
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

LON_MIN = 0.0  # minimum permissible longitude, degrees
LON_MAX = +360.0  # maximum permissible longitude, degrees
LAT_MIN = -90.0  # minimum permissible latitude, degrees
LAT_MAX = +90.0  # maximum permissible latitude, degrees


@dataclass
class PlanetConstants:
    # physical constants
    RAD_EARTH: float = 6371220.0  # radius of the earth, m
    GRAVITY: float = 9.80665  # gravitational acceleration of the earth, m/s^2
    C_PD: float = 1004.0  # specific heat of dry air at constant pressure, J/kg/K
    C_PV: float = 1885.0  # specific heat of vapour water at constant pressure, J/kg/K
    C_L: float = 4186.0  # specific heat of liquid water at constant pressure, J/kg/K
    C_I: float = 2106.0  # specific heat of ice water at constant pressure, J/kg/K
    T_0: float = 0.0  # reference temperature, K
    R_V: float = 461.0  # gas constant for vapour, J/kg/K
    L_V: float = 2.5008e6  # specific latent heat of vaporisation, J/kg
    L_S: float = 2.834e6  # specific latent heat of vaporisation, J/kg

    def __init__(
        self,
        RAD_EARTH: float = 6371220.0,
        GRAVITY: float = 9.80665,
        C_PD: float = 1004.0,
        C_PV: float = 1885.0,
        C_L: float = 4186.0,
        C_I: float = 2106.0,
        T_0: float = 0.0,
        R_V: float = 461.0,
        L_V: float = 2.5008e6,
        L_S: float = 2.834e6,
    ):
        self.RAD_EARTH = RAD_EARTH
        self.GRAVITY = GRAVITY
        self.C_PD = C_PD
        self.C_PV = C_PV
        self.C_L = C_L
        self.C_I = C_I
        self.T_0 = T_0
        self.R_V = R_V
        self.L_V = L_V
        self.L_S = L_S
        self.L_F = self.L_S - self.L_V  # specific latent heat of freezing, J/kg

    # conversion from degrees to meters, m
    def meters_per_degree(self):
        return 2.0 * np.pi * self.RAD_EARTH / 360.0


standard_constants = PlanetConstants()


def _integration_weights(
    longitude: np.ndarray,
    latitude: np.ndarray,
    longitude_name: str,
    latitude_name: str,
    constants: PlanetConstants,
):
    """
    Integration weights for a two dimensional latitude-longitude field on
    the surface of the sphere

    Args:
        longitude: zonal coordinate
        latitude: meridional coordinate
        longitude_name: string giving the textual name of the longitude coordinate
        latitude_name: string giving the textual name of the latitude coordinate
        constants: planetary constants used for integration

    Returns:
        two xarray dataarrays containing the integration weights at the domain latitudes and longitudes
    """

    dlon = np.zeros(len(longitude))
    dlon[1:-1] = 0.5 * (longitude[2:] - longitude[:-2])
    dlon[0] = longitude[1] - longitude[0]
    dlon[-1] = longitude[-1] - longitude[-2]
    dlon[:] = constants.meters_per_degree() * dlon[:]
    error_msg = ValueError("-ve value detected in longitudinal weights")
    if (dlon < 0.0).any():
        raise error_msg  # pragma: no cover

    dlat = np.zeros(len(latitude))
    for ii in np.arange(len(latitude) - 2) + 1:
        dlat[ii] = 0.5 * (latitude[ii + 1] - latitude[ii - 1])
        dlat[ii] = dlat[ii] * np.cos(np.deg2rad(latitude[ii]))

    if latitude[0] > -90.0 + 1.0e-4:
        dlat[0] = np.abs(latitude[1] - latitude[0]) * np.cos(np.deg2rad(latitude[0]))
    if latitude[-1] < +90.0 - 1.0e-4:
        dlat[-1] = np.abs(latitude[-1] - latitude[-2]) * np.cos(np.deg2rad(latitude[-1]))

    dlat[:] = constants.meters_per_degree() * dlat[:]
    error_msg = ValueError("-ve value detected in latitudinal weights")
    if (dlat < 0.0).any():
        raise error_msg  # pragma: no cover

    return xr.DataArray(dlon, dims=longitude_name, coords={longitude_name: longitude}), xr.DataArray(
        dlat, dims=latitude_name, coords={latitude_name: latitude}
    )


def _integrate_horizontal(data: xr.DataArray, dlon, dlat, preserve_horizontal):
    """
    Integrate an xarray dataarray in the horizontal (latitude, longitude) dimensions

    Args:
        data: physical field to integrate in the horizontal dimensions
        dlon: xarray dataarray containing the integration weights in the zonal direction
        dlat: xarray dataarray containing the integration weights in the meridional direction
        preserve_horizontal: apply area weighting to the energy components in the horizontal dimensions (latitude,
            longitude), but do not sum these area weighted components in the horizontal (optional, default is false).

    Returns:
        int_tot: xarray dataarray containing a weighted integral of the input field. May be either a global integal
            (sum) over the domain, or a 2D field weighted by the integration weights
    """
    if preserve_horizontal:
        int_tot = data * dlon * dlat
    else:
        int_lon = data.dot(dlon)
        int_tot = int_lon.dot(dlat)

    return int_tot


def _trig_fields(longitude, latitude, longitude_name, latitude_name):
    """
    Compute metric terms for integration and differentiation on the sphere in latitude/longitude coorinds

    Args:
        longitude: zonal coordinate
        latitude: meridional coordinate
        longitude_name: string giving the textual name of the longitude coordinate
        latitude_name: string giving the textual name of the latitude coordinate

    Returns:
        xarray dataarrays for the cosine, sine and inverse cosine of the latitude as two dimensional fields
            over the domain
    """

    nlon = len(longitude)

    lat_rad = np.deg2rad(latitude).to_numpy()

    cos_theta_np = np.cos(lat_rad)[:, None] * np.ones((1, nlon))
    sin_theta_np = np.sin(lat_rad)[:, None] * np.ones((1, nlon))

    cos_theta = xr.DataArray(
        cos_theta_np,
        dims=[latitude_name, longitude_name],
        coords={latitude_name: latitude, longitude_name: longitude},
    )

    sin_theta = xr.DataArray(
        sin_theta_np,
        dims=[latitude_name, longitude_name],
        coords={latitude_name: latitude, longitude_name: longitude},
    )

    cos_theta_inv = 1.0 / cos_theta

    return cos_theta, sin_theta, cos_theta_inv


def _pressure_level_thickness(levels, constants: PlanetConstants):
    """
    Compute the physical thickness (in meters) of each (hydrostatic) pressure level for vertical integration

    Args:
        levels: hydrostatic pressure levels (the vertical dimension)
        constants: physical constants

    Returns:
        dp: the vertical integration weights
    """
    nl = len(levels)
    dp = np.zeros(nl)

    dp[1:-1] = 0.5 * (levels[2:] - levels[:-2])
    dp[0] = 0.5 * (levels[1] - levels[0])
    dp[-1] = 0.5 * (levels[-1] - levels[-2])

    # convert pressure level thickness from hPa to Pa and normalise by gravity to get \rho dz
    dp = 100.0 * dp / constants.GRAVITY

    return dp


def _integrate_energy_exchange(
    field_scalar,
    field_vector_x,
    field_vector_y,
    dlon,
    dlat,
    cos_theta,
    sin_theta,
    cos_theta_inv,
    longitude_name,
    latitude_name,
    constants: PlanetConstants,
    preserve_horizontal: bool = False,
):
    """
    Compute the two volumetric components of the divergence theorem on the sphere in latitude, longitude coordinates
    for scalar field (f) and two dimensional horizontal vector field (u,v).

    Williamson et. al., JCP (1992), eqns (3-4):

    lambda: longitude
    theta:  latitude (from the equator)
    grad f:  1/(r cos(theta)) d f/d lambda, 1/r df/d theta
    div(u):  1/(r cos(theta)) (d u/d lambda + d(v cos(theta))/d theta)

    Args:
        field_scaler: scalar field (f)
        field_vector_x: zonal component of the vector field (u)
        field_vector_y: meridional component of the vector field (v)
        dlon: integration weights in the zonal direction
        dlat: integration weights in the meridional direction
        cos_theta: two dimensional field for the cosine of the zonal and meridional coordinates
        sin_theta: two dimensional field for the sine of the zonal and meridional coordinates
        cos_theta_inv: two dimensional field for the inverse of the cosine of the zonal and meridional coordinates
        longitude_name: string giving the textual name of the zonal coordinate in the xarray DataArray fields
        latitude_name: string giving the textual name of the meridional coordinate in the xarray DataArray fields
        constants: planetary constants used for differentiation
        preserve_horizontal: apply area weighting to the energy components in the horizontal dimensions (latitude,
            longitude), but do not sum these area weighted components in the horizontal (optional, default is false).

    Returns:
        Xarray DataArrays for grad f . (u,v) and f div (u,v)
    """

    mpd_inv = 1.0 / constants.meters_per_degree()

    # grad f:  1/(r \cos(\theta)) df/d\lambda, 1/r df/d\theta
    dfdx = field_scalar.differentiate(longitude_name) * cos_theta_inv * mpd_inv
    dfdy = field_scalar.differentiate(latitude_name) * mpd_inv
    grad_f_dot_u = dfdx * field_vector_x + dfdy * field_vector_y
    int_grad_f_dot_u = _integrate_horizontal(grad_f_dot_u, dlon, dlat, preserve_horizontal)

    # div(u):  1/(r \cos(\theta)) (du/d\lambda + d(v\cos(\theta))/d\theta)
    dudx = field_vector_x.differentiate(longitude_name) * mpd_inv
    dvdy = (
        field_vector_y.differentiate(latitude_name) * cos_theta * mpd_inv
        - sin_theta * field_vector_y / constants.RAD_EARTH
    )
    div_u = cos_theta_inv * (dudx + dvdy)
    f_div_u = field_scalar * div_u
    int_f_div_u = _integrate_horizontal(f_div_u, dlon, dlat, preserve_horizontal)

    return int_grad_f_dot_u, int_f_div_u
