"""
Common backend functionality required for the energy budget diagnosics
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import xarray as xr

RAD_EARTH = 6371220.0  # radius of the earth, m
METERS_PER_DEGREE = 2.0 * np.pi * RAD_EARTH / 360.0  # conversion from degrees to meters, m
LON_MIN = 0.0  # minimum permissible longitude, degrees
LON_MAX = +360.0  # maximum permissible longitude, degrees
LAT_MIN = -90.0  # minimum permissible latitude, degrees
LAT_MAX = +90.0  # maximum permissible latitude, degrees
GRAVITY = 9.80665  # gravitational acceleration of the earth, m/s^2
C_P = 1006.0  # specific heat of dry air at constant pressure, J/kg/K
C_PV = 1872.0  # specific heat of water vapor at constant pressure, J/kg/K
L_V = 2.5008e6  # specific latent heat of vaporisation, J/kg

"""
Integration weights for a two dimensional latitude-longitude field on
the surface of the sphere
"""


def _integration_weights(
    longitude: np.ndarray,
    latitude: np.ndarray,
    dimension_names: dict,
    sub_domain_lon: np.ndarray | None = None,
    sub_domain_lat: np.ndarray | None = None,
):
    # Check the longitude sub domain is valid if specified
    if sub_domain_lon is not None:
        error_msg = ValueError(
            "sub-domain longitude outside valid range: "
            + f"{LON_MIN} <= minimum longitude < maximum longitude < {LON_MAX}"
        )
        if len(sub_domain_lon) != 2:
            raise error_msg  # pragma: no cover
        if (
            sub_domain_lon[1] <= sub_domain_lon[0]
            or sub_domain_lon[0] < LON_MIN
            or sub_domain_lon[0] > LON_MAX
            or sub_domain_lon[1] < LON_MIN
            or sub_domain_lon[1] > LON_MAX
        ):
            raise error_msg  # pragma: no cover

        cond_lon = (longitude >= sub_domain_lon[0]) & (longitude <= sub_domain_lon[1])
        longitude = longitude[cond_lon]

    dlon = np.zeros(len(longitude))
    dlon[1:-1] = 0.5 * (longitude[2:] - longitude[:-2])
    dlon[0] = longitude[1] - longitude[0]
    dlon[-1] = longitude[-1] - longitude[-2]
    dlon[:] = METERS_PER_DEGREE * dlon[:]

    # Check the latitude sub domain is valid if specified
    if sub_domain_lat is not None:
        error_msg = ValueError(
            "sub-domain latitude outside valid range: "
            + f"{LAT_MIN} <= minimum latitude < maximum latitude < {LAT_MAX}"
        )
        if len(sub_domain_lat) != 2:
            raise error_msg  # pragma: no cover
        if (
            sub_domain_lat[1] <= sub_domain_lat[0]
            or sub_domain_lat[0] < LAT_MIN
            or sub_domain_lat[0] > LAT_MAX
            or sub_domain_lat[1] < LAT_MIN
            or sub_domain_lat[1] > LAT_MAX
        ):
            raise error_msg  # pragma: no cover

        cond_lat = (latitude >= sub_domain_lat[0]) & (latitude <= sub_domain_lat[1])
        latitude = latitude[cond_lat]

    dlat = np.zeros(len(latitude))
    for ii in np.arange(len(latitude) - 2) + 1:
        dlat[ii] = 0.5 * (latitude[ii + 1] - latitude[ii - 1])
        dlat[ii] = dlat[ii] * np.cos(np.deg2rad(latitude[ii]))

    if latitude[0] > -90.0 + 1.0e-4:
        dlat[0] = np.abs(latitude[1] - latitude[0]) * np.cos(np.deg2rad(latitude[0]))
    if latitude[-1] < +90.0 - 1.0e-4:
        dlat[-1] = np.abs(latitude[-1] - latitude[-2]) * np.cos(np.deg2rad(latitude[-1]))

    dlat[:] = METERS_PER_DEGREE * dlat[:]

    return xr.DataArray(
        dlon, dims=dimension_names["longitude"], coords={dimension_names["longitude"]: longitude}
    ), xr.DataArray(dlat, dims=dimension_names["latitude"], coords={dimension_names["latitude"]: latitude})


def _integrate_horizontal(field: xr.DataArray, dlon, dlat, preserve_dims: Optional[Iterable[str]] = None):
    if preserve_dims is not None and "longitude" in preserve_dims and "latitude" in preserve_dims:
        int_tot = field * dlon * dlat
    else:
        int_lon = field.dot(dlon)
        int_tot = int_lon.dot(dlat)

    return int_tot


def _trig_fields(longitude, latitude, dimension_names: list):
    nlon = len(longitude)

    lat_rad = np.deg2rad(latitude).to_numpy()

    cos_theta_np = np.cos(lat_rad)[:, None] * np.ones((1, nlon))
    sin_theta_np = np.sin(lat_rad)[:, None] * np.ones((1, nlon))

    cos_theta = xr.DataArray(
        cos_theta_np,
        dims=[dimension_names["latitude"], dimension_names["longitude"]],
        coords={dimension_names["latitude"]: latitude, dimension_names["longitude"]: longitude},
    )

    sin_theta = xr.DataArray(
        sin_theta_np,
        dims=[dimension_names["latitude"], dimension_names["longitude"]],
        coords={dimension_names["latitude"]: latitude, dimension_names["longitude"]: longitude},
    )

    cos_theta_inv = 1.0 / cos_theta

    return cos_theta, sin_theta, cos_theta_inv


def _pressure_level_thickness(levels):
    nl = len(levels)
    dp = np.zeros(nl)

    dp[1:-1] = 0.5 * (levels[2:] - levels[:-2])
    dp[0] = 0.5 * (levels[1] - levels[0])
    dp[-1] = 0.5 * (levels[-1] - levels[-2])

    # convert pressure level thickness from hPa to Pa and normalise by gravity to get \rho dz
    dp = 100.0 * dp / GRAVITY

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
    dimension_names: dict,
    preserve_dims: Optional[Iterable[str]] = None,
):
    r"""
    Williamson et. al., JCP (1992), eqns (3-4):

    lambda: longitude
    theta:  latitude (from the equator)
    grad f:  1/(r \cos(theta)) d f/d lambda, 1/r df/d theta
    div(u):  1/(r \cos(theta)) (d u/d lambda + d(v\cos(theta))/d theta)
    """

    # grad f:  1/(r \cos(\theta)) df/d\lambda, 1/r df/d\theta
    dfdx = field_scalar.differentiate(dimension_names["longitude"]) * cos_theta_inv / METERS_PER_DEGREE
    dfdy = field_scalar.differentiate(dimension_names["latitude"]) / METERS_PER_DEGREE
    grad_f_dot_u = dfdx * field_vector_x + dfdy * field_vector_y
    int_grad_f_dot_u = _integrate_horizontal(grad_f_dot_u, dlon, dlat, preserve_dims)

    # div(u):  1/(r \cos(\theta)) (du/d\lambda + d(v\cos(\theta))/d\theta)
    dudx = field_vector_x.differentiate(dimension_names["longitude"]) / METERS_PER_DEGREE
    dvdy = (
        field_vector_y.differentiate(dimension_names["latitude"]) * cos_theta / METERS_PER_DEGREE
        - sin_theta * field_vector_y / RAD_EARTH
    )
    div_u = cos_theta_inv * (dudx + dvdy)
    f_div_u = field_scalar * div_u
    int_f_div_u = _integrate_horizontal(f_div_u, dlon, dlat, preserve_dims)

    return int_grad_f_dot_u, int_f_div_u


def _resort_lon_from_m180to180_to_0to360(ds, lon_name):
    # customised from:
    # https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray

    # Adjust lon values to make sure they are within (0, 360)
    ds["_longitude_adjusted"] = xr.where(ds[lon_name] < 0, ds[lon_name] + 360, ds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds.swap_dims({lon_name: "_longitude_adjusted"})
        .sel(**{"_longitude_adjusted": sorted(ds._longitude_adjusted)})
        .drop_vars(lon_name)
    )

    ds = ds.rename({"_longitude_adjusted": lon_name})

    return ds
