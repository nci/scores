'''
Common backend functionality required for the energy budget diagnosics
'''

import numpy as np

RAD_EARTH = 6371220.0 # radius of the earth, m
METERS_PER_DEGREE = 2.0*np.pi*RAD_EARTH/360.0 # conversion from degrees to meters, m
LON_MIN = -180.0 # minimum permissible longitude, degrees
LON_MAX = +180.0 # maximum permissible longitude, degrees
LAT_LON = -90.0 # minimum permissible latitude, degrees
LAT_MAX = +90.0 # maximu permissible latitude, degrees
GRAVITY = 9.80665 # gravitational acceleration of the earth, m/s^2
C_P = 1006.0 # specific heat of dry air at constant pressure, J/kg/K
C_PV = 1872.0 # specific heat of water vapor at constant pressure, J/kg/K
L_V = 2.5008e+6 # specific latent heat of vaporisation, J/kg

'''
Integration weights for a two dimensional latitude-longitude field on
the surface of the sphere
'''
def integration_weights(longitude,latitude,sub_domain_lon=np.array([None]),sub_domain_lat=np.array([None])):
    # Check the longitude sub domain is valid if specified
    if sub_domain_lon[0] != None:
        error_msg = ValueError(f'sub-domain longitude outside valid range: ' + \
            f'{LON_MIN} <= minimum longitude < maximum longitude < {LON_MAX}')
        if len(sub_domain_lon) != 2:
            raise error_msg
        if sub_domain_lon[1] <= sub_domain_lon[0] or sub_domain_lon[0] < LON_MIN or \
                sub_domain_lon[0] > LON_MAX or sub_domain_lon[1] < LON_MIN or \
                sub_domain_lon[1] > LON_MAX:
            raise error_msg

        cond_lon = (longitude >= sub_domain_lon[0]) & (longitude <= sub_domain_lon[1])
        longitude = longitude[cond_lon]

    # assume the longitudes are regularly spaced
    _dlon = (longitude[1] - longitude[0])*METERS_PER_DEGREE
    dlon = _dlon*np.ones(len(longitude))

    # Check the latitude sub domain is valid if specified
    if sub_domain_lat[0] != None:
        error_msg = ValueError(f'sub-domain latitude outside valid range: ' + \
            f'{LAT_MIN} <= minimum latitude < maximum latitude < {LAT_MAX}')
        if len(sub_domain_lat) != 2:
            raise error_msg
        if sub_domain_lat[1] <= sub_domain_lat[0] or sub_domain_lat[0] < LAT_MIN or \
                sub_domain_lat[0] > LAT_MAX or sub_domain_lat[1] < LAT_MIN or \
                sub_domain_lat[1] > LAT_MAX:
            raise error_msg

        cond_lat = (latitude >= sub_domain_lat[0]) & (latitude <= sub_domain_lat[1])
        latitude = latitude[cond_lat]

    dlat = np.zeros(len(latitude))
    for ii in np.arange(len(latitude)-2) + 1:
        dlat[ii] = 0.5*(latitude[ii+1]-latitude[ii-1])
        dlat[ii] = dlat[ii]*np.cos(np.deg2rad(latitude[ii]))

    if latitude[0] > -90.0 + 1.0e-4:
        dlat[0] = np.abs(latitude[1] - latitude[0])*np.cos(np.deg2rad(latitude[0]))
    if latitude[-1] < +90.0 - 1.0e-4:
        dlat[-1] = np.abs(latitude[-1] - latitude[-2])*np.cos(np.deg2rad(latitude[-1]))

    dlat[:] = METERS_PER_DEGREE*dlat[:]

    return dlon, dlat, longitude, latitude

# dimensions are assumed as: field[num_latitudes,num_longitudes]
def integrate_horizontal(field, dlon, dlat):
    int_lon = np.dot(field, dlon)
    return np.dot(int_lon, dlat)

def trig_fields(longitude, latitude):
    nlon = len(longitude)
    nlat = len(latitude)

    cos_theta = np.zeros((nlat,nlon))
    sin_theta = np.zeros((nlat,nlon))
    for ii in np.arange(nlon):
        cos_theta[:,ii] = np.cos(np.deg2rad(latitude))
        sin_theta[:,ii] = np.sin(np.deg2rad(latitude))

    cos_theta_inv = 1.0/cos_theta

    return cos_theta, sin_theta, cos_theta_inv

def pressure_level_thickness(levels):
    nl = len(levels)
    dp = np.zeros(nl)
    for ll in np.arange(nl):
        if ll == 0:
            dp[ll] = 0.5*(levels[ll+1] - levels[ll])
        elif ll == nl-1:
            dp[ll] = 0.5*(levels[ll] - levels[ll-1])
        else:
            dp[ll] = 0.5*(levels[ll+1] - levels[ll-1])

    # convert pressure level thickness from hPa to Pa and normalise by gravity to get \rho dz
    dp = 100.0*dp/GRAVITY

    return dp

def integrate_energy_exchange(field_scalar, field_vector_x, field_vector_y, longitude, latitude, dlon, dlat, \
        cos_theta, sin_theta, cos_theta_inv):

    '''
    Williamson et. al., JCP (1992), eqns (3-4):

    lambda: longitude
    theta:  latitude (from the equator)
    grad f:  1/(r \cos(theta)) d f/d lambda, 1/r df/d theta
    div(u):  1/(r \cos(theta)) (d u/d lambda + d(v\cos(theta))/d theta)
    '''

    _dlon = np.abs(np.deg2rad(longitude[1] - longitude[0])*METERS_PER_DEGREE)
    _dlat = np.abs(np.deg2rad(latitude[1] - latitude[0])*METERS_PER_DEGREE)

    # grad f:  1/(r \cos(\theta)) df/d\lambda, 1/r df/d\theta
    dfdx = np.gradient(field_scalar,_dlon,axis=1)*cos_theta_inv
    dfdy = np.gradient(field_scalar,_dlat,axis=0)
    grad_f_dot_u = dfdx*field_vector_x + dfdy*field_vector_y
    int_grad_f_dot_u = integrate_horizontal(grad_f_dot_u,dlon,dlat)

    # div(u):  1/(r \cos(\theta)) (du/d\lambda + d(v\cos(\theta))/d\theta)
    dudx = np.gradient(field_vector_x,_dlon,axis=1)
    dvdy = np.gradient(field_vector_y,_dlat,axis=0)*cos_theta - sin_theta*field_vector_y/METERS_PER_DEGREE
    div_u = cos_theta_inv*(dudx + dvdy)
    f_div_u = field_scalar*div_u
    int_f_div_u = integrate_horizontal(f_div_u,dlon,dlat)

    return int_grad_f_dot_u, int_f_div_u
