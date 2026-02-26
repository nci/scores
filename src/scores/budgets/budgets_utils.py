'''
Common backend functionality required for the energy budget diagnosics
'''

import numpy as np

rad_earth = 6371220.0
meters_per_degree = 2.0*np.pi*rad_earth/360.0
lon_min = -180.0
lon_max = +180.0
lat_min = -90.0
lat_max = +90.0
gravity = 9.80665
c_p = 1006.0
c_pv = 1872.0
L_v = 2.5008e+6

'''
Integration weights for a two dimensional latitude-longitude field on
the surface of the sphere
'''
def integration_weights(longitude,latitude,sub_domain_lon=np.array([None]),sub_domain_lat=np.array([None])):
    # Check the longitude sub domain is valid if specified
    if sub_domain_lon[0] != None:
        error_msg = ValueError(f'sub-domain longitude outside valid range: ' + \
            f'{lon_min} <= minimum longitude < maximum longitude < {lon_max}')
        if len(sub_domain_lon) != 2:
            raise error_msg
        if sub_domain_lon[1] <= sub_domain_lon[0] or sub_domain_lon[0] < lon_min or \
                sub_domain_lon[0] > lon_max or sub_domain_lon[1] < lon_min or \
                sub_domain_lon[1] > lon_max:
            raise error_msg

        cond_lon = (longitude >= sub_domain_lon[0]) & (longitude <= sub_domain_lon[1])
        longitude = longitude[cond_lon]

    # assume the longitudes are regularly spaced
    _dlon = (longitude[1] - longitude[0])*meters_per_degree
    dlon = _dlon*np.ones(len(longitude))

    # Check the latitude sub domain is valid if specified
    if sub_domain_lat[0] != None:
        error_msg = ValueError(f'sub-domain latitude outside valid range: ' + \
            f'{lat_min} <= minimum latitude < maximum latitude < {lat_max}')
        if len(sub_domain_lat) != 2:
            raise error_msg
        if sub_domain_lat[1] <= sub_domain_lat[0] or sub_domain_lat[0] < lat_min or \
                sub_domain_lat[0] > lat_max or sub_domain_lat[1] < lat_min or \
                sub_domain_lat[1] > lat_max:
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

    dlat[:] = meters_per_degree*dlat[:]

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
    cos_theta_inv = np.zeros((nlat,nlon))
    for ii in np.arange(nlon):
        cos_theta[:,ii] = np.cos(np.deg2rad(latitude))
        sin_theta[:,ii] = np.sin(np.deg2rad(latitude))
        cos_theta_inv[:,ii] = 1.0/cos_theta[:,ii]

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
    dp = 100.0*dp/gravity

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

    _dlon = np.abs(np.deg2rad(longitude[1] - longitude[0])*meters_per_degree)
    _dlat = np.abs(np.deg2rad(latitude[1] - latitude[0])*meters_per_degree)

    # grad f:  1/(r \cos(\theta)) df/d\lambda, 1/r df/d\theta
    dfdx = np.gradient(field_scalar,_dlon,axis=1)*cos_theta_inv
    dfdy = np.gradient(field_scalar,_dlat,axis=0)
    grad_f_dot_u = dfdx*field_vector_x + dfdy*field_vector_y
    int_grad_f_dot_u = integrate_horizontal(grad_f_dot_u,dlon,dlat)

    # div(u):  1/(r \cos(\theta)) (du/d\lambda + d(v\cos(\theta))/d\theta)
    dudx = np.gradient(field_vector_x,_dlon,axis=1)
    dvdy = np.gradient(field_vector_y,_dlat,axis=0)*cos_theta - sin_theta*field_vector_y/meters_per_degree
    div_u = cos_theta_inv*(dudx + dvdy)
    f_div_u = field_scalar*div_u
    int_f_div_u = integrate_horizontal(f_div_u,dlon,dlat)

    return int_grad_f_dot_u, int_f_div_u
