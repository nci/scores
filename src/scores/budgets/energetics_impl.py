import numpy as np
import xarray as xr

from scores.budgets.budgets_utils import *
from scores.typing import XarrayLike

def prepare_fields(
        fields: XarrayLike,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None])):
    '''
    Select the subdomain in longitude and latitude over which the energetic integrals
    are to be computed.

    Args:
        fields: Input fields for the 4D space-time quantities, 3D space-time quantities and 2D time only 
            quantities used to compute the energetics, and their space time dimensional attributes
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional)
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional)

    Returns:
        The fields within the sub-domain region.
    '''

    fields.sortby("latitude")

    if sub_domain_longitude.any() != None:
        fields = fields.sel(longitude=sub_domain_longitude)

    if sub_domain_latitude.any() != None:
        fields = fields.sel(latitude=sub_domain_latitude)

    return fields

def energy_components(
        fields: XarrayLike,
        fieldnames,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None]),
        output_file_name = [],
        ) -> XarrayLike:
    '''
    Compute the time series for the energy budget on pressure levels

    .. math::
        \\text{Internal}  = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}(C_p(1-q) + C_{pv}q)T\\text{d}\\Omega\\text{d}p
        \\text{Latent}    = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}L_v q\\text{d}\\Omega\\text{d}p
        \\text{Potential} = \\frac{1}{g}\\int_{\\Omega}\\Phi_s\\text{d}\\Omega
        \\text{Kinetic}   = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}\\frac{1}{2}(u^2 + v^2 + w^2)\\text{d}\\Omega\\text{d}p

    Args:
        fields: Input fields for the 4D space-time quantities used to compute the energy components (specifically 
            the water vapor, temperature and the zonal, meridional and vertical velocities, and the 3D space-time
            surface pressure, and the 2D space only surface geopotential), and their space time dimensional 
            attributes
        fieldnames: list of strings denoting the names for the different fields, specifically - 0: zonal velocity, 
            1: meridional velocity, 2: vertical velocity, 3: temperature, 4: water vapor, 6: surface pressure,
            7: surface geopotential
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional)
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional)
        output_file_name: name of the test file to which to write the output (optional)
            
    Returns:
        2D array containing the time series for the domain integrals of the energy components at each time

    References:
        Trenberth et. al. J. Clim. (2002) v. 15 pp 3343--3360, eqn (5)
        Sha et. al (2025), eqn (12)
        Taylor, M. "Conservation of Mass and Energy for the Moist 
        Atmospheric Primitive Equations on Unstructured Grids", (2011) Section 12.4.5
    '''
    dlon, dlat, lon, lat = integration_weights(fields.longitude.values, fields.latitude.values, \
            sub_domain_longitude, sub_domain_latitude)

    # select the latitude and longitude sub-domain
    fields = prepare_fields(fields, lon, lat)

    # get the surface geopotential from the 2d fields
    zs = fields[fieldnames[7]]

    dp = pressure_level_thickness(sub_domain_level)

    nt = len(fields.time)

    I = np.zeros(nt)    # global integral of the interal energy
    L = np.zeros(nt)    # global integral of the latent energy
    P = np.zeros(nt)    # global integral of the potential energy
    Kh = np.zeros(nt)   # global integral of the kinetic energy (horizontal component)
    Kv = np.zeros(nt)   # global integral of the kinetic energy (vertical component)

    time_array = fields.time.values
    level_array = fields.level.values

    tt = 0
    for _time in time_array:
        #  From Taylor (2011), eqn (12.8)
        sp    = fields[fieldnames[6]].sel(time=_time)
        sp_zs = sp*zs
        P[tt] = integrate_horizontal(sp_zs,dlon,dlat)

        ll = 0
        for _level in level_array:
            ult = fields[fieldnames[0]].sel(level=_level,time=_time)
            vlt = fields[fieldnames[1]].sel(level=_level,time=_time)
            wlt = fields[fieldnames[2]].sel(level=_level,time=_time)
            tlt = fields[fieldnames[3]].sel(level=_level,time=_time)
            qlt = fields[fieldnames[4]].sel(level=_level,time=_time)

            khlt = ult*ult + vlt*vlt
            kvlt = wlt*wlt
            cpt  = (C_P*(1.0 - qlt) + C_PV*qlt)*tlt

            I[tt]  = I[tt] + dp[ll]*integrate_horizontal(cpt,dlon,dlat)
            L[tt]  = L[tt] + dp[ll]*L_V*integrate_horizontal(qlt,dlon,dlat)
            Kh[tt] = Kh[tt] + dp[ll]*0.5*integrate_horizontal(khlt,dlon,dlat)
            Kv[tt] = Kv[tt] + dp[ll]*0.5*integrate_horizontal(kvlt,dlon,dlat)

            ll = ll + 1

        if output_file_name != []:
            I_str = "{:16.15e}".format(I[tt])
            L_str = "{:16.15e}".format(L[tt])
            P_str = "{:16.15e}".format(P[tt])
            Kh_str = "{:16.15e}".format(Kh[tt])
            Kv_str = "{:16.15e}".format(Kv[tt])

            with open(output_file_name, "a") as outfile:
                outfile.write(f"{I_str}\t{L_str}\t{P_str}\t{Kh_str}\t{Kv_str}\n")

        tt = tt + 1

    energy_integrals = xr.DataArray([I, L, P, Kh, Kv])

    return energy_integrals

def energy_exchanges(
        fields: XarrayLike,
        fieldnames,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None]),
        output_file_name = [],
        ) -> XarrayLike:
    """
    Compute the exchanges between kinetic to internal, internal to kinetic, kinetic to potential and
    potential to kinetic energies.

    .. math::
        \\text{Kinetic to Internal}  = \\int_{p_1}^{p_0}\\int_{\\Omega}\\nabla (z-z_s)\\cdot\\boldsymbol{u}\\text{d}\\Omega\\text{d}p
        \\text{Internal to Kinetic}  = \\int_{p_1}^{p_0}\\int_{\\Omega}(z-z_s) \\nabla\\cdot\\boldsymbol{u}\\text{d}\\Omega\\text{d}p
        \\text{Kinetic to Potential} = \\int_{p_1}^{p_0}\\int_{\\Omega}\\nabla z_s\\cdot\\boldsymbol{u}\\text{d}\\Omega\\text{d}p
        \\text{Potential to Kinetic} = \\int_{p_1}^{p_0}\\int_{\\Omega}z_s \\nabla\\cdot\\boldsymbol{u}\\text{d}\\Omega\\text{d}p

    Args:
        fields: Input fields for the 4D space-time quantities used to compute the energy components (specifically 
            the zonal and meridional velocities, and the geopotential, and the 2D space only surface geopotential), 
            and their space time dimensional attributes
        fieldnames: list of strings denoting the names for the different fields, specifically - 0: zonal velocity, 
            1: meridional velocity, 2: vertical velocity, 5: geopotential, 7: surface geopotential
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional)
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional)
        output_file_name: name of the test file to which to write the output (optional)

    Returns:
        2D array containing the time series for the domain integrals of the energy exchanges at each time

    References:
        Taylor, M. "Conservation of Mass and Energy for the Moist 
        Atmospheric Primitive Equations on Unstructured Grids", (2011) Section 12.4.5
    """

    dlon, dlat, lon, lat = integration_weights(fields.longitude.values, fields.latitude.values, \
            sub_domain_longitude, sub_domain_latitude)

    cos_theta, sin_theta, cos_theta_inv = trig_fields(lon, lat)

    # select the temporal, vertical and horizontal sub-domain
    fields = prepare_fields(fields, lon, lat)

    nt = len(fields.time)

    KtoI = np.zeros(nt) # kinetic to internal energy exchange (horizontal)
    ItoK = np.zeros(nt) # internal to kinetic energy exchange (horizontal)
    KtoP = np.zeros(nt) # kinetic to potential energy exchange (horizontal)
    PtoK = np.zeros(nt) # potential to kinetic energy exchange (horizontal)

    dp = pressure_level_thickness(sub_domain_level)

    time_array = fields.time.values
    level_array = fields.level.values

    zs = fields[fieldnames[7]]

    tt = 0
    for _time in time_array:
        ll = 0
        for _level in level_array:
            ult = fields[fieldnames[0]].sel(level=_level,time=_time)
            vlt = fields[fieldnames[1]].sel(level=_level,time=_time)
            zlt = fields[fieldnames[5]].sel(level=_level,time=_time)

            z_m_zs = zlt - zs

            KtoI_t, ItoK_t = integrate_energy_exchange(z_m_zs, ult, vlt, lon, lat, dlon, dlat, cos_theta, sin_theta, cos_theta_inv)
            KtoI[tt] = KtoI[tt] + dp[ll]*KtoI_t
            ItoK[tt] = ItoK[tt] + dp[ll]*ItoK_t

            KtoP_t, PtoK_t = integrate_energy_exchange(zs, ult, vlt, lon, lat, dlon, dlat, cos_theta, sin_theta, cos_theta_inv)
            KtoP[tt] = KtoP[tt] + dp[ll]*KtoP_t
            PtoK[tt] = PtoK[tt] + dp[ll]*PtoK_t

            ll = ll + 1

        if output_file_name != []:
            KtoI_str = "{:16.15e}".format(KtoI[tt])
            ItoK_str = "{:16.15e}".format(ItoK[tt])
            KtoP_str = "{:16.15e}".format(KtoP[tt])
            PtoK_str = "{:16.15e}".format(PtoK[tt])

            with open(output_file_name, "a") as outfile:
                outfile.write(f"{KtoI_str}\t{ItoK_str}\t{KtoP_str}\t{PtoK_str}\n")

        tt = tt + 1

    energy_exchange_integrals = xr.DataArray([KtoI, ItoK, KtoP, PtoK])

    return energy_exchange_integrals
