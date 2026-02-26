import numpy as np
import xarray as xr

from scores.budgets.budgets_utils import *
from scores.typing import XarrayLike

def prepare_fields(
        fourDFields: XarrayLike,
        threeDFields: XarrayLike,
        twoDFields: XarrayLike,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None]),
        sub_domain_time = [],
        sub_domain_level = np.array([None])):

    twoDFields.sortby("latitude")
    threeDFields.sortby("latitude")
    fourDFields.sortby("latitude")

    if sub_domain_time != []:
        threeDFields = threeDFields.sel(time=sub_domain_time)
        fourDFields = fourDFields.sel(time=sub_domain_time)

    if sub_domain_level.any() != None:
        fourDFields = fourDFields.sel(level=sub_domain_level)

    if sub_domain_longitude.any() != None:
        twoDFields = twoDFields.sel(longitude=sub_domain_longitude)
        threeDFields = threeDFields.sel(longitude=sub_domain_longitude)
        fourDFields = fourDFields.sel(longitude=sub_domain_longitude)

    if sub_domain_latitude.any() != None:
        twoDFields = twoDFields.sel(latitude=sub_domain_latitude)
        threeDFields = threeDFields.sel(latitude=sub_domain_latitude)
        fourDFields = fourDFields.sel(latitude=sub_domain_latitude)

    return fourDFields, threeDFields, twoDFields

def energy_components(
        fourDFields: XarrayLike,
        threeDFields: XarrayLike,
        twoDFields: XarrayLike,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None]),
        sub_domain_time = [],
        sub_domain_level = np.array([None]),
        output_file_name = [],
        ) -> XarrayLike:

    '''
    compute the time series for the energy budget on pressure levels
    for details see:
        Trenberth et. al. J. Clim. (2002) v. 15 pp 3343--3360, eqn (5)
        Sha et. al (2025), eqn (12)

        1/g\int_{p1}^{p0} c_p* T + L_v q + \Phi_s + 0.5*(u*u + v*v + w*w) dp
        c_p* = cp(1-q) + c_{pv}q
    '''
    dlon, dlat, lon, lat = integration_weights(fourDFields.longitude.values, fourDFields.latitude.values, \
            sub_domain_longitude, sub_domain_latitude)

    # select the temporal, vertical and horizontal sub-domain
    fourDFields, threeDFields, twoDFields = prepare_fields(fourDFields, threeDFields, twoDFields, \
            lon, lat, sub_domain_time, sub_domain_level)

    # get the surface geopotential from the 2d fields
    zs = twoDFields["z_surf"]

    dp = pressure_level_thickness(sub_domain_level)

    nt = len(fourDFields.time)

    I = np.zeros(nt)    # global integral of the interal energy
    L = np.zeros(nt)    # global integral of the latent energy
    P = np.zeros(nt)    # global integral of the potential energy
    Kh = np.zeros(nt)   # global integral of the kinetic energy (horizontal component)
    Kv = np.zeros(nt)   # global integral of the kinetic energy (vertical component)

    if sub_domain_time != []:
        time_array = sub_domain_time
    else:
        time_array = fourDFields.time.values

    if sub_domain_level.any() != None:
        level_array = sub_domain_level
    else:
        level_array = fourDFields.level.values

    tt = 0
    for _time in time_array:
        #  From Taylor (2011), eqn (12.8)
        sp    = threeDFields["sp"].sel(time=_time)
        sp_zs = sp*zs
        P[tt] = integrate_horizontal(sp_zs,dlon,dlat)

        for _level in level_array:
            ult = fourDFields["u"].sel(level=_level,time=_time)
            vlt = fourDFields["v"].sel(level=_level,time=_time)
            wlt = fourDFields["w"].sel(level=_level,time=_time)
            tlt = fourDFields["t"].sel(level=_level,time=_time)
            qlt = fourDFields["q"].sel(level=_level,time=_time)

            khlt = ult*ult + vlt*vlt
            kvlt = wlt*wlt
            cpt  = (c_p*(1.0 - qlt) + c_pv*qlt)*tlt

            I[tt]  = I[tt] + dp[ll]*integrate_horizontal(cpt,dlon,dlat)
            L[tt]  = L[tt] + dp[ll]*L_v*integrate_horizontal(qlt,dlon,dlat)
            Kh[tt] = Kh[tt] + dp[ll]*0.5*integrate_horizontal(khlt,dlon,dlat)
            Kv[tt] = Kv[tt] + dp[ll]*0.5*integrate_horizontal(kvlt,dlon,dlat)

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
        fourDFields: XarrayLike,
        threeDFields: XarrayLike,
        twoDFields: XarrayLike,
        sub_domain_longitude = np.array([None]),
        sub_domain_latitude = np.array([None]),
        sub_domain_time = [],
        sub_domain_level = np.array([None]),
        output_file_name = [],
        ) -> XarrayLike:

    dlon, dlat, lon, lat = integration_weights(zs.longitude.values, zs.latitude.values, \
            sub_domain_longitude, sub_domain_latitude)

    cos_theta, sin_theta, cos_theta_inv = trig_fields(lon, lat)

    # select the temporal, vertical and horizontal sub-domain
    fourDFields, threeDFields, twoDFields = prepare_fields(fourDFields, threeDFields, twoDFields, \
            lon, lat, sub_domain_time, sub_domain_level)

    nt = len(fourDFields.time)

    KtoI = np.zeros(nt) # kinetic to internal energy exchange (horizontal)
    ItoK = np.zeros(nt) # internal to kinetic energy exchange (horizontal)
    KtoP = np.zeros(nt) # kinetic to potential energy exchange (horizontal)
    PtoK = np.zeros(nt) # potential to kinetic energy exchange (horizontal)

    dp = get_pressure_thickness(sub_domain_level)

    if sub_domain_time != []:
        time_array = sub_domain_time
    else:
        time_array = fourDFields.time.values

    if sub_domain_level.any() != None:
        level_array = sub_domain_level
    else:
        level_array = fourDFields.level.values

    tt = 0
    for _time in time_array:
        for _level in level_array:
            ult = fourDFields["u"].sel(level=_level,time=_time)
            vlt = fourDFields["v"].sel(level=_level,time=_time)
            zlt = fourDFields["z"].sel(level=_level,time=_time)

            z_m_zs = zlt - zs

            KtoI_t, ItoK_t = integrate_energy_exchange(z_m_zs, ult, vlt, dlon, dlat, cos_theta, sin_theta, cos_theta_inv)
            KtoI[tt] = KtoI[tt] + dp[ll]*KtoI_t
            ItoK[tt] = ItoK[tt] + dp[ll]*ItoK_t

            KtoP_t, PtoK_t = integrate_energy_exchange(zs, ult, vlt, dlon, dlat, cos_theta, sin_theta, cos_theta_inv)
            KtoP[tt] = KtoP[tt] + dp[ll]*KtoP_t
            PtoK[tt] = PtoK[tt] + dp[ll]*PtoK_t

        if output_file_name != []:
            KtoI_str = "{:16.15e}".format(KtoI[tt])
            ItoK_str = "{:16.15e}".format(ItoK[tt])
            KtoP_str = "{:16.15e}".format(KtoP[tt])
            PtoK_str = "{:16.15e}".format(PtoK[tt])

            with open(outfilename, "a") as outfile:
                outfile.write(f"{KtoI_str}\t{ItoK_str}\t{KtoP_str}\t{PtoK_str}\n")

        tt = tt + 1

    energy_exchange_integrals = xr.DataArray([KtoI, ItoK, KtoP, PtoK])

    return energy_exchange_integrals
