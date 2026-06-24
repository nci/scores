import numpy as np
import xarray as xr

from scores.physical.budgets_utils import (
    _integrate_energy_exchange,
    _integrate_horizontal,
    _integration_weights,
    _pressure_level_thickness,
    _trig_fields,
    planet_constants,
)
from scores.typing import XarrayLike


def energy_components_lat_lon(
    data: xr.Dataset,
    *,
    preserve_horizontal: bool = False,
    preserve_vertical: bool = False,
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    zonal_velocity_name: str = "u",
    meridional_velocity_name: str = "v",
    vertical_velocity_name: str = "w",
    temperature_name: str = "t",
    vapour_mass_fraction_name: str = "q",
    liquid_mass_fraction_name: str | None = None,
    ice_mass_fraction_name: str | None = None,
    surface_pressure_name: str = "sp",
    surface_geopotential_name: str = "zs",
    constants: planet_constants | None = None,
) -> XarrayLike:
    """
    Compute the time series for the energy budget on pressure levels

    .. math::
        \\text{Internal}  = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}(C_p(1-q) + C_{pv}q)T\\text{d}\\Omega\\text{d}p
        \\text{Latent}    = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}L_v q\\text{d}\\Omega\\text{d}p
        \\text{Potential} = \\frac{1}{g}\\int_{\\Omega}z_s\\Phi_s\\text{d}\\Omega
        \\text{Kinetic}   = \\frac{1}{g}\\int_{p_1}^{p_0}\\int_{\\Omega}
                            \\frac{1}{2}(u^2 + v^2 + w^2)\\text{d}\\Omega\\text{d}p

    Args:
        data: Input fields for the 4D space-time quantities used to compute the energy components (specifically
            the water vapor, temperature and the zonal, meridional and vertical velocities, and the 3D space-time
            surface pressure, and the 2D space only surface geopotential), and their space time dimensional
            attributes.
        field_names: dictionary of strings denoting the names for the different fields, specifically - zonal velocity
            (u), meridional velocity (v), vertical velocity (w), temperature (t), vapour mass fraction (q), liquid
            mass fraction (ql), ice mass fraction (qi), surface pressure (sp), surface geopotential (zs).
        longitude_name: string giving the textual name of the longitude coordinate
        latitude_name: string giving the textual name of the latitude coordinate
        prese: textual list of dimensions not to integrate over (optional). May be "latitude" and "longitude"
            to preserve the horizontal dimensions and/or "level" to preserve the vertical pressure level.

    Returns:
        2D array containing the time series for the domain integrals of the energy components at each time.

    References:
        Trenberth, K. E., Stepaniak, D. P., Caron, J. M. (2002) "Accuracy of Atmospheric Energy Budgets from Analyses"
          J. Clim. 15 3343--3360
        Sha, Y., Schreck, J. S., Chapman, W., Gagne, D. J. (2025) "Improving AI Weather Prediction Models using Global
          Mass and Energy Conservation Schemes" arXiv:2501.05648v2
        Taylor, M. A. (2011). Conservation of mass and energy for the moist atmospheric primitive equations on
          unstructured grids. In P. H. Lauritzen, et al. (Eds.), Numerical techniques for global atmospheric models,
          Lecture Notes Comput. Sci. Eng. (Vol. 80, pp. 357--380). Heidelberg, Germany: Springer.
    """
    # test for NaN values in input data
    field_names = [
        zonal_velocity_name,
        meridional_velocity_name,
        vertical_velocity_name,
        temperature_name,
        vapour_mass_fraction_name,
        liquid_mass_fraction_name,
        ice_mass_fraction_name,
        surface_pressure_name,
        surface_geopotential_name,
    ]
    for field_name in field_names:
        if field_name is None:
            continue
        error_msg = ValueError("NaN value found in field: {field_name}")
        if np.any(np.isnan(data[field_name].as_numpy())):
            raise error_msg  # pragma: no cover

    # iniitialise the planetary constants with their default values if not supplied
    if constants is None:
        constants = planet_constants()

    dlon, dlat = _integration_weights(
        data.longitude.values,
        data.latitude.values,
        longitude_name,
        latitude_name,
        constants,
    )

    nt = len(data.time)

    time_array = data.time.values
    level_array = data.level.values

    dp = _pressure_level_thickness(level_array, constants)

    # get the surface geopotential (constant in time)
    zs = data[surface_geopotential_name]

    #  From Taylor (2011), eqn (12.8)
    sp = data[surface_pressure_name].sel(time=time_array)
    sp_zs = sp * zs
    P = _integrate_horizontal(sp_zs, dlon, dlat, preserve_horizontal)

    ult = data[zonal_velocity_name].sel(time=time_array, level=level_array)
    vlt = data[meridional_velocity_name].sel(time=time_array, level=level_array)
    wlt = data[vertical_velocity_name].sel(time=time_array, level=level_array)
    tlt = data[temperature_name].sel(time=time_array, level=level_array)
    qlt = data[vapour_mass_fraction_name].sel(time=time_array, level=level_array)

    # Eldred, et. al., QJRMS (2022), eqn 65, internal energy:
    #   \int\rho ( (c_vd * q_d + c_vv * q_l + c_l * q_l + c_i * q_i) * (T - T_0)
    #              - q_v * R_v * T_0 + q_v * (L_v0 + L_f0) + q_l * L_f0 ) d\Omega
    cpt = constants.C_PV * qlt * tlt
    qdlt = 1.0 - qlt
    # increment the specific heat at constant pressure by the liquid component if present
    if liquid_mass_fraction_name is not None and liquid_mass_fraction_name in data:
        qllt = data[liquid_mass_fraction_name].sel(time=time_array, level=level_array)
        cpt += constants.C_L * qllt * tlt
        qdlt -= qllt
    # increment the specific heat at constant pressure by the ice component if present
    if ice_mass_fraction_name is not None and ice_mass_fraction_name in data:
        qilt = data[ice_mass_fraction_name].sel(time=time_array, level=level_array)
        cpt += constants.C_I * qilt * tlt
        qdlt -= qilt
    cpt += constants.C_PD * qdlt * tlt

    khlt = ult**2 + vlt**2
    kvlt = wlt**2

    cpt_x = _integrate_horizontal(cpt, dlon, dlat, preserve_horizontal)
    qlt_x = _integrate_horizontal(qlt, dlon, dlat, preserve_horizontal)
    if liquid_mass_fraction_name is not None and liquid_mass_fraction_name in data:
        qllt_x = _integrate_horizontal(qllt, dlon, dlat, preserve_horizontal)
    khlt_x = _integrate_horizontal(khlt, dlon, dlat, preserve_horizontal)
    kvlt_x = _integrate_horizontal(kvlt, dlon, dlat, preserve_horizontal)

    dp_x = xr.DataArray(np.zeros((nt, len(dp))), dims=(time_name, pressure_level_name))
    dp_x = dp_x + dp

    if preserve_vertical:
        I = dp_x * cpt_x
        L = dp_x * (constants.L_V - constants.R_V * constants.T_0) * qlt_x
        if liquid_mass_fraction_name is not None and liquid_mass_fraction_name in data:
            L += dp_x * constants.L_F * (qlt_x + qllt_x)
        Kh = dp_x * 0.5 * khlt_x
        Kv = dp_x * 0.5 * kvlt_x
    else:
        I = (dp_x * cpt_x).sum(dim=pressure_level_name)
        L = (dp_x * (constants.L_V - constants.R_V * constants.T_0) * qlt_x).sum(dim=pressure_level_name)
        if liquid_mass_fraction_name is not None and liquid_mass_fraction_name in data:
            L += (dp_x * constants.L_F * (qlt_x + qllt_x)).sum(dim=pressure_level_name)
        Kh = (dp_x * 0.5 * khlt_x).sum(dim=pressure_level_name)
        Kv = (dp_x * 0.5 * kvlt_x).sum(dim=pressure_level_name)

    I.name = "Internal"
    L.name = "Latent"
    P.name = "Potential"
    Kh.name = "HorizontalKinetic"
    Kv.name = "VerticalKinetic"

    energy_integrals = xr.merge([I, L, P, Kh, Kv])

    return energy_integrals


def energy_exchanges_lat_lon(
    data: xr.Dataset,
    *,
    preserve_horizontal: bool = False,
    preserve_vertical: bool = False,
    reduce_time: bool = False,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    zonal_velocity_name: str = "u",
    meridional_velocity_name: str = "v",
    geopotential_name: str = "z",
    surface_geopotential_name: str = "zs",
    constants: planet_constants | None = None,
) -> XarrayLike:
    """
    Compute the exchanges between kinetic to internal, internal to kinetic, kinetic to potential and
    potential to kinetic energies.

    .. math::
        \\text{Kinetic to Internal}  = \\int_{p_1}^{p_0}\\int_{\\Omega}\\nabla (z-z_s)\\cdot\\boldsymbol{u}
                                       \\text{d}\\Omega\\text{d}p
        \\text{Internal to Kinetic}  = \\int_{p_1}^{p_0}\\int_{\\Omega}(z-z_s) \\nabla\\cdot\\boldsymbol{u}
                                       \\text{d}\\Omega\\text{d}p
        \\text{Kinetic to Potential} = \\int_{p_1}^{p_0}\\int_{\\Omega}\\nabla z_s\\cdot\\boldsymbol{u}
                                       \\text{d}\\Omega\\text{d}p
        \\text{Potential to Kinetic} = \\int_{p_1}^{p_0}\\int_{\\Omega}z_s \\nabla\\cdot\\boldsymbol{u}
                                       \\text{d}\\Omega\\text{d}p

    Args:
        data: Input fields for the 4D space-time quantities used to compute the energy components (specifically
            the zonal and meridional velocities, and the geopotential, and the 2D space only surface geopotential),
            and their space time dimensional attributes
        field_names: dictionary of strings denoting the names for the different fields, specifically - zonal velocity
            (u), meridional velocity (v), vertical velocity (w), geopotential (z), surface geopotential (zs)
        longitude_name: string giving the textual name of the longitude coordinate
        latitude_name: string giving the textual name of the latitude coordinate
        reduce_time: Integrate over the temporal dimension rather than return a time series.
        preserve_dims: textual list of dimensions not to integrate over (optional). May be "latitude" and "longitude"
            to preserve the horizontal dimensions and/or "level" to preserve the vertical pressure level.

    Returns:
        2D array containing the time series for the domain integrals of the energy exchanges at each time

    References:
        Taylor, M. A. (2011). Conservation of mass and energy for the moist atmospheric primitive equations on
          unstructured grids. In P. H. Lauritzen, et al. (Eds.), Numerical techniques for global atmospheric models,
          Lecture Notes Comput. Sci. Eng. (Vol. 80, pp. 357–380). Heidelberg, Germany: Springer.
    """
    # test for NaN values in input data
    field_names = [zonal_velocity_name, meridional_velocity_name, geopotential_name, surface_geopotential_name]
    for field_name in field_names:
        error_msg = ValueError(f"NaN value found in field: {field_name}")
        if np.any(np.isnan(data[field_name].as_numpy())):
            raise error_msg  # pragma: no cover

    # cannot integrate over time if less than two time values
    error_msg = ValueError(
        f"Length of time dimension is {len(data.time)},"
        + "can only run with 'reduce_time=True' if the"
        + "temporal dimension has two entries or more."
    )
    if reduce_time and len(data.time) < 2:
        raise error_msg  # pragma: no cover

    # iniitialise the planetary constants with their default values if not supplied
    if constants is None:
        constants = planet_constants()

    dlon, dlat = _integration_weights(
        data.longitude.values,
        data.latitude.values,
        longitude_name,
        latitude_name,
        constants,
    )

    cos_theta, sin_theta, cos_theta_inv = _trig_fields(data.longitude, data.latitude, longitude_name, latitude_name)

    nt = len(data.time)

    KtoI = np.zeros(nt)  # kinetic to internal energy exchange (horizontal)
    ItoK = np.zeros(nt)  # internal to kinetic energy exchange (horizontal)
    KtoP = np.zeros(nt)  # kinetic to potential energy exchange (horizontal)
    PtoK = np.zeros(nt)  # potential to kinetic energy exchange (horizontal)

    time_array = data.time.values
    level_array = data.level.values

    dp = _pressure_level_thickness(level_array, constants)

    # get the surface geopotential (constant in time)
    zs = data[surface_geopotential_name]

    ult = data[zonal_velocity_name].sel(level=level_array, time=time_array)
    vlt = data[meridional_velocity_name].sel(level=level_array, time=time_array)
    zlt = data[geopotential_name].sel(level=level_array, time=time_array)

    dp_x = xr.DataArray(np.zeros((nt, len(dp))), dims=(time_name, pressure_level_name))
    dp_x = dp_x + dp

    zs_v = xr.DataArray(
        np.zeros((nt, len(dp), len(dlat), len(dlon))),
        dims=(
            time_name,
            pressure_level_name,
            latitude_name,
            longitude_name,
        ),
    )
    zs_v = zs_v + zs

    z_m_zs = zlt - zs_v

    KtoI_t, ItoK_t = _integrate_energy_exchange(
        z_m_zs,
        ult,
        vlt,
        dlon,
        dlat,
        cos_theta,
        sin_theta,
        cos_theta_inv,
        longitude_name,
        latitude_name,
        constants,
        preserve_horizontal,
    )
    KtoP_t, PtoK_t = _integrate_energy_exchange(
        zs_v,
        ult,
        vlt,
        dlon,
        dlat,
        cos_theta,
        sin_theta,
        cos_theta_inv,
        longitude_name,
        latitude_name,
        constants,
        preserve_horizontal,
    )

    if preserve_vertical:
        KtoP = dp_x * KtoP_t
        PtoK = dp_x * PtoK_t
        KtoI = dp_x * KtoI_t
        ItoK = dp_x * ItoK_t
    else:
        KtoP = (dp_x * KtoP_t).sum(dim=pressure_level_name)
        PtoK = (dp_x * PtoK_t).sum(dim=pressure_level_name)
        KtoI = (dp_x * KtoI_t).sum(dim=pressure_level_name)
        ItoK = (dp_x * ItoK_t).sum(dim=pressure_level_name)

    if reduce_time and len(ult.time) > 0:
        dt = ult.time[1] - ult.time[0]
        dt_seconds = float(dt.data * 1.0e-9)
        weights = xr.DataArray(dt_seconds * np.ones(len(ult.time)), dims=[time_name])
        KtoP = KtoP.weighted(weights).sum(dim=time_name)
        PtoK = PtoK.weighted(weights).sum(dim=time_name)
        KtoI = KtoI.weighted(weights).sum(dim=time_name)
        ItoK = ItoK.weighted(weights).sum(dim=time_name)

    KtoP.name = "KineticToPotential"
    PtoK.name = "PotentialToKinetic"
    KtoI.name = "KineticToInternal"
    ItoK.name = "InternalToKinetic"

    energy_exchange_integrals = xr.merge([KtoI, ItoK, KtoP, PtoK])

    return energy_exchange_integrals
