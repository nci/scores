from collections.abc import Iterable
from typing import Optional

import numpy as np
import xarray as xr

from scores.physical.budgets_utils import (
    C_P,
    C_PV,
    L_V,
    _integrate_energy_exchange,
    _integrate_horizontal,
    _integration_weights,
    _pressure_level_thickness,
    _resort_lon_from_m180to180_to_0to360,
    _trig_fields,
)
from scores.typing import XarrayLike


def _prepare_fields(
    fields: xr.Dataset,
    longitude: np.ndarray,
    latitude: np.ndarray,
    sub_domain_longitude: np.ndarray | None = None,
    sub_domain_latitude: np.ndarray | None = None,
):
    """
    Select the subdomain in longitude and latitude over which the energetic integrals
    are to be computed.

    Args:
        fields: Input fields for the 4D space-time quantities, 3D space-time quantities and 2D time only
            quantities used to compute the energetics, and their space time dimensional attributes
        longitude: array of longitudes
        latitude: array of latitudes
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional)
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional)

    Returns:
        The fields within the sub-domain region.
    """

    if sub_domain_longitude is not None:
        fields = fields.sel(longitude=longitude)

    if sub_domain_latitude is not None:
        fields = fields.sel(latitude=latitude)

    return fields


def energy_components(
    fields: xr.Dataset,
    field_names: dict,
    dimension_names: dict,
    sub_domain_longitude: np.ndarray | None = None,
    sub_domain_latitude: np.ndarray | None = None,
    preserve_dims: Optional[Iterable[str]] = None,
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
        fields: Input fields for the 4D space-time quantities used to compute the energy components (specifically
            the water vapor, temperature and the zonal, meridional and vertical velocities, and the 3D space-time
            surface pressure, and the 2D space only surface geopotential), and their space time dimensional
            attributes.
        field_names: dictionary of strings denoting the names for the different fields, specifically - zonal velocity
            (u), meridional velocity (v), vertical velocity (w), temperature (t), water mass fraction (q), surface
            pressure (sp), surface geopotential (zs).
        dimension_names: list of names for the spatio-temporal dimensions.
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional).
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional).
        preserve_dims: textual list of dimensions not to integrate over (optional). May be "latitude" and "longitude"
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
          Lecture Notes Comput. Sci. Eng. (Vol. 80, pp. 357–380). Heidelberg, Germany: Springer.
    """
    # test for NaN values in input data
    for field_name_long, field_name in field_names.items():
        error_msg = ValueError("NaN value found in field: {field_name_long}")
        if np.any(np.isnan(fields[field_name].as_numpy())):
            raise error_msg  # pragma: no cover

    # re-order the longitudes to range between 0 and 360 degrees (global)
    if fields.longitude.values[0] < -0.1:
        fields = _resort_lon_from_m180to180_to_0to360(fields, dimension_names["longitude"])

    dlon, dlat = _integration_weights(
        fields.longitude.values, fields.latitude.values, dimension_names, sub_domain_longitude, sub_domain_latitude
    )

    # select the latitude and longitude sub-domain
    fields = _prepare_fields(fields, dlon.longitude, dlat.latitude, sub_domain_longitude, sub_domain_latitude)

    nt = len(fields.time)

    time_array = fields.time.values
    level_array = fields.level.values

    dp = _pressure_level_thickness(level_array)

    # get the surface geopotential (constant in time)
    zs = fields[field_names["surface_geopotential"]]

    #  From Taylor (2011), eqn (12.8)
    sp = fields[field_names["surface_pressure"]].sel(time=time_array)
    sp_zs = sp * zs
    P = _integrate_horizontal(sp_zs, dlon, dlat, preserve_dims)

    ult = fields[field_names["zonal_velocity"]].sel(time=time_array, level=level_array)
    vlt = fields[field_names["meridional_velocity"]].sel(time=time_array, level=level_array)
    wlt = fields[field_names["vertical_velocity"]].sel(time=time_array, level=level_array)
    tlt = fields[field_names["temperature"]].sel(time=time_array, level=level_array)
    qlt = fields[field_names["water_mass_fraction"]].sel(time=time_array, level=level_array)

    khlt = ult**2 + vlt**2
    kvlt = wlt**2
    cpt = (C_P * (1.0 - qlt) + C_PV * qlt) * tlt

    cpt_x = _integrate_horizontal(cpt, dlon, dlat, preserve_dims)
    qlt_x = _integrate_horizontal(qlt, dlon, dlat, preserve_dims)
    khlt_x = _integrate_horizontal(khlt, dlon, dlat, preserve_dims)
    kvlt_x = _integrate_horizontal(kvlt, dlon, dlat, preserve_dims)

    dp_x = xr.DataArray(np.zeros((nt, len(dp))), dims=(dimension_names["time"], dimension_names["level"]))
    dp_x = dp_x + dp

    if preserve_dims is not None and "level" in preserve_dims:
        I = dp_x * cpt_x
        L = dp_x * L_V * qlt_x
        Kh = dp_x * 0.5 * khlt_x
        Kv = dp_x * 0.5 * kvlt_x
    else:
        I = (dp_x * cpt_x).sum(dim=dimension_names["level"])
        L = (dp_x * L_V * qlt_x).sum(dim=dimension_names["level"])
        Kh = (dp_x * 0.5 * khlt_x).sum(dim=dimension_names["level"])
        Kv = (dp_x * 0.5 * kvlt_x).sum(dim=dimension_names["level"])

    I.name = "Internal"
    L.name = "Latent"
    P.name = "Potential"
    Kh.name = "HorizontalKinetic"
    Kv.name = "VerticalKinetic"

    energy_integrals = xr.merge([I, L, P, Kh, Kv])

    return energy_integrals


def energy_exchanges(
    fields: xr.Dataset,
    field_names: dict,
    dimension_names: dict,
    sub_domain_longitude: np.ndarray | None = None,
    sub_domain_latitude: np.ndarray | None = None,
    reduce_time: bool = False,
    preserve_dims: Optional[Iterable[str]] = None,
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
        fields: Input fields for the 4D space-time quantities used to compute the energy components (specifically
            the zonal and meridional velocities, and the geopotential, and the 2D space only surface geopotential),
            and their space time dimensional attributes
        field_names: dictionary of strings denoting the names for the different fields, specifically - zonal velocity
            (u), meridional velocity (v), vertical velocity (w), geopotential (z), surface geopotential (zs)
        dimension_names: list of names for the spatio-temporal dimensions
        sub_domain_longitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the longitudinal direction (optional)
        sub_domain_latitude: array containing the minimum and maximum values of the sub-domain over which the
            energy components are to be computed in the latitudinal direction (optional)
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
    for field_name_long, field_name in field_names.items():
        error_msg = ValueError(f"NaN value found in field: {field_name_long}")
        if np.any(np.isnan(fields[field_name].as_numpy())):
            raise error_msg  # pragma: no cover

    # cannot integrate over time if less than two time values
    error_msg = ValueError(
        f"Length of time dimension is {len(fields.time)},"
        + "can only run with 'reduce_time=True' if the"
        + "temporal dimension has two entries or more."
    )
    if reduce_time and len(fields.time) < 2:
        raise error_msg  # pragma: no cover

    # re-order the longitudes to range between 0 and 360 degrees (global)
    if fields.longitude.values[0] < -0.1:
        fields = _resort_lon_from_m180to180_to_0to360(fields, dimension_names["longitude"])

    dlon, dlat = _integration_weights(
        fields.longitude.values, fields.latitude.values, dimension_names, sub_domain_longitude, sub_domain_latitude
    )

    cos_theta, sin_theta, cos_theta_inv = _trig_fields(fields.longitude, fields.latitude, dimension_names)

    # select the temporal, vertical and horizontal sub-domain
    fields = _prepare_fields(fields, dlon.longitude, dlat.latitude, sub_domain_longitude, sub_domain_latitude)

    nt = len(fields.time)

    KtoI = np.zeros(nt)  # kinetic to internal energy exchange (horizontal)
    ItoK = np.zeros(nt)  # internal to kinetic energy exchange (horizontal)
    KtoP = np.zeros(nt)  # kinetic to potential energy exchange (horizontal)
    PtoK = np.zeros(nt)  # potential to kinetic energy exchange (horizontal)

    time_array = fields.time.values
    level_array = fields.level.values

    dp = _pressure_level_thickness(level_array)

    # get the surface geopotential (constant in time)
    zs = fields[field_names["surface_geopotential"]]

    ult = fields[field_names["zonal_velocity"]].sel(level=level_array, time=time_array)
    vlt = fields[field_names["meridional_velocity"]].sel(level=level_array, time=time_array)
    zlt = fields[field_names["geopotential"]].sel(level=level_array, time=time_array)

    dp_x = xr.DataArray(np.zeros((nt, len(dp))), dims=(dimension_names["time"], dimension_names["level"]))
    dp_x = dp_x + dp

    zs_v = xr.DataArray(
        np.zeros((nt, len(dp), len(dlat), len(dlon))),
        dims=(
            dimension_names["time"],
            dimension_names["level"],
            dimension_names["latitude"],
            dimension_names["longitude"],
        ),
    )
    zs_v = zs_v + zs

    z_m_zs = zlt - zs_v

    KtoI_t, ItoK_t = _integrate_energy_exchange(
        z_m_zs, ult, vlt, dlon, dlat, cos_theta, sin_theta, cos_theta_inv, dimension_names, preserve_dims
    )
    KtoP_t, PtoK_t = _integrate_energy_exchange(
        zs_v, ult, vlt, dlon, dlat, cos_theta, sin_theta, cos_theta_inv, dimension_names, preserve_dims
    )

    if preserve_dims is not None and "level" in preserve_dims:
        KtoP = dp_x * KtoP_t
        PtoK = dp_x * PtoK_t
        KtoI = dp_x * KtoI_t
        ItoK = dp_x * ItoK_t
    else:
        KtoP = (dp_x * KtoP_t).sum(dim=dimension_names["level"])
        PtoK = (dp_x * PtoK_t).sum(dim=dimension_names["level"])
        KtoI = (dp_x * KtoI_t).sum(dim=dimension_names["level"])
        ItoK = (dp_x * ItoK_t).sum(dim=dimension_names["level"])

    if reduce_time and len(ult.time) > 0:
        dt = ult.time[1] - ult.time[0]
        dt_seconds = float(dt.data * 1.0e-9)
        weights = xr.DataArray(dt_seconds * np.ones(len(ult.time)), dims=["time"])
        KtoP = KtoP.weighted(weights).sum(dim=dimension_names["time"])
        PtoK = PtoK.weighted(weights).sum(dim=dimension_names["time"])
        KtoI = KtoI.weighted(weights).sum(dim=dimension_names["time"])
        ItoK = ItoK.weighted(weights).sum(dim=dimension_names["time"])

    KtoP.name = "KineticToPotential"
    PtoK.name = "PotentialToKinetic"
    KtoI.name = "KineticToInternal"
    ItoK.name = "InternalToKinetic"

    energy_exchange_integrals = xr.merge([KtoI, ItoK, KtoP, PtoK])

    return energy_exchange_integrals
