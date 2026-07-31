import numpy as np
import xarray as xr

from scores.dynamics import STANDARD_CONSTANTS, PlanetConstants
from scores.dynamics.budgets_utils import (
    _integration_weights,
    _pressure_level_thickness,
    _trig_fields,
)
from scores.typing import XarrayLike

def power_spectra(
    data: xr.Dataset,
    *,
    preserve_vertial: bool = False,
    preserve_meridional: bool = False,
    reduce_time: bool = False,
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    zonal_velocity_name: str = "u",
    meridional_velocity_name: str = "v",
    custom_field_name: str = "none",
    constants: PlanetConstants = STANDARD_CONSTANTS,
) -> XarrayLike:
    """
    Compute the spectra. By default this is done by first computing the kinetic energy from the horizontal velocity 
    components (u,v), but may alternatively be computed for a custom field.
    """

    # get the kinetic energy, K
    if custom_field_name != "none":
        K = data[custom_field_name]
    else:
        K = 0.5 * (data[zonal_velocity_name] + data[meridional_field_name])

    # average over the vertical dimension (pressure levels)
    if not preserve_vertical and len(data.level.values) > 1:
        dp = _pressure_level_thickness(data.level.values, constants)
        dp_x = xr.zeros_like(K)
        dp_x = dp_x + dp
        K = (dp_x * K).sum(dim=pressure_level_name)
        K = K / np.sum(dp)

    
