import numpy as np
import xarray as xr
try:
    import pyshtools as pysh
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    pysh = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

from scores.dynamics import STANDARD_CONSTANTS, PlanetConstants
from scores.dynamics.budgets_utils import (
    _integration_weights,
    _pressure_level_thickness,
    _scaled_rfft,
)
from scores.typing import XarrayLike

def power_spectra(
    data: xr.Dataset,
    *,
    preserve_vertial: bool = False,
    reduce_time: bool = False,
    spherical_harmonic: bool = False,
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    zonal_velocity_name: str = "u",
    meridional_velocity_name: str = "v",
    custom_field_name: str = "none",
    zonal_filter_width: float = 0.125,
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

    # average over the time dimension
    nt = len(data.time.values)
    if reduce_time and nt > 1:
        K = K.sum(dim=time_name)
        K = K / nt

    if spherical_harmonic:
        error_msg = ImportError("The 'pyshtools' package in not installed, "
                + "cannot perform a spherical harmonic transform.")
        if pysh == "Unavailable":
            raise error_msg

    else:
        n_lon = len(K.longitude.values)
        n_lat = len(K.latitude.values)
        L_theta = np.max(K.longitude.values) - np.min(K.longitude.values)
        d_theta = L_theta / (n_lon - 1)
        L = np.pi * constants.RAD_EARTH / 180.0 * n_lon * d_theta
        if L < 2.0 * np.pi * constants.RAD_EARTH - 1.0e-6:
            # regional domain, apply cosine bell filtering
	    error_msg = ValueError(f"The zonal_filter_width is: {zonal_filter_with}, must be < 0.5.")
	    if zonal_filter_width >= 0.5:
		    raise error_msg

	    theta_l = np.min(K.longitude.values)
	    theta_r = theta_l + L
	    filter_l = 0.5 * (1.0 - np.cos(np.pi * (K.longitude.values - theta_l) / (zonal_filter_width * L)))
	    filter_r = 0.5 * (1.0 - np.cos(np.pi * (theta_r - K.longitude.values) / (zonal_filter_width * L)))
	    mask_l = K.longitude.values < theta_l + zonal_filter_width * L
	    mask_r = K.longitude.values > theta_r - zonal_filter_width * L
	    filter_1d = np.ones(n_lon)
	    filter_1d[mask_l] = filter_l
	    filter_1d[mask_r] = filter_r
	    filter_2d = np.ones((nlat, 1)) * filter_1d[None, :]
	    filter_xr = xr.DataArray(filter_2d, dims=[latitude_name, longitude_name], coords={latitude_name: K.latitude, longitude_name: K.longitude})
	    K = filter_xr * K

        lon_index = K.get_index(longitude_name)
        fft_lon = xr.apply_ufunc(
                _scaled_rfft,
                K,
                input_core_dims=[[longitude_name]],
                output_core_dims=[["wavenumber"]],
                kwargs={index: lon_index, norm="forward"},
        )

        # ensure that the wavelengths are consistent for each latitude
        cos_theta_inv = 1.0 / np.cos(K.latitude.values)
        equator_freq = np.fft.rttffreq(n_lon, L / 2.0 / np.pi)
        freq_2d = cos_theta_inv[:, None] * equator_freq[None, :]
        fft_lon.["frequency"] = (["waqvenumber", latitude_name], freq_2d)

        return fft_lon
