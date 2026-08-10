import numpy as np
import xarray as xr

try:
    import dask
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

try:
    import pyshtools as pysh
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    pysh = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

from scores.dynamics import STANDARD_CONSTANTS, PlanetConstants
from scores.dynamics.budgets_utils import (
    _pressure_level_thickness,
    _scaled_rfft,
)
from scores.typing import XarrayLike


def power_spectra(
    data: xr.Dataset,
    *,
    reduce_vertical: bool = False,
    reduce_time: bool = False,
    spherical_harmonic: bool = False,
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    zonal_velocity_name: str = "u",
    meridional_velocity_name: str = "v",
    custom_field_name: str | None = None,
    zonal_filter_width: float = 0.125,
    constants: PlanetConstants = STANDARD_CONSTANTS,
) -> XarrayLike:
    """
    Compute the spectra. By default this is done by first computing the kinetic energy from the horizontal velocity
    components (u,v), but may alternatively be computed for a custom field.
    """

    _data = data.copy(deep=True)

    # average over the vertical dimension (pressure levels)
    if reduce_vertical and len(_data.level.values) > 1:
        dp = _pressure_level_thickness(_data.level.values, constants)
        dp = dp / np.sum(dp)
        dp_x = xr.DataArray(dp, dims=(pressure_level_name))
        _data = (dp_x * _data).sum(dim=pressure_level_name)

    # average over the time dimension
    nt = len(data.time.values)
    if reduce_time:
        _data = _data.sum(dim=time_name) / nt

    if spherical_harmonic:
        error_msg = ImportError(
            "The 'pyshtools' package is not installed, cannot perform a spherical harmonic transform."
        )
        if pysh == "Unavailable":
            raise error_msg

        if custom_field_name is None:
            K = 0.5 * (
                _data[zonal_velocity_name] * _data[zonal_velocity_name]
                + _data[meridional_velocity_name] * _data[meridional_velocity_name]
            )
        else:
            K = _data[custom_field_name]

        coeffs = pysh.expand.SHExpandDH(K)
        spec = pysh.spectralanalysis.spectrum(coeffs, unit="per_l")
        _coeffs = coeffs[0, :, :] * coeffs[0, :, :] + coeffs[1, :, :] * coeffs[1, :, :]
        if custom_field_name is None:
            _coeffs = np.sqrt(_coeffs)

        k_lat = np.arange(len(K.latitude) // 2)
        k_lon = np.arange(len(K.longitude) // 4)
        ds = xr.Dataset(
            data_vars={
                "amplitude_squared": (("meridional_wavenumber", "zonal_wavenumber"), _coeffs),
                "frequency": (("zonal_wavenumber"), spec),
            },
            coords={
                "meridional_wavenumber": k_lat,
                "zonal_wavenumber": k_lon,
            },
        )

        return ds

    else:
        n_lon = len(_data.longitude.values)
        L_theta = np.max(_data.longitude.values) - np.min(_data.longitude.values)
        d_theta = L_theta / (n_lon - 1)
        L = n_lon * d_theta
        if L < 360.0 - 1.0e-6:
            # regional domain, apply cosine bell filtering
            error_msg = ValueError(f"The zonal_filter_width is: {zonal_filter_width}, must be < 0.5.")
            if zonal_filter_width >= 0.5:
                raise error_msg

            theta_l = np.min(_data.longitude.values)
            theta_r = theta_l + L
            filter_l = 0.5 * (1.0 - np.cos(np.pi * (_data.longitude.values - theta_l) / (zonal_filter_width * L)))
            filter_r = 0.5 * (1.0 - np.cos(np.pi * (theta_r - _data.longitude.values) / (zonal_filter_width * L)))
            mask_l = _data.longitude.values < theta_l + zonal_filter_width * L
            mask_r = _data.longitude.values > theta_r - zonal_filter_width * L
            filter_1d = np.ones(n_lon)
            filter_1d[mask_l] = filter_l[mask_l]
            filter_1d[mask_r] = filter_r[mask_r]
            filter_xr = xr.DataArray(
                filter_1d,
                dims=[longitude_name],
                coords={longitude_name: _data.longitude},
            )
            _dims = _data.dims
            _data = (filter_xr * _data).transpose(*_dims)

        # ensure that the wavelengths are consistent for each latitude
        cos_theta_inv = 1.0 / np.cos(_data.latitude.values)
        equator_freq = np.fft.rfftfreq(n_lon, L / 2.0 / np.pi)
        freq_2d = cos_theta_inv[:, None] * equator_freq[None, :]
        freq_2d = xr.DataArray(freq_2d, dims=(latitude_name, "wavenumber"))

        if dask != "Unavailable":
            _data = _data.compute()

        if custom_field_name is None:
            lon_index = _data[zonal_velocity_name].get_axis_num(longitude_name)
            fft_lon_u = xr.apply_ufunc(
                _scaled_rfft,
                _data[zonal_velocity_name],
                input_core_dims=[[longitude_name]],
                output_core_dims=[["wavenumber"]],
                exclude_dims={longitude_name},
                kwargs={"axis": lon_index},
            )
            fft_lon_v = xr.apply_ufunc(
                _scaled_rfft,
                _data[meridional_velocity_name],
                input_core_dims=[[longitude_name]],
                output_core_dims=[["wavenumber"]],
                exclude_dims={longitude_name},
                kwargs={"axis": lon_index},
            )
            fft_lon = 0.5 * (fft_lon_u + fft_lon_v)
        else:
            lon_index = _data[custom_field_name].get_axis_num(longitude_name)
            fft_lon = xr.apply_ufunc(
                _scaled_rfft,
                _data[custom_field_name],
                input_core_dims=[[longitude_name]],
                output_core_dims=[["wavenumber"]],
                exclude_dims={longitude_name},
                kwargs={"axis": lon_index},
            )

        nw = fft_lon.sizes["wavenumber"]
        fft_lon = fft_lon.assign_coords(wavenumber=np.linspace(0.0, float(nw), nw, endpoint=False))

        ds = xr.Dataset(
            data_vars={
                "amplitude_squared": (fft_lon.dims, fft_lon.data),
                "frequency": (freq_2d.dims, freq_2d.data),
            },
        )
        if time_name in _data.dims and not reduce_time:
            ds = ds.assign_coords(time=_data.time)
        if pressure_level_name in _data.dims and not reduce_vertical:
            ds = ds.assign_coords(level=_data.level)
        ds = ds.assign_coords(latitude=_data.latitude)
        ds = ds.assign_coords(wavenumber=fft_lon.wavenumber)

        return ds
