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


def power_spectra_scalar(
    data: xr.DataArray,
    *,
    reduce_vertical: bool = False,
    reduce_time: bool = False,
    spherical_harmonic: bool = False,
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    pressure_level_name: str = "level",
    time_name: str = "time",
    scalar_field_name: str = "w",
    zonal_filter_width: float = 0.125,
    vertical_weights: np.ndarray | None = None,
    constants: PlanetConstants = STANDARD_CONSTANTS,
) -> XarrayLike:
    """
    Compute the spectra for a scalar field.

    Args:
        data: Xarray DataArray containing the field to transform (maybe derived from the zonal and meridional velocity
            components, or from a custom field).
        reduce_vertical: average over the vertical pressure levels (default is False).
        reduce_time: average over the time levels (default is False).
        spherical_harmonic: compute spectra using spherical harmonics in both the zonal and meridional dimensions,
            instead of just using a Fourier transform in the zonal dimension (default is False). Requires that the
            'pyshtools' python library be installed.
        longitude_name: string giving the textual name of the longitude coordinate (optional, default is "longitude").
        latitude_name: string giving the textual name of the latitude coordinate (optional, default is "latitude").
        pressure_level_name: string giving the textual name of the vertical coordinate on pressure levels (optional,
            default is "level").
        time_name: string giving the textual name of the time coordinate (optional, default is "time").
        scalar_field_name: string giving the textual name of the field from which to compute the spectra. If absent
            then compute the spectra for the kinetic energy as determined from the zonal and meridional velocity
            components.
        zonal_filter_width: ratio of the left and right domain size to the total zonal domain size over which to apply
            a filter to the input data in the case that the Fourier spectra is to be computed for a zonal sub-domain
            (default is 0.125).
        vertical_weights: numpy array of vertical weights for averaging over when reducing the vertical dimension.
            Only required (optionally) if reduce_vertical is True. If not supplied then it is assumed that the
            vertical coordinate is hydrostatic pressure.
        constants: class containing the planetary constants used to specify the geometry and thermodynamics (optional,
            will instantiate a version of the planet_constants class with default values if not supplied).

    Returns:
        ds: an Xarray Dataset containing the square of the amplitudes of the input data transformed into frequency
            space, and the corresponding frequencies.
    """

    _data = data.copy(deep=True)

    # average over the vertical dimension (pressure levels)
    if reduce_vertical and len(_data.level.values) > 1:
        if vertical_weights is not None:
            if len(vertical_weights) != _data.sizes[pressure_level_name]:
                error_msg = IndexError(
                    f"Length of supplied vertical weights array: {len(vertical_weights)} does not match length of",
                    +"{pressure_level_name}: {_data.sizes[pressure_level_name]}.",
                )
                raise error_msg
        else:
            dp = _pressure_level_thickness(_data.level.values, constants)
            vertical_weights = dp / np.sum(dp)

        dp_x = xr.DataArray(vertical_weights, dims=(pressure_level_name))
        _data = (dp_x * _data).sum(dim=pressure_level_name)

    # average over the time dimension
    nt = len(data.time.values)
    if reduce_time:
        _data = _data.sum(dim=time_name) / nt

    if spherical_harmonic:
        error_msg = NotImplementedError(
            "Support for spherical harmonic transform spectra has not been implemented yet."
        )
        raise error_msg

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

        if dask != "Unavailable":  # pragma: no cover
            _data = _data.compute()

        error_msg = ValueError(f"NaN value found in field: {scalar_field_name}.")
        if np.any(np.isnan(data[scalar_field_name].as_numpy())):
            raise error_msg

        lon_index = _data[scalar_field_name].get_axis_num(longitude_name)
        fft_lon = xr.apply_ufunc(
            _scaled_rfft,
            _data[scalar_field_name],
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


def power_spectra_vector(
    data: xr.DataArray,
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
    zonal_filter_width: float = 0.125,
    vertical_weights: np.ndarray | None = None,
    constants: PlanetConstants = STANDARD_CONSTANTS,
) -> XarrayLike:
    """
    Compute the spectra for a vector field.

    Args:
        data: Xarray DataArray containing the field to transform (maybe derived from the zonal and meridional velocity
            components, or from a custom field).
        reduce_vertical: average over the vertical pressure levels (default is False).
        reduce_time: average over the time levels (default is False).
        spherical_harmonic: compute spectra using spherical harmonics in both the zonal and meridional dimensions,
            instead of just using a Fourier transform in the zonal dimension (default is False). Requires that the
            'pyshtools' python library be installed.
        longitude_name: string giving the textual name of the longitude coordinate (optional, default is "longitude").
        latitude_name: string giving the textual name of the latitude coordinate (optional, default is "latitude").
        pressure_level_name: string giving the textual name of the vertical coordinate on pressure levels (optional,
            default is "level").
        time_name: string giving the textual name of the time coordinate (optional, default is "time").
        zonal_velocity_name: string giving the textual name of the zonal velocity (optional, default is "u").
        meridional_velocity_name: string giving the textual name of the meridional velocity (optional, default is "v").
        zonal_filter_width: ratio of the left and right domain size to the total zonal domain size over which to apply
            a filter to the input data in the case that the Fourier spectra is to be computed for a zonal sub-domain
            (default is 0.125).
        vertical_weights: numpy array of vertical weights for averaging over when reducing the vertical dimension.
            Only required (optionally) if reduce_vertical is True. If not supplied then it is assumed that the
            vertical coordinate is hydrostatic pressure.
        constants: class containing the planetary constants used to specify the geometry and thermodynamics (optional,
            will instantiate a version of the planet_constants class with default values if not supplied).

    Returns:
        ds: an Xarray Dataset containing the square of the amplitudes of the input data transformed into frequency
            space, and the corresponding frequencies.
    """

    _data = data.copy(deep=True)

    # average over the vertical dimension (pressure levels)
    if reduce_vertical and len(_data.level.values) > 1:
        if vertical_weights is not None:
            if len(vertical_weights) != _data.sizes[pressure_level_name]:
                error_msg = IndexError(
                    f"Length of supplied vertical weights array: {len(vertical_weights)} does not match length of",
                    +"{pressure_level_name}: {_data.sizes[pressure_level_name]}.",
                )
                raise error_msg
        else:
            dp = _pressure_level_thickness(_data.level.values, constants)
            vertical_weights = dp / np.sum(dp)

        dp_x = xr.DataArray(vertical_weights, dims=(pressure_level_name))
        _data = (dp_x * _data).sum(dim=pressure_level_name)

    # average over the time dimension
    nt = len(data.time.values)
    if reduce_time:
        _data = _data.sum(dim=time_name) / nt

    if spherical_harmonic:
        error_msg = NotImplementedError(
            "Support for spherical harmonic transform spectra has not been implemented yet."
        )
        raise error_msg

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

        if dask != "Unavailable":  # pragma: no cover
            _data = _data.compute()

        error_msg = ValueError(f"NaN value found in field: {zonal_velocity_name}.")
        if np.any(np.isnan(data[zonal_velocity_name].as_numpy())):
            raise error_msg

        error_msg = ValueError(f"NaN value found in field: {meridional_velocity_name}.")
        if np.any(np.isnan(data[meridional_velocity_name].as_numpy())):
            raise error_msg

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
