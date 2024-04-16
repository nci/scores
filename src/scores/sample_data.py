"""
Module to generate simple sample data for users (not tests). Supports tutorials and demos.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import skewnorm  # type: ignore


def simple_forecast() -> xr.DataArray:
    """Generate a simple series of prediction values"""
    return xr.DataArray(data=[10, 10, 11, 13, 14, 17, 15, 14])


def simple_observations() -> xr.DataArray:
    """Generate a simple series of observation values"""
    return xr.DataArray(data=[11, 11, 12, 14, 11, 14, 12, 11])


def simple_forecast_pandas() -> pd.Series:
    """Generate a simple series of prediction values"""
    return pd.Series([10, 10, 11, 13, 14, 17, 15, 14])


def simple_observations_pandas() -> pd.Series:
    """Generate a simple series of observation values"""
    return pd.Series([11, 11, 12, 14, 11, 14, 12, 11])


def continuous_observations(*, large_size: bool = False) -> xr.DataArray:
    """Creates a obs array with continuous values.

    Args:
        large_size (bool): If True, then returns a large global array with ~0.5 degree
            grid spacing, otherwise returns a cut down, lower resolution array.

    Returns:
        xr.Datarray: Containing synthetic observation data.
    """

    num_lats = 10
    num_lons = 20
    periods = 10

    if large_size:  # pragma: no cover
        num_lats = 364  # pragma: no cover - used in notebooks and tested manually
        num_lons = 720  # pragma: no cover - used in notebooks and tested manually
        periods = 240  # pragma: no cover - used in notebooks and tested manually

    lat = np.linspace(-90, 90, num_lats)
    lon = np.linspace(0, 360, num_lons)
    time_series = pd.date_range(
        start="2022-11-20T01:00:00.000000000",
        freq="h",
        periods=periods,
    )

    np.random.seed(42)
    data = 10 * np.random.rand(len(lat), len(lon), len(time_series))
    obs = xr.DataArray(coords={"lat": lat, "lon": lon, "time": time_series}, data=data)

    return obs


def continuous_forecast(*, large_size: bool = False, lead_days: bool = False) -> xr.DataArray:
    """Creates a forecast array with continuous values.

    Args:
        large_size (bool): If True, then returns a large global array with ~0.5 degree
            grid spacing, otherwise returns a cut down, lower resolution array.
        lead_days (bool): If True, returns an array with a "lead_day" dimension.

    Returns:
        xr.Datarray: Containing synthetic forecast data.
    """
    obs = continuous_observations(large_size=large_size)
    np.random.seed(42)
    forecast = obs + np.random.normal(0, 2, obs.shape)
    if lead_days:
        forecast2 = obs + np.random.normal(0, 3, obs.shape)
        forecast = xr.concat([forecast, forecast2], dim="lead_time")
        forecast = forecast.assign_coords(lead_time=[1, 2])
    return forecast


def cdf_forecast(*, lead_days: bool = False) -> xr.DataArray:
    """
    Creates a forecast array with a CDF at each point.

    Args:
        lead_days (bool): If True, returns an array with a "lead_day" dimension.

    Returns:
        xr.Datarray: Containing synthetic CDF forecast data.
    """
    x = np.arange(0, 10, 0.1)
    cdf_list = []

    if lead_days:
        for _ in np.arange(0, 16):
            cdf_list.append(skewnorm.cdf(x, a=10, loc=2, scale=2))
        cdfs = np.reshape(cdf_list, (2, 2, 2, 2, 100))
        forecast = xr.DataArray(
            coords={
                "x": [10, 20],
                "y": [30, 40],
                "time": [10, 20],
                "lead_day": pd.date_range("2022-01-01", "2022-01-02"),
                "threshold": x,
            },
            data=cdfs,
        )
    else:
        for _ in np.arange(0, 8):
            cdf_list.append(skewnorm.cdf(x, a=10, loc=2, scale=2))
        cdfs = np.reshape(cdf_list, (2, 2, 2, 100))
        forecast = xr.DataArray(
            coords={
                "x": [10, 20],
                "y": [30, 40],
                "time": [10, 20],
                "threshold": x,
            },
            data=cdfs,
        )

    return forecast


def cdf_observations() -> xr.DataArray:
    """
    Creates an obs array to use with `cdf_forecast`.

    Returns:
        xr.Datarray: Containing synthetic observations betwen 0 and 9.9
    """
    np.random.seed(42)
    obs = xr.DataArray(
        coords={
            "x": [10, 20],
            "y": [30, 40],
            "time": [10, 20],
        },
        data=10 * np.random.uniform(high=9.9, size=(2, 2, 2)),
    )
    return obs
