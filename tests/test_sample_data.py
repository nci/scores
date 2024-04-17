"""
Contains unit tests for scores.sample_data
"""

import xarray as xr

import scores.sample_data as sd


def test_simple_forecast():
    """
    Very basic test that the functions return comparable things
    """
    fcst = sd.simple_forecast()
    obs = sd.simple_observations()

    assert len(fcst) == len(obs)


def test_simple_forecast_pandas():
    """
    Very basic test that the functions return comparable things
    """
    fcst = sd.simple_forecast_pandas()
    obs = sd.simple_observations_pandas()

    assert len(fcst) == len(obs)


def test_continuous_data():
    """tests sample_data.continuous_forecast and sample_data.continuous_observations"""
    obs_small = sd.continuous_observations(large_size=False)
    forecast_small = sd.continuous_forecast(large_size=False)
    forecast_small_lead_days = sd.continuous_forecast(large_size=False, lead_days=True)

    assert obs_small.shape == forecast_small.shape
    assert forecast_small_lead_days.size == 2 * forecast_small.size
    assert forecast_small_lead_days.dims == ("lead_time", "lat", "lon", "time")
    assert forecast_small.dims == ("lat", "lon", "time")
    assert isinstance(forecast_small, xr.DataArray)
    assert isinstance(obs_small, xr.DataArray)


def test_cdf_data():
    """tests sample_data.cdf_forecast and sample_data.obs_for_cdf_forecast"""
    obs = sd.cdf_observations()
    fcst = sd.cdf_forecast()
    fcst_lead_days = sd.cdf_forecast(lead_days=True)

    assert 2 * fcst.size == fcst_lead_days.size
    assert 100 * obs.size == fcst.size
    assert len(obs.dims) == 3
    assert len(fcst.dims) == 4
    assert len(fcst_lead_days.dims) == 5
    assert isinstance(obs, xr.DataArray)
    assert isinstance(fcst, xr.DataArray)
    assert isinstance(fcst_lead_days, xr.DataArray)
