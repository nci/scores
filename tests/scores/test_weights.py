import numpy as np
import pandas as pd
import xarray as xr

import scores.continuous

ZERO = np.array([[1 for i in range(10)] for j in range(10)])



# Standard forecast and observed test data which is static and can be used
# across tests
np.random.seed(0)
lats = [50, 51, 52, 53]
lons = [30, 31, 32, 33]
fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
FCST_2D = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
OBS_2D = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[lats, lons])
IDENTITY = np.ones((4, 4))
ZEROS = np.zeros((4, 4))

# fcst_synth_2d = [np.Nan, 0, 1, 2, 3, 4] * 4
# obs_synth_2d = [np.Nan, np.Nan, 0, 1, 2, 3] * 4
# popn_density_synth_1980 = [0, 2, 4, 8, 10, 6] * 4
# popn_density_synth_2022 = [0, 4, 16, 64, 100, 36] * 4

# FCST_SYNTHETIC_2D = xr.DataArray(fcst_synth_2d, dims=["latitude", "longitude"], coords=[lats, lons])

# These scores will be tested for valid processing of weights
all_scores = [scores.continuous.mse, scores.continuous.mae]

def test_weights_identity():

    for score in all_scores:
    
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=IDENTITY)
        assert unweighted == weighted

def test_weights_zeros():

    for score in all_scores:
    
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=ZEROS)

        assert unweighted != weighted
        assert weighted.sum() == 0


def test_weights_latitude():
    '''
    Tests the use of latitude weightings, not the correctness
    '''
    # TODO: Write a correctness test for latitude weighting conversions to be specified by hand

    lat_weightings_values = scores.functions.create_latitude_weights(OBS_2D.latitude)

    for score in all_scores:
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=lat_weightings_values)
        assert unweighted != weighted


# def test_weights_multiple_dimensions():    

#     errors = ['identity', 'zeros']
#     identity = xr.DataArray(IDENTITY, dims=["latitude", "longitude"], coords=[lats, lons])
#     zeros = xr.DataArray(ZEROS, dims=["latitude", "longitude"], coords=[lats, lons])
#     complex_weights = xr.concat([identity, zeros], pd.Index(['identity', 'zeros', 'latitude_weights', 'station_weighted', 'population', 'masked_population'], name='errors'))

#     popn_density_1950_weights = xr.DataArray(IDENTITY, dims=["latitude", "longitude"], coords=[lats, lons])
#     popn_density_2022_weights = xr.DataArray(IDENTITY, dims=["latitude", "longitude"], coords=[lats, lons])

#     for score in all_scores:
#         unweighted = score(FCST_2D, OBS_2D)
#         weighted = score(FCST_2D, OBS_2D, weights=[popn_density_1950_weights, popn_density_2022_weights])
#         assert unweighted == weighted