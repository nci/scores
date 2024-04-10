import xarray as xr
import numpy as np
import pytest
import scores

from tests.assertions import assert_dataarray_equal

# Provides a basic forecast data structure in three dimensions
simple_forecast = xr.DataArray(
    [
		[
			[0.9, 0.0,   5], 
			[0.7, 1.4, 2.8],
			[.4,  0.5, 2.3],
		], 
			[
			[1.9, 1.0,  1.5], 
			[1.7, 2.4,  1.1],
			[1.4,  1.5, 3.3],
		], 
	],
	coords=[[10, 20], [0, 1, 2], [5, 6, 7]], dims=["height", "lat", "lon"])

# Within 0.1 or 0.2 of the forecast in all cases except one
# Can be used to find some exact matches, and some close matches
simple_obs = xr.DataArray(
    [
		[
			[0.9, 0.0,   5], 
			[0.7, 1.3, 2.7],
			[.3,  0.4, 2.2],
		], 
			[
			[1.7, 1.2,  1.7], 
			[1.7, 2.2,  3.9],
			[1.6,  1.2, 9.9],
		], 
	],
	coords=[[10, 20], [0, 1, 2], [5, 6, 7]], dims=["height", "lat", "lon"])

# This truth table shows where the forecast and obs match precisely
exact_matches = xr.DataArray(
    [
		[
			[True,  True,  True], 
			[True,  False, False],
			[False, False, False],
		], 
			[
			[False, False, False], 
			[True,  False, False],
			[False, False, False],
		], 
	],
	coords=[[10, 20], [0, 1, 2], [5, 6, 7]], dims=["height", "lat", "lon"])

# This truth table shows where the forecast and obs match within 0.2
somewhat_near_matches = xr.DataArray(
    [
		[
			[True,  True,  True], 
			[True,  True,  True],
			[True,  True,  True],
		], 
			[
			[True, True, True], 
			[True, True, False],
			[True, False, False],
		], 
	],
	coords=[[10, 20], [0, 1, 2], [5, 6, 7]], dims=["height", "lat", "lon"])


def test_simple_binary_table():

	proximity = scores.continuous.mae(simple_forecast, simple_obs, preserve_dims="all")
	match = scores.categorical.BinaryProximityOperator()
	found = match.make_table(proximity)
	assert found.table.equals(exact_matches)
	assert found.hits() == 5

	# Ideally would check the nature of this failure but for now a non-match will do
	with pytest.raises(AssertionError):
		assert_dataarray_equal(found.table, somewhat_near_matches)


def test_nearby_binary_table():

	proximity = scores.continuous.mae(simple_forecast, simple_obs, preserve_dims="all")
	matchpoint2 = scores.categorical.BinaryProximityOperator(tolerance=0.2)	
	found = matchpoint2.make_table(proximity)

	assert_dataarray_equal(found.table, somewhat_near_matches)

	# Ideally would check the nature of this failure but for now a non-match will do
	with pytest.raises(AssertionError):
		assert_dataarray_equal(found.table, exact_matches)		


def test_categorical_table():

	match = scores.categorical.EventThresholdOperator()
	table = match.make_table(simple_forecast, simple_obs, 1.3)

	assert table.tp_count == 9
	assert table.tn_count == 6
	assert table.fp_count == 2	
	assert table.fn_count == 1
	assert table.total_count == 18

	assert table.accuracy() == (9 + 6) / 18
	assert table.probability_of_detection() == 9 / (9+1)
	assert table.false_alarm_rate() == 2 / (2 + 6)


def test_dask_if_available():
	'''
	A basic smoke test on a dask object. More rigorous exploration of dask
	is probably needed beyond this. Performance is not explored here, just
	compatibility.
	'''

	try:
		import dask  # noqa: F401
	except ImportError:
		pytest.skip("Dask not available on this system")

	proximity = scores.continuous.mae(simple_forecast, simple_obs, preserve_dims="all")
	match = scores.categorical.BinaryProximityOperator()
	dprox = proximity.chunk()
	table = match.make_table(dprox).table
	assert_dataarray_equal(table, exact_matches)

	assert isinstance(table.data, dask.array.Array)
	computed = table.compute()

	assert isinstance(computed.data, np.ndarray)
	assert_dataarray_equal(computed, exact_matches)