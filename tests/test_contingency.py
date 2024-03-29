import xarray as xr
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
			[1.7, 2.4,  3.8],
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
			[True, True, True],
			[True, False, False],
		], 
	],
	coords=[[10, 20], [0, 1, 2], [5, 6, 7]], dims=["height", "lat", "lon"])


def test_simple_binary_table():
	
	proximity = scores.continuous.mae(simple_forecast, simple_obs, preserve_dims="all")
	match = scores.contingency.BinaryProximityOperator()
	table = match.make_table(proximity)
	assert_dataarray_equal(table, exact_matches)

	# Ideally would check the nature of this failure but for now a non-match will do
	with pytest.raises(AssertionError):
		assert_dataarray_equal(table, somewhat_near_matches)


def test_nearby_binary_table():

	# import pudb; pudb.set_trace()
	
	proximity = scores.continuous.mae(simple_forecast, simple_obs, preserve_dims="all")
	matchpoint2 = scores.contingency.BinaryProximityOperator(tolerance=0.2)	
	table2 = matchpoint2.make_table(proximity)

	assert_dataarray_equal(table2, somewhat_near_matches)

	# Ideally would check the nature of this failure but for now a non-match will do
	with pytest.raises(AssertionError):
		assert_dataarray_equal(table2, exact_matches)

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
	match = scores.contingency.BinaryProximityOperator()
	dprox = proximity.chunk()
	table = match.make_table(dprox)
	assert_dataarray_equal(table, exact_matches)
