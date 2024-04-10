import xarray as xr

import scores

# Provides a basic forecast data structure in three dimensions
simple_forecast = xr.DataArray(
    [
        [
            [0.9, 0.0, 5],
            [0.7, 1.4, 2.8],
            [0.4, 0.5, 2.3],
        ],
        [
            [1.9, 1.0, 1.5],
            [1.7, 2.4, 1.1],
            [1.4, 1.5, 3.3],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)

# Within 0.1 or 0.2 of the forecast in all cases except one
# Can be used to find some exact matches, and some close matches
simple_obs = xr.DataArray(
    [
        [
            [0.9, 0.0, 5],
            [0.7, 1.3, 2.7],
            [0.3, 0.4, 2.2],
        ],
        [
            [1.7, 1.2, 1.7],
            [1.7, 2.2, 3.9],
            [1.6, 1.2, 9.9],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)

# This truth table shows where the forecast and obs match precisely
exact_matches = xr.DataArray(
    [
        [
            [True, True, True],
            [True, False, False],
            [False, False, False],
        ],
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)

# This truth table shows where the forecast and obs match within 0.2
somewhat_near_matches = xr.DataArray(
    [
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ],
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)


def test_categorical_table():
    match = scores.categorical.EventThresholdOperator()
    table = match.make_table(simple_forecast, simple_obs, event_threshold=1.3)

    assert table.tp_count == 9
    assert table.tn_count == 6
    assert table.fp_count == 2
    assert table.fn_count == 1
    assert table.total_count == 18

    assert table.accuracy() == (9 + 6) / 18
    assert table.probability_of_detection() == 9 / (9 + 1)
    assert table.false_alarm_rate() == 2 / (2 + 6)

    # Smoke tests only
    assert table.frequency_bias() is not None
    assert table.hit_rate() is not None
    assert table.probability_of_detection() is not None
    assert table.success_ratio() is not None
    assert table.threat_score() is not None
    assert table.critical_success_index() is not None
    assert table.pierce_skill_score() is not None
    assert table.sensitivity() is not None
    assert table.specificity() is not None


def test_functional_interface_accuracy():
    event_operator = scores.categorical.EventThresholdOperator(default_event_threshold=1.3)
    accuracy = scores.categorical.accuracy(simple_forecast, simple_obs, event_operator=event_operator)
    assert accuracy == (9 + 6) / 18


def test_categorical_table_dims_handling():
    match = scores.categorical.EventThresholdOperator()
    table = match.make_table(simple_forecast, simple_obs, event_threshold=1.3)

    acc_withheight = table.accuracy(preserve_dims=["height"])
    assert acc_withheight.sel(height=10).sum().values.item() == 8 / 9
    assert acc_withheight.sel(height=20).sum().values.item() == 7 / 9

    # pod_withheight = table.probability_of_detection(preserve_dims=['height'])
    # assert pod_withheight.sel(height=10).sum().values.item() == 9 / (9+1)
    # assert pod_withheight.sel(height=20).sum().values.item() == 9 / (9+1)

    # far_withheight = table.false_alarm_rate(preserve_dims=['height'])
    # assert far_withheight is not None  # TODO: Add a value-asserting test


# def test_dask_if_available_categorical():
# 	'''
# 	A basic smoke test on a dask object. More rigorous exploration of dask
# 	is probably needed beyond this. Performance is not explored here, just
# 	compatibility.
# 	'''

# 	try:
# 		import dask  # noqa: F401
# 	except ImportError:
# 		pytest.skip("Dask not available on this system")

# 	fcst = simple_forecast.chunk()
# 	obs = simple_obs.chunk()

# 	match = scores.categorical.EventThresholdOperator()
# 	table = match.make_table(fcst, obs, event_threshold=1.3)

# 	assert isinstance(table.forecast_events, dask.array.Array)
# 	assert isinstance(table.tp, dask.array.Array)

# 	computed = table.forecast_events.compute()

# 	assert isinstance(computed.data, np.ndarray)
