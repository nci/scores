"""
Test functions for contingency tables
"""

import numpy as np
import pytest
import xarray as xr

import scores

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

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
    """
    Test the basic calculations of the contingency table
    """
    match = scores.categorical.ThresholdEventOperator()
    table = match.make_table(simple_forecast, simple_obs, event_threshold=1.3)
    counts = table.get_counts()

    assert counts["tp_count"] == 9
    assert counts["tn_count"] == 6
    assert counts["fp_count"] == 2
    assert counts["fn_count"] == 1
    assert counts["total_count"] == 18

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


def test_categorical_table_dims_handling():
    """
    Test that the transform function correctly allows dimensional transforms
    """
    match = scores.categorical.ThresholdEventOperator()
    table = match.make_table(simple_forecast, simple_obs, event_threshold=1.3)
    transformed = table.transform(preserve_dims=["height"])

    acc_withheight = transformed.accuracy()
    assert acc_withheight.sel(height=10).sum().values.item() == 8 / 9
    assert acc_withheight.sel(height=20).sum().values.item() == 7 / 9


def test_dask_if_available_categorical():
    """
    A basic smoke test on a dask object. More rigorous exploration of dask
    is probably needed beyond this. Performance is not explored here, just
    compatibility.
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run dask tests")  # pragma: no cover

    fcst = simple_forecast.chunk()
    obs = simple_obs.chunk()

    match = scores.categorical.ThresholdEventOperator()
    table = match.make_table(fcst, obs, event_threshold=1.3)

    # Assert things start life as dask types
    assert isinstance(table.forecast_events.data, dask.array.Array)
    assert isinstance(table.tp.data, dask.array.Array)

    # That can be computed to hold numpy data types
    computed = table.forecast_events.compute()
    assert isinstance(computed.data, np.ndarray)

    # And that transformed tables are built out of computed things
    simple_counts = table.transform().get_counts()
    assert isinstance(simple_counts["tp_count"].data, np.ndarray)

    # And that transformed things get the same numbers
    assert table.false_alarm_rate() == table.transform().false_alarm_rate()
