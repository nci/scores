"""
Test functions for contingency tables
"""

import operator

import numpy as np
import pandas as pd
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

finleys_table = {
    "tp_count": xr.DataArray(28),
    "fp_count": xr.DataArray(72),
    "fn_count": xr.DataArray(23),
    "tn_count": xr.DataArray(2680),
    "total_count": xr.DataArray(2803),
}


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

simple_forecast_with_nan = xr.DataArray(
    [
        [
            [np.nan, 0.0, 5],
            [0.7, np.nan, 2.8],
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

simple_obs_with_nan = xr.DataArray(
    [
        [
            [0.9, 0.0, np.nan],
            [0.7, np.nan, 2.7],
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


def test_str_view():
    """
    Smoke test for string representation
    """
    match = scores.categorical.ThresholdEventOperator(default_op_fn=operator.gt)
    table = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.3)
    _stringrep = str(table)


def test_categorical_table():  # pylint: disable=too-many-statements
    """
    Test the basic calculations of the contingency table
    """
    match = scores.categorical.ThresholdEventOperator(default_event_threshold=1.3, default_op_fn=operator.gt)
    table = match.make_contingency_manager(simple_forecast, simple_obs)
    table2 = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.3, op_fn=operator.gt)
    counts = table.get_counts()
    actual_table = table.get_table()

    # Test event tables creation matches the stored tables
    fcst_events, obs_events = match.make_event_tables(simple_forecast, simple_obs)
    fcst_events2, obs_events2 = match.make_event_tables(
        simple_forecast, simple_obs, event_threshold=1.3, op_fn=operator.gt
    )
    xr.testing.assert_equal(fcst_events, table.fcst_events)
    xr.testing.assert_equal(fcst_events2, table.fcst_events)
    xr.testing.assert_equal(obs_events, table.obs_events)
    xr.testing.assert_equal(obs_events2, table.obs_events)
    xr.testing.assert_equal(table.fcst_events, table2.fcst_events)
    xr.testing.assert_equal(table.obs_events, table2.obs_events)

    # Confirm values in the contingency table are correct
    assert counts["tp_count"] == 9
    assert counts["tn_count"] == 6
    assert counts["fp_count"] == 2
    assert counts["fn_count"] == 1
    assert counts["total_count"] == 18

    # Confirm that the xarray table object has the correct counts too
    assert actual_table.sel(contingency="tp_count") == 9
    assert actual_table.sel(contingency="tn_count") == 6
    assert actual_table.sel(contingency="fp_count") == 2
    assert actual_table.sel(contingency="fn_count") == 1
    assert actual_table.sel(contingency="total_count") == 18

    # Confirm calculations of metrics are correct
    assert table.base_rate() == (9 + 1) / 18
    assert table.forecast_rate() == (9 + 2) / 18
    assert table.accuracy() == (9 + 6) / 18
    assert table.probability_of_detection() == 9 / (9 + 1)
    assert table.false_alarm_rate() == 2 / (2 + 6)
    assert table.threat_score() == 9 / (9 + 2 + 1)
    assert table.frequency_bias() == (9 + 2) / (9 + 1)
    assert table.hit_rate() == 9 / (9 + 1)
    assert table.probability_of_false_detection() == 2 / (6 + 2)
    assert table.success_ratio() == 9 / (9 + 2)
    assert table.negative_predictive_value() == 6 / (6 + 1)
    assert table.specificity() == 6 / (6 + 2)
    assert table.false_alarm_ratio() == 2 / (9 + 2)
    assert table.f1_score() == 18 / (18 + 2 + 1)
    assert table.equitable_threat_score() == (9 - (55 / 9)) / (9 + 1 + 2 - (55 / 9))
    assert table.heidke_skill_score() == (9 + 6 - (83 / 9)) / (18 - (83 / 9))
    assert table.odds_ratio() == (9 / (9 + 1)) / (1 - (9 / (9 + 1))) / ((2 / (6 + 2)) / (1 - (2 / (6 + 2))))
    assert table.odds_ratio_skill_score() == (9 * 6 - 1 * 2) / (9 * 6 + 1 * 2)
    assert table.symmetric_extremal_dependence_index() == (np.log(0.25) - np.log(0.9) + np.log(0.1) - np.log(0.75)) / (
        np.log(0.25) + np.log(0.9) + np.log(0.1) + np.log(0.75)
    )

    # These methods are redirects to each other
    assert table.critical_success_index() == table.threat_score()
    assert table.hit_rate() == table.probability_of_detection()
    assert table.true_positive_rate() == table.probability_of_detection()
    assert table.sensitivity() == table.hit_rate()
    assert table.false_alarm_rate() == table.probability_of_false_detection()
    assert table.frequency_bias() == table.bias_score()
    assert table.peirce_skill_score() == table.true_skill_statistic()
    assert table.hanssen_and_kuipers_discriminant() == table.peirce_skill_score()
    assert table.precision() == table.success_ratio()
    assert table.positive_predictive_value() == table.success_ratio()
    assert table.recall() == table.probability_of_detection()
    assert table.gilberts_skill_score() == table.equitable_threat_score()
    assert table.cohens_kappa() == table.heidke_skill_score()
    assert table.yules_q() == table.odds_ratio_skill_score()
    assert table.accuracy() == table.fraction_correct()
    assert table.specificity() == table.true_negative_rate()

    peirce_component_a = 9 / (9 + 1)
    peirce_component_b = 2 / (2 + 6)
    peirce_expected = peirce_component_a - peirce_component_b
    assert table.peirce_skill_score() == peirce_expected


def test_dimension_broadcasting():
    """
    Confirm that dimension broadcasting is working, through an example where a time
    dimension is introduced to the forecast object. Each element of the time dimension
    should be compared to the observation, and the final accuracy should have an
    accuracy score at each time.

    In this example, the same forecast values are repeated for each of 5 lead times
    A single observations array is provided
    Broadcasting rules mean the accuracy should be calculated at each lead time,
    comparing the observations against each lead time
    """

    base_forecasts = [simple_forecast + i * 0.5 for i in range(5)]
    complex_forecast = xr.concat(base_forecasts, dim="time")
    complex_obs = xr.concat([simple_obs, simple_obs + 1], dim="source")
    match = scores.categorical.ThresholdEventOperator(default_event_threshold=1.3, default_op_fn=operator.gt)

    # Check dimension broadcasting for forecasts
    table = match.make_contingency_manager(complex_forecast, simple_obs)
    withtime = table.transform(preserve_dims="time")
    accuracy = withtime.accuracy()
    assert accuracy.dims == ("time",)
    assert len(accuracy.time) == 5

    assert accuracy[0] == (9 + 6) / 18
    assert accuracy[1] == (10 + 4) / 18
    assert accuracy[2] == (10 + 1) / 18
    assert accuracy[3] == (10 + 0) / 18
    assert accuracy[4] == (10 + 0) / 18

    # Check dimension broadcasting against observations
    table = match.make_contingency_manager(simple_forecast, complex_obs)
    withsource = table.transform(preserve_dims="source")
    accuracy = withsource.accuracy()
    assert accuracy.dims == ("source",)
    assert len(accuracy.source) == 2

    # Check dimension broadcasting against forecasts and observations together
    table = match.make_contingency_manager(complex_forecast, complex_obs)
    preserved = table.transform(preserve_dims=["time", "source"])
    accuracy_time_source = preserved.accuracy()
    assert accuracy_time_source.dims == ("time", "source")
    assert len(accuracy_time_source.time) == 5
    assert len(accuracy_time_source.source) == 2

    # The first "source" should match the previous calculations
    xr.testing.assert_allclose(accuracy_time_source.sel(source=0), withtime.accuracy())


def test_nan_handling():
    """
    This is important because the default handling of NaN with regards to operators
    is not suitable for calculating contingency scores. The strategy in `scores`
    is to disregard any case with a NaN in it, regardless of whether that is present
    in the forecast data or the observed data or both. There may be specific use cases
    where for example a nan could be regarded as a 'missed forecast' and should count.
    This situation is not regarded as 'standard', and only fully valid data is considered
    and aggregated in the scores.

    This test sets up data which has a nan forecast matched to a valid observation, a
    nan observation is matched to a valid forecast, and a nan forecast is matched to a
    nan observation.
    """

    match = scores.categorical.ThresholdEventOperator(default_event_threshold=1.3, default_op_fn=operator.gt)
    table = match.make_contingency_manager(simple_forecast_with_nan, simple_obs_with_nan)
    counts = table.get_counts()

    # Confirm values in the contingency table are correct
    assert counts["tp_count"] == 8
    assert counts["tn_count"] == 5
    assert counts["fp_count"] == 1
    assert counts["fn_count"] == 1
    assert counts["total_count"] == 15

    # Confirm calculation of a score is correct
    assert table.accuracy() == (8 + 5) / 15


def test_threshold_variation():
    """
    Some basic tests to ensure the event operator is responding correctly to
    variations in event thredhold which should override the default
    """

    match = scores.categorical.ThresholdEventOperator(default_event_threshold=0.5, default_op_fn=operator.gt)
    table_05 = match.make_contingency_manager(simple_forecast, simple_obs)
    table_13 = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.3)
    table_15 = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.5)

    # Consistent with generally used examples from other tests
    counts_13 = table_13.get_counts()
    assert counts_13["tp_count"] == 9
    assert counts_13["tn_count"] == 6
    assert counts_13["fp_count"] == 2
    assert counts_13["fn_count"] == 1
    assert counts_13["total_count"] == 18

    counts_05 = table_05.get_counts()
    counts_15 = table_15.get_counts()
    assert counts_05["tp_count"] != counts_13["tp_count"]
    assert counts_05["tp_count"] != counts_15["tp_count"]
    assert counts_05["tp_count"] == 15
    assert counts_15["tp_count"] == 7


def test_categorical_table_dims_handling():
    """
    Test that the transform function correctly allows dimensional transforms
    """
    match = scores.categorical.ThresholdEventOperator(default_op_fn=operator.gt)
    table = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.3)
    transformed = table.transform(preserve_dims=["height"])
    transformed2 = table.transform(reduce_dims=["lat", "lon"])

    # Assert preserving and reducing are being handled consistently
    xr.testing.assert_equal(transformed.get_table(), transformed2.get_table())

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

    match = scores.categorical.ThresholdEventOperator(default_op_fn=operator.gt)
    table = match.make_contingency_manager(fcst, obs, event_threshold=1.3)

    # Assert things start life as dask types
    assert isinstance(table.fcst_events.data, dask.array.Array)
    assert isinstance(table.tp.data, dask.array.Array)

    # That can be computed to hold numpy data types
    computed = table.fcst_events.compute()
    assert isinstance(computed.data, np.ndarray)

    # And that transformed tables are built out of computed things
    simple_counts = table.transform().get_counts()
    assert isinstance(simple_counts["tp_count"].data, (np.ndarray, np.generic))

    # And that transformed things get the same numbers
    assert table.false_alarm_rate() == table.transform().false_alarm_rate()


def test_examples_with_finley():
    """
    Test some of the complex scores with the Finley tornado example
    """

    table = finleys_table
    cm = scores.categorical.BasicContingencyManager(table)
    heidke = cm.heidke_skill_score()
    gilbert = cm.gilberts_skill_score()

    # Note - the reference in the verification site has 0.36 for the expected
    # result, but presumably this is rounded to two decimal places
    # See https://www.cawcr.gov.au/projects/verification/Finley/Finley_Tornados.html
    heidke_expected = xr.DataArray(0.355325)
    xr.testing.assert_allclose(heidke, heidke_expected)

    # Note - the reference in the verification site has 0.22 for the expected
    # result, but presumably this is rounded to two decimal places
    # See https://www.cawcr.gov.au/projects/verification/Finley/Finley_Tornados.html
    gilbert_expected = xr.DataArray(0.216046)
    xr.testing.assert_allclose(gilbert_expected, gilbert)


def test_format_data():
    """
    Test the format table method.
    """

    # Simple 2x2 Example

    match = scores.categorical.ThresholdEventOperator(default_op_fn=operator.gt)
    table = match.make_contingency_manager(simple_forecast, simple_obs, event_threshold=1.3)
    expected_df = pd.DataFrame(
        {"Positive Observed": [9, 1, 10], "Negative Observed": [2, 6, 8], "Total": [11, 7, 18]},
        index=["Positive Forecast", "Negative Forecast", "Total"],
    )

    result_df = table.format_table()
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Higher-dimension handling example
    table = table.transform(preserve_dims="height")
    with pytest.warns(UserWarning) as warning_object:
        format_table = table.format_table()

    assert scores.categorical.contingency_impl.HIGH_DIMENSION_HTML_CONTINGENCY_WARNING in str(
        warning_object.list[0].message
    )
    get_table = table.get_table()
    xr.testing.assert_equal(format_table, get_table)
