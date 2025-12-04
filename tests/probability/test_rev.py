# pylint: disable=too-many-lines
"""
Contains unit tests for scores.probability.rev_impl
"""

try:
    import dask.array as da

    HAS_DASK = True  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover
    from dask.base import is_dask_collection
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    HAS_DASK = False  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover


import re
import sys
import warnings
from collections import OrderedDict
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from scores.probability import (
    relative_economic_value,
    relative_economic_value_from_rates,
)
from scores.probability.rev_impl import (
    _calculate_rev_core,
    _create_output_dataset,
    _validate_rev_inputs,
    calculate_climatology,
    check_monotonic_array,
)
from scores.utils import ERROR_INVALID_WEIGHTS

# Module-level test data ... Preserved as is because these come from Jive and we want to demonstrate
# that they've been carried over exactly.
SCALAR_DA = xr.DataArray(np.array([0.5]), dims=("x",), coords={"x": [0]})
BINARY_DA = xr.DataArray([0, 1, 0], dims=["time"])
PROB_FCST_DA = xr.DataArray([0.2, 0.8, 0.1], dims=["time"])

THRESHOLDS = [0, 0.3, 1]
CL_RATIOS = [0, 0.2, 0.5, 0.8, 1]

POD_LEADDAY = xr.DataArray(
    [[1, 0.75, 0.25], [1, 0.25, 0]],
    dims=["lead_day", "threshold"],
    coords=OrderedDict([("lead_day", [0, 1]), ("threshold", THRESHOLDS)]),
)
POFD_LEADDAY = xr.DataArray(
    [[1, 1 / 3, 0], [1, 0.75, 0]],
    dims=["lead_day", "threshold"],
    coords=OrderedDict([("lead_day", [0, 1]), ("threshold", THRESHOLDS)]),
)
CLIMATOLOGY_LEADDAY = xr.DataArray([4 / 7, 0.5], dims=["lead_day"], coords={"lead_day": [0, 1]})

HIT_RATE_REV_NONE = xr.DataArray([1, 0.5, 1 / 8], dims=["threshold"], coords={"threshold": THRESHOLDS})
FALSE_ALARM_RATE_REV_NONE = xr.DataArray([1, 4 / 7, 0], dims=["threshold"], coords={"threshold": THRESHOLDS})
OBAR_REV_NONE = xr.DataArray(8 / 15)

EXP_REV_CASE0 = xr.DataArray(
    np.array(
        [
            [
                [np.nan, 0, 0, -2, np.nan],
                [np.nan, -2 / 3, 1 / 3, -0.25, np.nan],
                [np.nan, -3, 0, 0.25, np.nan],
            ],
            [
                [np.nan, 0, 0, -3, np.nan],
                [np.nan, -2.75, -0.5, -2.75, np.nan],
                [np.nan, -3, 0, 0, np.nan],
            ],
        ]
    ).transpose(0, 2, 1),
    dims=["lead_day", "cost_loss_ratio", "threshold"],
    coords=OrderedDict(
        [
            ("lead_day", [0, 1]),
            ("cost_loss_ratio", CL_RATIOS),
            ("threshold", THRESHOLDS),
        ]
    ),
)

EXP_REV_CASE1 = xr.DataArray(
    np.array(
        [
            [np.nan, 0, 0, -2.5, np.nan],
            [np.nan, -13 / 7, -1 / 7, -1.5, np.nan],
            [np.nan, -3, 0, 0.125, np.nan],
        ]
    ).T,
    dims=["cost_loss_ratio", "threshold"],
    coords=OrderedDict([("cost_loss_ratio", CL_RATIOS), ("threshold", THRESHOLDS)]),
)


@pytest.fixture(name="make_contingency_data")
def _make_contingency_data():
    """
    Factory to create fcst/obs DataArrays from contingency table counts.

    Returns a function that takes (hits, misses, false_alarms, correct_negatives)
    and returns (fcst, obs) DataArrays.
    """

    def _make(hits, misses, false_alarms, correct_negatives):
        # fcst=1 for hits and false_alarms, fcst=0 for misses and correct_negatives
        # obs=1 for hits and misses, obs=0 for false_alarms and correct_negatives
        fcst = [1] * hits + [0] * misses + [1] * false_alarms + [0] * correct_negatives
        obs = [1] * hits + [1] * misses + [0] * false_alarms + [0] * correct_negatives
        return (xr.DataArray(fcst, dims=["time"]), xr.DataArray(obs, dims=["time"]))

    return _make


class TestBroadcastingAndDimensionHandling:
    """Tests for broadcasting and dimension handling in REV calculations."""

    @pytest.mark.parametrize(
        "fcst_dims,obs_dims,fcst_data,obs_data,expected_rev",
        [
            # obs missing space dimension - broadcasts over space, reduce over time
            (
                ["time", "space"],
                ["time"],
                [
                    [1, 0, 1, 1, 0],  # time=0, space=[0,1,2,3,4]
                    [1, 1, 0, 1, 1],  # time=1
                    [0, 1, 1, 0, 1],  # time=2
                    [1, 0, 1, 1, 0],
                ],  # time=3
                [1, 1, 0, 1],  # time=[0,1,2,3], broadcasts to all space
                [[1.0], [-2.0], [-1.0], [1.0], [-2.0]],  # REV for each space point after reducing time
            ),
            # fcst missing space dimension - broadcasts over space, reduce over time
            (
                ["time"],
                ["time", "space"],
                [1, 1, 0, 1],  # time=[0,1,2,3], broadcasts to all space
                [
                    [1, 0, 1, 1, 0],  # time=0, space=[0,1,2,3,4]
                    [1, 1, 0, 1, 1],  # time=1
                    [0, 1, 1, 0, 1],  # time=2
                    [1, 0, 1, 1, 0],
                ],  # time=3
                [[1.0], [-0.5], [-1], [1.0], [-0.5]],  # REV for each space point after reducing time
            ),
        ],
        ids=["obs_missing_space", "fcst_missing_space"],
    )
    def test_broadcasting_reducing(self, fcst_dims, obs_dims, fcst_data, obs_data, expected_rev):
        """Test that broadcasting works when reducing over dimensions."""
        time_size = 4
        space_size = 5

        time_coord = list(range(time_size))
        space_coord = list(range(space_size))

        fcst_coords = {d: time_coord if d == "time" else space_coord for d in fcst_dims}
        obs_coords = {d: time_coord if d == "time" else space_coord for d in obs_dims}

        fcst = xr.DataArray(fcst_data, dims=fcst_dims, coords=fcst_coords)
        obs = xr.DataArray(obs_data, dims=obs_dims, coords=obs_coords)

        actual = relative_economic_value(fcst, obs, [0.5], reduce_dims="time")

        expected = xr.DataArray(
            expected_rev, dims=["space", "cost_loss_ratio"], coords={"space": space_coord, "cost_loss_ratio": [0.5]}
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        "fcst_dims,obs_dims,fcst_data,obs_data,expected_rev",
        [
            # obs missing space dimension - broadcasts over space
            (
                ["time", "space"],
                ["time"],
                [[1, 0, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 0, 1]],  # time=0, space=[0,1,2,3,4]  # time=1  # time=2
                [1, 1, 0],  # time=[0,1,2]
                [[1.0], [-1.0], [-1.0], [1.0], [-1.0]],  # REV for each space point
            ),
            # fcst missing space dimension - broadcasts over space
            (
                ["time"],
                ["time", "space"],
                [1, 1, 0],  # time=[0,1,2]
                [[1, 0, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 0, 1]],  # time=0, space=[0,1,2,3,4]  # time=1  # time=2
                [[1.0], [-1.0], [-1.0], [1.0], [-1.0]],  # REV for each space point
            ),
        ],
        ids=["obs_missing_space", "fcst_missing_space"],
    )
    def test_broadcasting_keeping(self, fcst_dims, obs_dims, fcst_data, obs_data, expected_rev):
        """Test that broadcasting works correctly when preserving dimensions."""
        time_coord = [0, 1, 2]
        space_coord = [0, 1, 2, 3, 4]

        fcst_coords = {d: time_coord if d == "time" else space_coord for d in fcst_dims}
        obs_coords = {d: time_coord if d == "time" else space_coord for d in obs_dims}

        fcst = xr.DataArray(fcst_data, dims=fcst_dims, coords=fcst_coords)
        obs = xr.DataArray(obs_data, dims=obs_dims, coords=obs_coords)

        actual = relative_economic_value(fcst, obs, [0.5], preserve_dims="space")

        expected = xr.DataArray(
            expected_rev, dims=["space", "cost_loss_ratio"], coords={"space": space_coord, "cost_loss_ratio": [0.5]}
        )

        xr.testing.assert_allclose(actual, expected)

    def test_reduce_preserve_dims_basic(self):
        """Test that REV respects reduce_dims and preserve_dims."""
        # Simple, explicit data
        fcst = xr.DataArray([[[1, 0], [1, 1]], [[0, 1], [1, 0]]], dims=["time", "lat", "lon"])
        obs = xr.DataArray([[[1, 1], [0, 1]], [[0, 0], [1, 1]]], dims=["time", "lat", "lon"])

        # Test reduce_dims
        result = relative_economic_value(fcst, obs, [0.5], reduce_dims=["time"])
        assert set(result.dims) == {"lat", "lon", "cost_loss_ratio"}

        # Test preserve_dims
        result = relative_economic_value(fcst, obs, [0.5], preserve_dims=["time"])
        assert set(result.dims) == {"time", "cost_loss_ratio"}

        # Test reduce all (default)
        result = relative_economic_value(fcst, obs, [0.5])
        assert set(result.dims) == {"cost_loss_ratio"}


class TestCalculateClimatology:
    """Tests for climatology calculation."""

    def test_simple_mean(self):
        """Test basic mean calculation."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        actual = calculate_climatology(obs)
        expected = xr.DataArray(data=0.5)
        xr.testing.assert_allclose(expected, actual)

    def test_with_weights_spanning_all_dims(self):
        """Test weighted mean when weights span all reduction dims."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, 2, 1, 2], dims=["time"])

        actual = calculate_climatology(obs, weights=weights)
        expected_val = (0 * 1 + 1 * 2 + 0 * 1 + 1 * 2) / (1 + 2 + 1 + 2)
        expected = xr.DataArray(expected_val)

        xr.testing.assert_allclose(actual, expected)

    def test_climatology_ignores_weights_when_dims_not_matching(self):
        """Test that weights are ignored when their dims don't match reduce_dims."""
        # Create obs with a dimension 'time'
        obs = xr.DataArray([1, 0, 1, 1], dims="time")

        # Create weights with a different dimension, 'lat'
        weights = xr.DataArray([0.1, 0.2], dims="lat")

        # Reduce over 'time' (default)
        actual = calculate_climatology(obs, weights=weights)

        # Should be simple mean of obs since weights dims don't match reduce_dims
        expected = xr.DataArray(3 / 4)

        xr.testing.assert_allclose(actual, expected)

    def test_multidimensional_with_partial_weights(self):
        """Test when weights only span some dims being reduced."""
        obs = xr.DataArray([[0, 1], [1, 0], [1, 1]], dims=["time", "space"])
        weights = xr.DataArray([1, 2], dims=["space"])

        # Reduce all dims
        actual = calculate_climatology(obs, reduce_dims=["time", "space"], weights=weights)

        # Weighted mean over space, then mean over time
        expected_val = np.mean([(0 * 1 + 1 * 2) / 3, (1 * 1 + 0 * 2) / 3, (1 * 1 + 1 * 2) / 3])
        expected = xr.DataArray(expected_val)

        xr.testing.assert_allclose(actual, expected)

    def test_preserve_dims(self):
        """Test preserve_dims parameter."""
        obs = xr.DataArray([[0, 1], [1, 0]], dims=["time", "space"])

        actual = calculate_climatology(obs, preserve_dims="space")
        expected = xr.DataArray([0.5, 0.5], dims=["space"])
        xr.testing.assert_allclose(actual, expected)


class TestScienceCalculations:
    """Tests for core scientific calculations in REV."""

    @pytest.mark.parametrize(
        "pod,pofd,climatology,expected",
        [
            (POD_LEADDAY, POFD_LEADDAY, CLIMATOLOGY_LEADDAY, EXP_REV_CASE0),
            (HIT_RATE_REV_NONE, FALSE_ALARM_RATE_REV_NONE, OBAR_REV_NONE, EXP_REV_CASE1),
        ],
        ids=["with_lead_day", "no_dimensions"],
    )
    def test_from_rates_jive(self, pod, pofd, climatology, expected):
        """Tests relative economic value from rates against known Jive outputs."""
        result = relative_economic_value_from_rates(pod, pofd, climatology, cost_loss_ratios=CL_RATIOS)
        xr.testing.assert_allclose(result, expected)

    def test_perfect_forecast(self, make_contingency_data):
        """Perfect forecast (all correct) has REV=1 at any cost-loss ratio."""
        fcst, obs = make_contingency_data(hits=2, misses=0, false_alarms=0, correct_negatives=2)

        for alpha in [0.2, 0.5, 0.8]:
            result = relative_economic_value(fcst, obs, [alpha])
            assert result.item() == pytest.approx(1.0)

    def test_always_no_forecast(self, make_contingency_data):
        """Always predicting 'no' gives REV=0 at alpha=obar, negative otherwise."""
        # 0 hits, 2 misses, 0 FA, 2 CN -> POD=0, POFD=0, obar=0.5
        fcst, obs = make_contingency_data(0, 2, 0, 2)

        assert relative_economic_value(fcst, obs, [0.5]).item() == 0.0
        assert relative_economic_value(fcst, obs, [0.2]).item() == pytest.approx(-3.0)

    def test_always_yes_forecast(self, make_contingency_data):
        """Always predicting 'yes' gives REV=0 at alpha=obar, negative otherwise."""
        # 2 hits, 0 misses, 2 FA, 0 CN -> POD=1, POFD=1, obar=0.5
        fcst, obs = make_contingency_data(2, 0, 2, 0)

        assert relative_economic_value(fcst, obs, [0.5]).item() == 0.0
        assert relative_economic_value(fcst, obs, [0.8]).item() == pytest.approx(-3.0)

    def test_anti_correlated_forecast(self):
        """Forecast that's systematically wrong has REV < -1."""
        # This doesn't fit contingency model - fcst and obs are opposite
        fcst = xr.DataArray([0, 1, 0, 1], dims=["time"])
        obs = xr.DataArray([1, 0, 1, 0], dims=["time"])

        # At alpha=0.5: REV = -1.0
        assert relative_economic_value(fcst, obs, [0.5]).item() == -1.0
        # At extreme alphas: even worse
        assert relative_economic_value(fcst, obs, [0.2]).item() == -4.0

    @pytest.mark.parametrize(
        "hits,misses,fa,cn,alpha,expected",
        [
            (1, 1, 2, 16, 0.2, 0.25),
            (7, 3, 8, 82, 0.2, 0.5),
            (9, 1, 6, 84, 0.2, 0.75),
        ],
        ids=["low_skill", "medium_skill", "high_skill"],
    )
    def test_partial_skill_cases(self, make_contingency_data, hits, misses, fa, cn, alpha, expected):
        """Test cases with varying levels of forecast skill."""
        fcst, obs = make_contingency_data(hits, misses, fa, cn)
        result = relative_economic_value(fcst, obs, [alpha])
        assert result.item() == pytest.approx(expected)

    def test_undefined_when_obar_is_zero_or_one(self, make_contingency_data):
        """REV undefined when climatology is 0 or 1 (no variance in obs)."""
        # obar = 0: no events
        fcst, obs = make_contingency_data(0, 0, 2, 2)
        assert np.isnan(relative_economic_value(fcst, obs, [0.2]).item())

        # obar = 1: all events
        fcst, obs = make_contingency_data(2, 2, 0, 0)
        assert np.isnan(relative_economic_value(fcst, obs, [0.2]).item())

    def test_undefined_at_extreme_cost_loss(self, make_contingency_data):
        """REV undefined at cost_loss_ratio = 0 or 1."""
        fcst, obs = make_contingency_data(2, 0, 0, 2)  # perfect forecast

        assert np.isnan(relative_economic_value(fcst, obs, [0.0]).item())
        assert np.isnan(relative_economic_value(fcst, obs, [1.0]).item())

    def test_nan_values_excluded_from_calculation(self):
        """NaN values in fcst or obs are excluded pairwise."""
        # After removing NaNs: fcst=[1,1,0,0], obs=[1,0,1,0]
        # -> H=1, M=1, FA=1, CN=1 -> no skill -> REV=0 at alpha=0.5
        fcst = xr.DataArray([1, 1, 0, 0, 1, np.nan], dims=["time"])
        obs = xr.DataArray([1, 0, 1, 0, np.nan, 0], dims=["time"])

        result = relative_economic_value(fcst, obs, [0.5])
        assert result.item() == 0.0

    def test_multiple_cost_loss_ratios(self, make_contingency_data):
        """Test with multiple cost-loss ratios spanning the full range."""
        # Simple, verifiable data:
        # 10 samples: 6 hits, 2 misses, 1 false alarm, 1 correct negative
        # POD = 6/8 = 0.75, POFD = 1/2 = 0.5, climatology = 8/10 = 0.8
        binary_fcst, obs = make_contingency_data(6, 2, 1, 1)

        cost_loss_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        result = _calculate_rev_core(
            binary_fcst=binary_fcst,
            obs=obs,
            cost_loss_ratios=cost_loss_ratios,
            dims_to_reduce="all",
            weights=None,
        )

        expected = xr.DataArray(
            [np.nan, -2.5, -0.5, 1 / 6, np.nan],
            dims=["cost_loss_ratio"],
            coords={"cost_loss_ratio": cost_loss_ratios},
        )

        xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("scalar_value", [0, 0.0, 1, 1.0])  # test a mix of int and float values
    def test_cost_loss_is_converted_to_length_one_coordinate(self, scalar_value):
        """
        Passing an integer cost_loss_ratio should be converted to a length-1 coordinate.
        Also covers int -> list handling branch.
        """
        fcst = BINARY_DA
        obs = fcst.copy()
        actual = relative_economic_value(fcst, obs, cost_loss_ratios=scalar_value)

        expected = xr.DataArray([np.nan], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [scalar_value]})

        xr.testing.assert_identical(actual, expected)


class TestREVSpecialFeatures:
    """Tests for special features of the REV implementation."""

    def test_probabilistic_single_threshold(self):
        """Test with single threshold"""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.1, 0.9], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0, 1], dims=["time"])
        threshold = 0.5
        cost_loss_ratios = [0.2, 0.5, 0.8]

        expected = relative_economic_value(
            fcst, obs, cost_loss_ratios, threshold=threshold, threshold_outputs=[threshold]
        )
        actual = xr.Dataset(
            data_vars={"threshold_0_5": (["cost_loss_ratio"], [1.0, 1.0, 1.0])},
            coords={"cost_loss_ratio": [0.2, 0.5, 0.8]},
        )
        xr.testing.assert_allclose(expected, actual)

    def test_probabilistic_threshold_outputs(self):
        """Test asking for a single threshold output, but multiple thresholds supplied"""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.1, 0.9], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0, 1], dims=["time"])
        thresholds = [0.3, 0.5, 0.7]
        cost_loss_ratios = [0.2, 0.5, 0.8]

        expected = relative_economic_value(fcst, obs, cost_loss_ratios, threshold=thresholds, threshold_outputs=[0.5])
        actual = xr.Dataset(
            data_vars={"threshold_0_5": (["cost_loss_ratio"], [1.0, 1.0, 1.0])},
            coords={"cost_loss_ratio": [0.2, 0.5, 0.8]},
        )
        xr.testing.assert_allclose(expected, actual)

    def test_threshold_outputs_multiple_values(self):
        """Test the threshold_outputs feature for multiple specific thresholds"""
        fcst = xr.DataArray([0.1, 0.5, 0.9], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])
        thresholds = [0.2, 0.4, 0.6, 0.8]
        cost_loss_ratios = [0.3, 0.7]

        actual = relative_economic_value(
            fcst, obs, cost_loss_ratios, threshold=thresholds, threshold_outputs=[0.4, 0.8]
        )
        expected = xr.Dataset(
            data_vars={
                "threshold_0_4": (["cost_loss_ratio"], np.array([1.0, 1.0])),
                "threshold_0_8": (["cost_loss_ratio"], np.array([-4 / 3, 0.5])),
            },
            coords={"cost_loss_ratio": np.array(cost_loss_ratios)},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_threshold_outputs_invalid(self):
        """Asking for thresholds that weren't supplied raises ValueError."""
        fcst = xr.DataArray(np.array([0.2, 0.8]), dims=["time"])
        obs = xr.DataArray(np.array([0, 1]), dims=["time"])
        threshold = [0.2, 0.5]
        threshold_outputs = [0.7]  # not in threshold

        with pytest.raises(ValueError, match="values in threshold_outputs must be in the supplied threshold parameter"):
            _validate_rev_inputs(
                fcst=fcst,
                obs=obs,
                cost_loss_ratios=0.5,
                threshold=threshold,
                threshold_dim="threshold",
                cost_loss_dim="cost_loss_ratio",
                weights=None,
                derived_metrics=None,
                threshold_outputs=threshold_outputs,
            )

    def test_threshold_outputs_without_threshold(self):
        """Asking for thresholds when threshold is None raises ValueError."""
        fcst = xr.DataArray(np.array([0, 1]), dims=["time"])
        obs = xr.DataArray(np.array([0, 1]), dims=["time"])

        with pytest.raises(ValueError, match="threshold_outputs can only be used when threshold parameter is provided"):
            _validate_rev_inputs(
                fcst=fcst,
                obs=obs,
                cost_loss_ratios=0.5,
                threshold=None,
                threshold_dim="threshold",
                cost_loss_dim="cost_loss_ratio",
                weights=None,
                derived_metrics=None,
                threshold_outputs=[0.5],
            )

    def test_probabilistic_maximum_output(self):
        """Test maximum value output"""
        fcst = xr.DataArray(
            [0.75] * 4 + [0.25] * 3 + [0.75] * 2 + [0.25] * 1, dims=["time"], coords={"time": np.arange(10)}
        )
        obs = xr.DataArray([1] * 4 + [0] * 3 + [0] * 2 + [1] * 1, dims=["time"], coords={"time": np.arange(10)})
        thresholds = np.arange(0.1, 1.0, 0.1)
        cost_loss_ratios = [0.2, 0.4, 0.6, 0.8]

        actual_full_result = relative_economic_value(fcst, obs, cost_loss_ratios, threshold=thresholds)

        actual_max_result = relative_economic_value(
            fcst, obs, cost_loss_ratios, threshold=thresholds, derived_metrics=["maximum"]
        )

        expected_full_result_values = np.array(
            [
                [0.0, 0.0, -0.5, -3.0],
                [0.0, 0.0, -0.5, -3.0],
                [-0.2, 0.3, 0.2, -0.8],
                [-0.2, 0.3, 0.2, -0.8],
                [-0.2, 0.3, 0.2, -0.8],
                [-0.2, 0.3, 0.2, -0.8],
                [-0.2, 0.3, 0.2, -0.8],
                [-3.0, -0.5, 0.0, 0.0],
                [-3.0, -0.5, 0.0, 0.0],
            ]
        )

        expected_full_result = xr.DataArray(
            expected_full_result_values,
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": np.arange(0.1, 1.0, 0.1), "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )

        xr.testing.assert_allclose(expected_full_result, actual_full_result)

        expected_max_result = xr.Dataset(
            data_vars={"maximum": (["cost_loss_ratio"], [0.0, 0.3, 0.2, 0.0])},
            coords={"cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )

        xr.testing.assert_allclose(expected_max_result, actual_max_result)

    def test_probabilistic_rational_user_output(self):
        """Test rational user output extraction (diagonal extraction)"""
        fcst = xr.DataArray(
            [0.75] * 4 + [0.25] * 3 + [0.75] * 2 + [0.25] * 1, dims=["time"], coords={"time": np.arange(10)}
        )
        obs = xr.DataArray([1] * 4 + [0] * 3 + [0] * 2 + [1] * 1, dims=["time"], coords={"time": np.arange(10)})
        thresholds = [0.2, 0.4, 0.6, 0.8]
        cost_loss_ratios = [0.2, 0.4, 0.6, 0.8]

        actual_full_result = relative_economic_value(fcst, obs, cost_loss_ratios, threshold=thresholds)

        actual_rational_result = relative_economic_value(
            fcst, obs, cost_loss_ratios, threshold=thresholds, derived_metrics=["rational_user"]
        )

        expected_full_result_values = np.array(
            [[0.0, 0.0, -0.5, -3.0], [-0.2, 0.3, 0.2, -0.8], [-0.2, 0.3, 0.2, -0.8], [-3.0, -0.5, 0.0, 0.0]]
        )

        expected_full_result = xr.DataArray(
            expected_full_result_values,
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.2, 0.4, 0.6, 0.8], "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )

        expected_rational_result = xr.Dataset(
            data_vars={"rational_user": (["cost_loss_ratio"], [0.0, 0.3, 0.2, 0.0])},
            coords={"threshold": (["cost_loss_ratio"], [0.2, 0.4, 0.6, 0.8]), "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )
        xr.testing.assert_allclose(expected_full_result, actual_full_result)
        xr.testing.assert_allclose(expected_rational_result, actual_rational_result)

    def test_rational_user_threshold_drop(self):
        """Test that 'rational_user' output drops threshold dimension."""
        fcst = xr.DataArray([0.1, 0.9, 0.5, 0.7], dims=["time"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        cost_loss_ratios = [0.1, 0.5]
        thresholds = [0.1, 0.5]  # must match cost_loss_ratios exactly

        result = relative_economic_value(
            fcst, obs, cost_loss_ratios, threshold=thresholds, derived_metrics=["rational_user"]
        )

        # The 'rational_user' DataArray should not contain the threshold_dim anymore
        assert "threshold" not in result["rational_user"].dims

    def test_rational_user_no_threshold_in_coords(self):
        """Test rational_user path where threshold coord somehow not present"""
        # Create a REV array with threshold as dimension but use isel instead of sel
        # to avoid coordinate preservation
        rev = xr.DataArray(
            np.array([[0.2, 0.5], [0.7, 0.9]]),
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.1, 0.2], "cost_loss_ratio": [0.1, 0.2]},
        )

        with mock.patch("xarray.concat") as mock_concat:
            # Make concat return a DataArray without threshold in coords
            mock_result = xr.DataArray([0.2, 0.9], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.1, 0.2]})
            mock_concat.return_value = mock_result

            result = _create_output_dataset(
                rev=rev,
                thresholds=[0.1, 0.2],
                cost_loss_ratios=[0.1, 0.2],
                derived_metrics=["rational_user"],
                threshold_outputs=None,
                threshold_dim="threshold",
                cost_loss_dim="cost_loss_ratio",
            )

            assert "rational_user" in result

    def test_custom_dimension_names(self):
        """Test that custom dimension names work correctly."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        # Test custom threshold_dim only
        result = relative_economic_value(
            fcst, obs, cost_loss_ratios=[0.3, 0.7], threshold=[0.5], threshold_dim="my_threshold"
        )

        assert "my_threshold" in result.dims
        assert "cost_loss_ratio" in result.dims
        assert result.dims == ("my_threshold", "cost_loss_ratio")

    def test_custom_cost_loss_dim(self):
        """Test that custom cost_loss_dim works correctly."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        # Test custom cost_loss_dim only
        result = relative_economic_value(fcst, obs, cost_loss_ratios=[0.3, 0.7], threshold=[0.5], cost_loss_dim="alpha")

        assert "threshold" in result.dims
        assert "alpha" in result.dims
        assert result.dims == ("threshold", "alpha")

    def test_both_custom_dimension_names(self):
        """Test that both custom dimension names work together."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        # Test both custom dims
        result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=[0.3, 0.7],
            threshold=[0.5],
            threshold_dim="decision_threshold",
            cost_loss_dim="alpha",
        )

        assert "decision_threshold" in result.dims
        assert "alpha" in result.dims
        assert result.dims == ("decision_threshold", "alpha")
        assert "threshold" not in result.dims
        assert "cost_loss_ratio" not in result.dims

    def test_custom_dims_with_derived_metrics(self):
        """Test that custom dimension names work with derived metrics."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        matching_values = [0.3, 0.5, 0.7]

        result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=matching_values,
            threshold=matching_values,
            threshold_dim="decision_threshold",
            cost_loss_dim="alpha",
            derived_metrics=["maximum", "rational_user"],
        )

        # Check maximum output
        assert "maximum" in result
        assert "alpha" in result["maximum"].dims
        assert "decision_threshold" not in result["maximum"].dims

        # Check rational_user output
        assert "rational_user" in result
        assert "alpha" in result["rational_user"].dims
        assert "decision_threshold" not in result["rational_user"].dims

    def test_custom_dims_with_threshold_outputs(self):
        """Test that custom dimension names work with threshold_outputs."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=[0.3, 0.7],
            threshold=[0.3, 0.5, 0.7],
            threshold_dim="decision_threshold",
            cost_loss_dim="alpha",
            threshold_outputs=[0.5],
        )

        # Check that threshold_outputs creates correctly named variables
        assert "threshold_0_5" in result
        assert "alpha" in result["threshold_0_5"].dims
        assert "decision_threshold" not in result["threshold_0_5"].dims


class TestWeights:
    """Tests for handling of time/spatial weights in REV calculations."""

    @pytest.mark.parametrize("weight_value", [0.5, 1, 2])
    def test_equal_weights_same_as_unweighted(self, make_contingency_data, weight_value):
        """When all weights are equal, weighted result should match unweighted."""
        fcst, obs = make_contingency_data(2, 2, 2, 2)

        weights = xr.DataArray([weight_value] * 8, dims=["time"])

        rev_weighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.2], weights=weights)
        rev_unweighted = relative_economic_value(fcst, obs, [0.2])

        xr.testing.assert_allclose(rev_weighted, rev_unweighted)

    def test_weights_nonuniform(self, make_contingency_data):
        """Test that non-uniform weights correctly adjust REV calculation."""
        # Samples: 2 of each item on the contingency table
        fcst, obs = make_contingency_data(2, 2, 2, 2)
        # Without weights: POD = 2/4 = 0.5, POFD = 2/4 = 0.5, obar = 4/8 = 0.5
        # REV at alpha=0.5: numerator = min(0.5,0.5) - 0.5*0.5*0.5 + 0.5*0.5*0.5 - 0.5 = 0
        #                   denominator = min(0.5,0.5) - 0.5*0.5 = 0.25
        #                   REV = 0/0.25 = 0
        unweighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5])
        assert unweighted.item() == 0.0

        # Now weight the hits heavily (w=2) and false alarms lightly (w=0.5)
        weights = xr.DataArray([2, 2, 1, 1, 0.5, 0.5, 1, 1], dims=["time"])

        # Weighted counts:
        #   hits: 2+2 = 4,  misses: 1+1 = 2  -> events_total = 6
        #   FA: 0.5+0.5 = 1, CN: 1+1 = 2     -> non_events_total = 3
        # Weighted POD = 4/6 = 0.667, POFD = 1/3 = 0.333
        # Weighted obar = 6/9 = 0.667
        #
        # REV at alpha=0.5:
        #   num = min(0.5, 0.667) - 0.333*0.5*0.333 + 0.667*0.667*0.5 - 0.667
        #       = 0.5 - 0.0555 + 0.222 - 0.667 = 0.0
        #   den = min(0.5, 0.667) - 0.667*0.5 = 0.5 - 0.333 = 0.167
        #   REV = 0.0 / 0.167 = 0.0
        #
        # At alpha=0.3 (where weighting effect is more visible):
        #   num = min(0.3, 0.667) - 0.333*0.3*0.333 + 0.667*0.667*0.7 - 0.667
        #       = 0.3 - 0.0333 + 0.311 - 0.667 = -0.089
        #   den = min(0.3, 0.667) - 0.667*0.3 = 0.3 - 0.2 = 0.1
        #   REV = -0.089 / 0.1 = -0.89

        weighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.3], weights=weights)
        expected = xr.DataArray([-0.89], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.3]})

        xr.testing.assert_allclose(weighted, expected, atol=0.01)

    def test_spatial_weights_broadcast(self):
        """Test that latitude weights broadcast correctly over time dimension."""
        # Two latitudes, 4 timesteps each - small enough to verify by hand
        #
        # Lat 60°: Perfect forecast (2 hits, 2 correct negatives)
        # Lat 30°: No skill (1 hit, 1 miss, 1 FA, 1 CN)

        fcst = xr.DataArray(
            [[1, 0, 1, 0], [1, 0, 1, 0]],  # lat 60: fcst matches obs perfectly  # lat 30: fcst uncorrelated with obs
            dims=["lat", "time"],
            coords={"lat": [60, 30], "time": range(4)},
        )
        obs = xr.DataArray(
            [[1, 0, 1, 0], [1, 1, 0, 0]],  # lat 60  # lat 30
            dims=["lat", "time"],
            coords={"lat": [60, 30], "time": range(4)},
        )

        #                   Lat 60          Lat 30
        # Contingency:      H=2, CN=2       H=1, M=1, FA=1, CN=1
        # POD:              2/2 = 1.0       1/2 = 0.5
        # POFD:             0/2 = 0.0       1/2 = 0.5
        # obar:             2/4 = 0.5       2/4 = 0.5
        # REV (alpha=0.5):  1.0             0.0

        # Unweighted: simple average of REV values
        # REV = (1.0 + 0.0) / 2 = 0.5
        unweighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5])
        xr.testing.assert_allclose(
            unweighted, xr.DataArray([0.5], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})
        )

        # Cosine weights: lat 60° -> cos(60°) = 0.5, lat 30° -> cos(30°) = 0.866
        # This weights the LOW-skill latitude MORE heavily
        weights = xr.DataArray([0.5, 0.866], dims=["lat"], coords={"lat": [60, 30]})  # cos(60°), cos(30°)

        # Weighted calculation combines contingency tables:
        #   weighted_hits = 2*0.5 + 1*0.866 = 1.866
        #   weighted_misses = 0*0.5 + 1*0.866 = 0.866
        #   weighted_FA = 0*0.5 + 1*0.866 = 0.866
        #   weighted_CN = 2*0.5 + 1*0.866 = 1.866
        #
        #   weighted_POD = 1.866 / (1.866 + 0.866) = 0.683
        #   weighted_POFD = 0.866 / (0.866 + 1.866) = 0.317
        #   weighted_obar = (1.866 + 0.866) / 5.464 = 0.5
        #
        #   REV at alpha=0.5:
        #     num = 0.5 - 0.317*0.5*0.5 + 0.683*0.5*0.5 - 0.5 = 0.0915
        #     den = 0.5 - 0.5*0.5 = 0.25
        #     REV = 0.0915 / 0.25 = 0.366

        weighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights)
        expected = xr.DataArray([0.366], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})
        xr.testing.assert_allclose(weighted, expected, atol=0.001)

    def test_preserve_dims_with_weights(self):
        """Test that weights apply correctly when preserving a dimension (lon)."""
        # 2 latitudes x 2 longitudes x 4 timesteps
        # Weights vary by latitude only; we reduce over time and lat, preserve lon
        #
        # Lon 0: Good forecast at both latitudes
        # Lon 180: Inverted forecast (terrible skill)

        fcst = xr.DataArray(
            [
                [[1, 1], [0, 0], [1, 1], [0, 0]],  # lat 60: [lon0, lon180] per timestep
                [[1, 1], [0, 0], [0, 0], [1, 1]],
            ],  # lat 30
            dims=["lat", "time", "lon"],
            coords={"lat": [60, 30], "time": range(4), "lon": [0, 180]},
        )
        obs = xr.DataArray(
            [[[1, 0], [0, 1], [1, 0], [0, 1]], [[1, 0], [0, 1], [1, 0], [0, 1]]],  # lat 60  # lat 30
            dims=["lat", "time", "lon"],
            coords={"lat": [60, 30], "time": range(4), "lon": [0, 180]},
        )

        # At lon=0: fcst and obs align well
        #   Lat 60: H=2, CN=2 (perfect)
        #   Lat 30: H=1, M=1, FA=1, CN=1 (no skill)
        #
        # At lon=180: obs is inverted, so forecasts are anti-correlated
        #   Lat 60: M=2, FA=2 (anti-perfect)
        #   Lat 30: H=1, M=1, FA=1, CN=1 (no skill - same as lon=0)

        # Cosine weights: lat 60° -> 0.5, lat 30° -> 0.866
        weights = xr.DataArray([0.5, 0.866], dims=["lat"], coords={"lat": [60, 30]})

        result = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights, preserve_dims=["lon"])

        # Lon=0 weighted calculation (same as previous test):
        #   weighted_H = 2*0.5 + 1*0.866 = 1.866
        #   weighted_M = 0*0.5 + 1*0.866 = 0.866
        #   weighted_FA = 0*0.5 + 1*0.866 = 0.866
        #   weighted_CN = 2*0.5 + 1*0.866 = 1.866
        #   POD = 1.866/2.732 = 0.683, POFD = 0.866/2.732 = 0.317, obar = 0.5
        #   REV = 0.366
        #
        # Lon=180 weighted calculation:
        #   weighted_H = 0*0.5 + 1*0.866 = 0.866
        #   weighted_M = 2*0.5 + 1*0.866 = 1.866
        #   weighted_FA = 2*0.5 + 1*0.866 = 1.866
        #   weighted_CN = 0*0.5 + 1*0.866 = 0.866
        #   POD = 0.866/2.732 = 0.317, POFD = 1.866/2.732 = 0.683, obar = 0.5
        #   num = 0.5 - 0.683*0.5*0.5 + 0.317*0.5*0.5 - 0.5 = -0.0915
        #   den = 0.5 - 0.25 = 0.25
        #   REV = -0.0915 / 0.25 = -0.366

        expected = xr.DataArray(
            [[0.366], [-0.366]], dims=["lon", "cost_loss_ratio"], coords={"lon": [0, 180], "cost_loss_ratio": [0.5]}
        )
        xr.testing.assert_allclose(result, expected, atol=0.001)

    def test_weights_negative(self, make_contingency_data):
        """Test that negative weights raise a ValueError during calculation."""
        fcst, obs = make_contingency_data(1, 0, 0, 1)  # actual data doesn't matter

        # Negative weights should raise during calculation, not validation
        with pytest.raises(ValueError, match=re.escape(ERROR_INVALID_WEIGHTS.strip())):
            relative_economic_value(
                fcst,
                obs,
                cost_loss_ratios=[0.2, 0.5],
                weights=xr.DataArray([1, -1], dims=["time"], coords={"time": [1, 2]}),
            )


class TestDatasetInputs:
    """Test that REV works with xr.Dataset inputs."""

    def test_forecast_as_dataset(self):
        """Test with forecast as Dataset."""
        fcst_ds = xr.Dataset(
            {"ecmwf": xr.DataArray([0, 1, 1, 0], dims=["time"]), "access": xr.DataArray([1, 0, 0, 1], dims=["time"])}
        )
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        actual = relative_economic_value(fcst_ds, obs, [0.5])

        expected = xr.Dataset(
            data_vars={"ecmwf": (["cost_loss_ratio"], [1.0]), "access": (["cost_loss_ratio"], [-1.0])},
            coords={"cost_loss_ratio": [0.5]},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_obs_as_dataset(self):
        """Test with observations as Dataset."""
        fcst = xr.DataArray([0, 1, 1, 0], dims=["time"])
        obs_ds = xr.Dataset(
            {
                "station_data": xr.DataArray([0, 1, 1, 0], dims=["time"]),
                "radar_data": xr.DataArray([1, 0, 0, 1], dims=["time"]),
            }
        )

        actual = relative_economic_value(fcst, obs_ds, [0.3, 0.7], threshold=[0.5])
        expected = xr.Dataset(
            data_vars={
                "station_data": (["threshold", "cost_loss_ratio"], [[1.0, 1.0]]),
                "radar_data": (["threshold", "cost_loss_ratio"], [[-7 / 3, -7 / 3]]),
            },
            coords={"threshold": [0.5], "cost_loss_ratio": [0.3, 0.7]},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_both_as_dataset(self):
        """Test with both as Dataset."""
        fcst_ds = xr.Dataset(
            {"ecmwf": xr.DataArray([1, 1, 1, 0], dims=["time"]), "access": xr.DataArray([0, 0, 0, 1], dims=["time"])}
        )
        obs_ds = xr.Dataset(
            {
                "station_data": xr.DataArray([0, 0, 1, 1], dims=["time"]),
                "radar_data": xr.DataArray([1, 1, 0, 0], dims=["time"]),
            }
        )

        actual = relative_economic_value(fcst_ds, obs_ds, [0.3, 0.7], threshold=[0.5])

        expected = xr.Dataset(
            data_vars={
                "access__vs__radar_data": (["threshold", "cost_loss_ratio"], np.array([[-11 / 6, -7 / 6]])),
                "access__vs__station_data": (["threshold", "cost_loss_ratio"], np.array([[-1 / 6, 0.5]])),
                "ecmwf__vs__radar_data": (["threshold", "cost_loss_ratio"], np.array([[0.5, -1 / 6]])),
                "ecmwf__vs__station_data": (["threshold", "cost_loss_ratio"], np.array([[-7 / 6, -11 / 6]])),
            },
            coords={
                "threshold": [0.5],
                "cost_loss_ratio": [0.3, 0.7],
            },
        )

        xr.testing.assert_allclose(actual, expected)

    def test_weights_as_dataset_raises_error(self):
        """Test that weights as Dataset raises an error."""
        fcst = xr.DataArray([0, 1, 1, 0], dims=["time"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights_ds = xr.Dataset(
            {
                "pizza": xr.DataArray([1.0, 2.0, 1.5, 1.0], dims=["time"]),
                "burrito": xr.DataArray([0.5, 1.0, 1.0, 0.5], dims=["time"]),
            }
        )

        with pytest.raises(ValueError, match="Weights cannot be Datasets."):
            relative_economic_value(fcst, obs, [0.5], weights=weights_ds)

    def test_pod_as_dataset(self):
        """Test with POD as Dataset, others as DataArrays."""
        pod_ds = xr.Dataset(
            {
                "model_a": xr.DataArray([0.8, 0.6], dims=["threshold"]),
                "model_b": xr.DataArray([0.9, 0.5], dims=["threshold"]),
            }
        )
        pofd = xr.DataArray([0.2, 0.1], dims=["threshold"])
        climatology = xr.DataArray(0.3)

        actual = relative_economic_value_from_rates(pod_ds, pofd, climatology, [0.3, 0.7])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"model_a", "model_b"}
        assert "cost_loss_ratio" in actual["model_a"].dims

    def test_pod_and_pofd_as_dataset(self):
        """Test with POD and POFD as Datasets."""
        pod_ds = xr.Dataset(
            {
                "model_a": xr.DataArray([0.8, 0.6], dims=["threshold"]),
                "model_b": xr.DataArray([0.7, 0.5], dims=["threshold"]),
            }
        )
        pofd_ds = xr.Dataset(
            {
                "model_a": xr.DataArray([0.2, 0.1], dims=["threshold"]),
                "model_b": xr.DataArray([0.15, 0.08], dims=["threshold"]),
            }
        )
        climatology = xr.DataArray(0.4)

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, climatology, [0.5])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"model_a", "model_b"}

    def test_pod_and_climatology_as_dataset(self):
        """Test with POD and climatology as Datasets."""
        pod_ds = xr.Dataset(
            {"region_1": xr.DataArray([0.8], dims=["threshold"]), "region_2": xr.DataArray([0.6], dims=["threshold"])}
        )
        pofd = xr.DataArray([0.15], dims=["threshold"])
        climatology_ds = xr.Dataset({"region_1": xr.DataArray(0.3), "region_2": xr.DataArray(0.45)})

        actual = relative_economic_value_from_rates(pod_ds, pofd, climatology_ds, [0.3, 0.7])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"region_1", "region_2"}

    def test_all_as_dataset(self):
        """Test with POD, POFD, and climatology all as Datasets."""
        pod_ds = xr.Dataset(
            {
                "var_1": xr.DataArray([0.7, 0.9], dims=["threshold"]),
                "var_2": xr.DataArray([0.6, 0.8], dims=["threshold"]),
            }
        )
        pofd_ds = xr.Dataset(
            {
                "var_1": xr.DataArray([0.1, 0.05], dims=["threshold"]),
                "var_2": xr.DataArray([0.2, 0.1], dims=["threshold"]),
            }
        )
        climatology_ds = xr.Dataset({"var_1": xr.DataArray(0.25), "var_2": xr.DataArray(0.35)})

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, climatology_ds, [0.2, 0.5, 0.8])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"var_1", "var_2"}
        assert actual["var_1"].dims == ("cost_loss_ratio", "threshold")
        assert len(actual.cost_loss_ratio) == 3

    def test_dataset_preserves_numeric_results(self):
        """Test that Dataset processing produces correct numeric values."""
        # Use simple values where we can verify the math
        pod_scalar = xr.DataArray(1.0)  # Perfect detection
        pofd_scalar = xr.DataArray(0.0)  # No false alarms
        climatology_scalar = xr.DataArray(0.5)

        # Calculate with scalars
        expected_scalar = relative_economic_value_from_rates(pod_scalar, pofd_scalar, climatology_scalar, [0.5])

        # Calculate with Dataset
        pod_ds = xr.Dataset({"test": pod_scalar})
        actual = relative_economic_value_from_rates(pod_ds, pofd_scalar, climatology_scalar, [0.5])

        xr.testing.assert_allclose(actual["test"], expected_scalar)


class TestErrorHandling:
    """Tests that check that error handling is done correctly"""

    @pytest.mark.parametrize(
        "fcst_data,obs_data,cost_loss_ratios,threshold,expected_error",
        [
            # Probabilistic forecasts without threshold
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], None, "0, 1, or NaN"),
            # Cost-loss ratios out of range
            ([0, 1, 1], [0, 1, 0], [-0.1, 0.5, 1.2], None, "between 0 and 1"),
            # Cost-loss ratios not monotonic
            ([0, 1, 1], [0, 1, 0], [0.5, 0.3, 0.7], None, "monotonically increasing"),
            # Threshold values out of range
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], [-0.1, 0.5], "between 0 and 1"),
            # Threshold values not monotonic
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], [0.7, 0.3], "monotonically increasing"),
            # Invalid observation values
            ([0, 1, 1], [0, 1, 2], [0.5], None, "0, 1, or NaN"),
            # Invalid forecast values
            ([0, 1, 2], [0, 1, 1], [0.5], None, "0, 1, or NaN"),
        ],
        ids=[
            "probabilistic_without_threshold",
            "cost_loss_out_of_range",
            "cost_loss_not_monotonic",
            "threshold_out_of_range",
            "threshold_not_monotonic",
            "invalid_obs_values",
            "invalid_fcst_values",
        ],
    )
    def test_input_validation(self, fcst_data, obs_data, cost_loss_ratios, threshold, expected_error):
        """
        Test that relative_economic_value validates inputs correctly.

        Validates that the function raises ValueError for:
        - Probabilistic forecasts (values between 0 and 1) without threshold parameter
        - Cost-loss ratios outside [0, 1] range or not strictly monotonically increasing
        - Threshold values outside [0, 1] range or not strictly monotonically increasing
        - Forecast or observation values outside {0, 1, NaN}
        """
        fcst = xr.DataArray(fcst_data, dims=["time"])
        obs = xr.DataArray(obs_data, dims=["time"])

        with pytest.raises(ValueError, match=expected_error):
            relative_economic_value(fcst, obs, cost_loss_ratios, threshold=threshold)

    @pytest.mark.parametrize(
        "array_name,forbidden_dim",
        [
            ("fcst", "threshold"),
            ("fcst", "cost_loss_ratio"),
            ("obs", "threshold"),
            ("obs", "cost_loss_ratio"),
            ("weights", "threshold"),
            ("weights", "cost_loss_ratio"),
        ],
    )
    def test_forbidden_dimensions(self, array_name, forbidden_dim):
        """Test that 'threshold' and 'cost_loss_ratio' cannot be dimensions in fcst, obs, or weights."""
        # Default valid arrays
        fcst = xr.DataArray([0, 1], dims=["time"])
        obs = xr.DataArray([0, 1], dims=["time"])
        weights = None

        # Create the problematic array
        if array_name == "fcst":
            fcst = xr.DataArray([0, 1], dims=[forbidden_dim])
        elif array_name == "obs":
            obs = xr.DataArray([0, 1], dims=[forbidden_dim])
        elif array_name == "weights":
            weights = xr.DataArray([1.0, 2.0], dims=[forbidden_dim])

        with pytest.raises(ValueError, match=f"'{forbidden_dim}' cannot be a dimension in {array_name}"):
            relative_economic_value(fcst, obs, [0.5], weights=weights)

    def test_value_without_matching_thresholds(self):
        """Test that 'rational_user' output requires matching thresholds"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.3, 0.5], threshold=[0.2, 0.5], derived_metrics=["rational_user"])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "identical" in str(e)

    def test_invalid_derived_metrics(self):
        """Test validation of derived_metrics values"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])
        cost_loss_ratios = [0.5]
        derived_metrics = ["pizza_oven"]

        try:
            relative_economic_value(fcst, obs, cost_loss_ratios, threshold=[0.5], derived_metrics=derived_metrics)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Invalid derived_metrics" in str(e)

    def test_create_output_dataset_invalid_derived_metrics(self):
        """Test that _create_output_dataset raises ValueError for invalid derived_metrics."""

        rev = xr.DataArray(
            [[0.5, 0.6], [0.7, 0.8]],
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.3, 0.7], "cost_loss_ratio": [0.2, 0.5]},
        )

        with pytest.raises(ValueError, match="Invalid derived_metrics value: 'invalid'"):
            _create_output_dataset(
                rev=rev,
                thresholds=[0.3, 0.7],
                cost_loss_ratios=[0.2, 0.5],
                derived_metrics=["invalid"],
                threshold_outputs=None,
                threshold_dim="threshold",
                cost_loss_dim="cost_loss_ratio",
            )

    def test_non_monotonic_cost_loss_ratios_raises(self):
        """Test that REV catches non-monotonic cost loss ratios"""
        pod = SCALAR_DA
        pofd = pod.copy()
        climatology = pod.copy()

        # non-monotonic sequence (should trigger check_monotonic_array)
        bad_cost_loss = np.array([0.0, 0.5, 0.3])

        with pytest.raises(ValueError) as excinfo:
            relative_economic_value_from_rates(pod, pofd, climatology, bad_cost_loss)

        # ensure the error refers to cost_loss_ratios (the wrapper prepends that text)
        assert "cost_loss_ratios" in str(excinfo.value)

    @pytest.mark.parametrize("arr", [[-0.1, 0.5], [0.0, 1.5]])
    def test_out_of_range_arrays(self, arr):
        """Test the monotonic array checker rejects out-of-range values"""
        with pytest.raises(ValueError, match="array values should be between 0 and 1."):
            check_monotonic_array(arr)

    def test_binary_forecast_strict_validation(self, make_contingency_data):
        """Test strict binary validation for binary forecasts."""
        # Valid binary forecast (only 0 and 1)
        fcst, obs = make_contingency_data(1, 1, 1, 1)

        result = relative_economic_value(fcst, obs, [0.5], check_args=True)
        assert result is not None

        # Invalid binary forecast (contains 0.5)
        fcst_bad = xr.DataArray([0, 0.5, 1, 0], dims=["time"])

        with pytest.raises(ValueError, match="fcst must contain only 0, 1, or NaN values"):
            relative_economic_value(fcst_bad, obs, [0.5], check_args=True)

    @pytest.mark.parametrize("which_input", ["pod", "pofd", "climatology"])
    def test_cost_loss_ratio_dim_in_inputs_raises(self, which_input):
        """create a DataArray that contains the forbidden dimension name"""
        da_with_forbidden_dim = xr.DataArray(
            np.array([0.5, 0.5]),
            dims=("cost_loss_ratio",),
            coords={"cost_loss_ratio": [0.1, 0.9]},
        )

        # other inputs are normal scalar-like arrays
        pod = SCALAR_DA
        pofd = pod.copy()
        climatology = pod.copy()

        # replace the selected input with the bad one
        if which_input == "pod":
            pod = da_with_forbidden_dim
        elif which_input == "pofd":
            pofd = da_with_forbidden_dim
        else:
            climatology = da_with_forbidden_dim

        good_cost_loss = np.array([0.1, 0.5, 0.9])  # valid monotonic ratios

        with pytest.raises(ValueError) as excinfo:
            relative_economic_value_from_rates(pod, pofd, climatology, good_cost_loss)

        assert "dimension 'cost_loss_ratio' must not be in input data" in str(excinfo.value)

    def test_single_float_cost_loss_is_converted_to_length_one_coordinate(self):
        """
        Passing a single float for cost_loss_ratios should produce an output whose
        cost_loss_ratio coordinate has length 1 and contains that value.
        """
        fcst = PROB_FCST_DA
        obs = BINARY_DA
        single_alpha = 0.3

        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=single_alpha,
            threshold=[0.5],  # to trigger probabilistic branch
            check_args=True,
        )

        expected = xr.DataArray(
            [[1.0]], dims=["threshold", "cost_loss_ratio"], coords={"threshold": [0.5], "cost_loss_ratio": [0.3]}
        )

        xr.testing.assert_allclose(actual, expected)

    def test_check_monotonic_array_non_convertible_raises_typeerror(self):
        """
        If the input cannot be converted to a numpy array of floats, a TypeError is raised.
        """
        with pytest.raises(TypeError, match="could not convert array into a numpy ndarray of floats"):
            check_monotonic_array(["not", "numbers"])

    def test_check_monotonic_array_rejects_multidimensional_arrays(self):
        """
        Passing a 2-D (or higher) array should raise the one-dimensional check.
        """
        with pytest.raises(ValueError, match="array must be one-dimensional"):
            check_monotonic_array([[0.1, 0.2], [0.3, 0.4]])

    def test_check_args_true_validates_and_raises_for_probabilistic_without_threshold(self):
        """
        When check_args=True, probabilistic forecasts require a threshold.
        _validate_rev_inputs should raise a ValueError in that case.
        """
        fcst = PROB_FCST_DA
        obs = BINARY_DA

        with pytest.raises(ValueError, match="When threshold is None, fcst must contain only 0, 1, or NaN values"):
            relative_economic_value(
                fcst,
                obs,
                cost_loss_ratios=[0.1, 0.5],
                threshold=None,
                check_args=True,
            )

    def test_fcst_outside_01_with_threshold(self):
        """Check when threshold provided, you need probability forecasts between 0 and 1."""
        fcst = xr.DataArray(np.array([-0.1, 1.2]), dims=["time"])
        obs = xr.DataArray(np.array([0, 1]), dims=["time"])

        with pytest.raises(ValueError, match="When threshold is provided, fcst must contain values between 0 and 1"):
            relative_economic_value(
                fcst,
                obs,
                cost_loss_ratios=[0.1, 0.5],
                threshold=[0.5],
                check_args=True,
            )

    def test_check_args_false_skips_validation_allows_probabilistic_without_threshold(self):
        """
        When check_args=False, the function should skip input validation and proceed.
        This test asserts no ValueError is raised for a probabilistic fcst without threshold.
        """
        fcst = PROB_FCST_DA
        obs = BINARY_DA

        # Nothing should raise; we expect an xr.DataArray back (or Dataset depending on other args)
        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=[0.1, 0.5],
            threshold=None,
            check_args=False,  # skip validation
        )

        expected = xr.DataArray([np.nan, np.nan], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.1, 0.5]})
        xr.testing.assert_allclose(actual, expected)

    def test_rational_user_without_threshold_raises(self, make_contingency_data):
        """Test derived metrics 'rational_user' without threshold raises ValueError"""
        fcst, obs = make_contingency_data(1, 1, 1, 1)

        with pytest.raises(
            ValueError, match="derived_metrics 'rational_user' can only be used when threshold parameter is provided"
        ):
            relative_economic_value(
                fcst=fcst,
                obs=obs,
                cost_loss_ratios=[0.2, 0.5],
                derived_metrics=["rational_user"],
            )


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
class TestDaskCompatibility:
    """Test that REV works correctly with Dask arrays."""

    def test_binary_forecast_with_dask(self, make_contingency_data):
        """Test basic REV calculation with Dask arrays."""

        fcst, obs = make_contingency_data(2, 0, 0, 3)

        # Convert to Dask
        fcst = fcst.chunk({"time": 2})
        obs = obs.chunk({"time": 2})

        result = relative_economic_value(fcst, obs, [0.5], check_args=False)

        # Verify result is still lazy
        assert isinstance(result.data, da.Array)

        # Verify computation produces correct result
        computed = result.compute()
        expected = xr.DataArray([1.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})
        xr.testing.assert_allclose(computed, expected)

    def test_probabilistic_forecast_with_dask(self):
        """Test probabilistic REV with Dask arrays."""
        # fcst = [0.8, 0.5, 0.1, 0.9, 0.4, 0.2, 0.75, 0.6, 0.15, 0.35]
        # obs  = [1,   0,   0,   1,   1,   0,   0,    1,   0,    1   ]
        #
        # =========================================================================
        # Threshold 0.3: fcst >= 0.3 gives binary forecasts
        # =========================================================================
        # fcst:  1    1    0    1    1    0    1    1    0    1
        # obs:   1    0    0    1    1    0    0    1    0    1
        #
        # Contingency table:
        #   hits (fcst=1, obs=1): 5
        #   false_alarms (fcst=1, obs=0): 2
        #   misses (fcst=0, obs=1): 0
        #   correct_neg (fcst=0, obs=0): 3
        #
        # POD = hits / (hits + misses) = 5 / (5 + 0) = 1.0
        # POFD = false_alarms / (false_alarms + correct_neg) = 2 / (2 + 3) = 0.4
        # obar = (hits + misses) / n = 5 / 10 = 0.5
        #
        # =========================================================================
        # Threshold 0.7: fcst >= 0.7 gives binary forecasts
        # =========================================================================
        # fcst:   1    0    0    1    0    0    1    0    0    0
        # obs:    1    0    0    1    1    0    0    1    0    1
        #
        # Contingency table:
        #   hits (fcst=1, obs=1): 2
        #   false_alarms (fcst=1, obs=0): 1
        #   misses (fcst=0, obs=1): 3
        #   correct_neg (fcst=0, obs=0): 4
        #
        # POD = hits / (hits + misses) = 2 / (2 + 3) = 0.4
        # POFD = false_alarms / (false_alarms + correct_neg) = 1 / (1 + 4) = 0.2
        # obar = (hits + misses) / n = 5 / 10 = 0.5
        #
        # =========================================================================
        # REV calculations
        # =========================================================================
        # REV = (min(alpha, obar) - F*alpha*(1-obar) + H*obar*(1-alpha) - obar) / (min(alpha, obar) - obar*alpha)
        #
        # --- Threshold 0.3 (H=1.0, F=0.4, obar=0.5) ---
        #
        # alpha=0.2:
        #   num = min(0.2, 0.5) - 0.4*0.2*0.5 + 1.0*0.5*0.8 - 0.5
        #       = 0.2 - 0.04 + 0.4 - 0.5 = 0.06
        #   den = min(0.2, 0.5) - 0.5*0.2 = 0.2 - 0.1 = 0.1
        #   REV = 0.06 / 0.1 = 0.6
        #
        # alpha=0.5:
        #   num = min(0.5, 0.5) - 0.4*0.5*0.5 + 1.0*0.5*0.5 - 0.5
        #       = 0.5 - 0.1 + 0.25 - 0.5 = 0.15
        #   den = min(0.5, 0.5) - 0.5*0.5 = 0.5 - 0.25 = 0.25
        #   REV = 0.15 / 0.25 = 0.6
        #
        # --- Threshold 0.7 (H=0.4, F=0.2, obar=0.5) ---
        #
        # alpha=0.2:
        #   num = min(0.2, 0.5) - 0.2*0.2*0.5 + 0.4*0.5*0.8 - 0.5
        #       = 0.2 - 0.02 + 0.16 - 0.5 = -0.16
        #   den = min(0.2, 0.5) - 0.5*0.2 = 0.2 - 0.1 = 0.1
        #   REV = -0.16 / 0.1 = -1.6
        #
        # alpha=0.5:
        #   num = min(0.5, 0.5) - 0.2*0.5*0.5 + 0.4*0.5*0.5 - 0.5
        #       = 0.5 - 0.05 + 0.1 - 0.5 = 0.05
        #   den = min(0.5, 0.5) - 0.5*0.5 = 0.5 - 0.25 = 0.25
        #   REV = 0.05 / 0.25 = 0.2
        #
        # Expected result: [[0.6, 0.6], [-1.6, 0.2]]

        fcst_np = np.array([0.8, 0.5, 0.1, 0.9, 0.4, 0.2, 0.75, 0.6, 0.15, 0.35])
        obs_np = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])

        fcst = xr.DataArray(da.from_array(fcst_np, chunks=5), dims=["time"])
        obs = xr.DataArray(da.from_array(obs_np, chunks=5), dims=["time"])

        result = relative_economic_value(fcst, obs, [0.2, 0.5], threshold=[0.3, 0.7], check_args=False)

        assert isinstance(result.data, da.Array)
        computed = result.compute()

        expected = xr.DataArray(
            [[0.6, 0.6], [-1.6, 0.2]],
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.3, 0.7], "cost_loss_ratio": [0.2, 0.5]},
        )
        xr.testing.assert_allclose(computed, expected)

    @pytest.mark.parametrize("check_args", [True, False])
    def test_dask_array_check_args_behavior(self, check_args):
        """Test that check_args parameter controls validation with Dask arrays."""
        pytest.importorskip("dask")

        fcst_dask = xr.DataArray(da.from_array([0, 1, 1, 0], chunks=2), dims=["time"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])

        # Both should work without raising
        result = relative_economic_value(fcst_dask, obs, [0.5], check_args=check_args)

        # Result should still be lazy (validation only computes min/max, not full result)
        assert isinstance(result.data, da.Array)

        # Verify correct result after computation
        computed = result.compute()
        expected = xr.DataArray([0.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})
        xr.testing.assert_allclose(computed, expected)

    def test_no_dask_warning_when_dask_unavailable(self, monkeypatch, make_contingency_data):
        """Test that no warning is issued when dask is not available."""
        # Temporarily hide dask module

        monkeypatch.setitem(sys.modules, "dask", None)
        monkeypatch.setitem(sys.modules, "dask.array", None)

        fcst, obs = make_contingency_data(1, 1, 1, 1)

        # Should not raise any warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = relative_economic_value(fcst, obs, [0.5], check_args=True)

        assert result is not None

    def test_dask_dataset_validation(self):
        """Test validation with Dask-backed Datasets."""
        # Valid Dask Dataset for obs
        obs_ds = xr.Dataset(
            {
                "var1": xr.DataArray(da.from_array([0, 1, 0, 1], chunks=2), dims=["time"]),
                "var2": xr.DataArray(da.from_array([1, 0, 1, 0], chunks=2), dims=["time"]),
            }
        )
        fcst = xr.DataArray([0, 1, 1, 0], dims=["time"])

        # Should work with check_args=True
        result = relative_economic_value(fcst, obs_ds, [0.5], check_args=True)
        assert isinstance(result, xr.Dataset)

        # Invalid Dask Dataset for obs (contains 2)
        obs_bad_ds = xr.Dataset({"var1": xr.DataArray(da.from_array([0, 2, 0, 1], chunks=2), dims=["time"])})

        with pytest.raises(ValueError, match="obs must contain only 0, 1, or NaN values"):
            relative_economic_value(fcst, obs_bad_ds, [0.5], check_args=True)

        # Valid Dask Dataset for fcst with threshold
        fcst_ds = xr.Dataset(
            {
                "model1": xr.DataArray(da.from_array([0.2, 0.8, 0.6, 0.4], chunks=2), dims=["time"]),
                "model2": xr.DataArray(da.from_array([0.1, 0.9, 0.5, 0.3], chunks=2), dims=["time"]),
            }
        )
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        result = relative_economic_value(fcst_ds, obs, [0.5], threshold=[0.5], check_args=True)
        assert isinstance(result, xr.Dataset)

        # Invalid Dask Dataset for fcst (contains 1.5)
        fcst_bad_ds = xr.Dataset({"model1": xr.DataArray(da.from_array([0.2, 1.5, 0.6, 0.4], chunks=2), dims=["time"])})

        with pytest.raises(ValueError, match="fcst must contain values between 0 and 1"):
            relative_economic_value(fcst_bad_ds, obs, [0.5], threshold=[0.5], check_args=True)

    def test_rev_with_dask_inputs(self):
        """Test that REV works correctly when inputs are Dask arrays."""

        # Create Dask Arrays from the numpy data
        fcst_data = da.from_array(np.array([0.2, 0.8, 0.1]), chunks=1)
        obs_data = da.from_array(np.array([0, 1, 0]), chunks=1)

        fcst = xr.DataArray(fcst_data, dims=["time"])
        obs = xr.DataArray(obs_data, dims=["time"])

        cost_loss_ratios = [0.5]
        threshold = [0.5]  # Ensure a threshold is used for the probabilistic path

        result = relative_economic_value(fcst, obs, cost_loss_ratios, threshold=threshold)

        # Test that it's still a dask object
        assert is_dask_collection(result.data)

        expected = xr.DataArray(
            [[1.0]], dims=["threshold", "cost_loss_ratio"], coords={"threshold": [0.5], "cost_loss_ratio": [0.5]}
        )

        xr.testing.assert_allclose(result, expected)

        result_computed = result.compute()
        assert not is_dask_collection(result_computed.data)

    def test_rev_with_dask_multidimensional(self):
        """Test REV with multi-dimensional Dask arrays."""
        fcst = xr.DataArray(da.from_array(np.random.rand(10, 5), chunks=(5, 5)), dims=["time", "location"])
        obs = xr.DataArray(da.from_array(np.random.randint(0, 2, (10, 5)), chunks=(5, 5)), dims=["time", "location"])

        # Reduce over time, preserve location
        result = relative_economic_value(fcst, obs, [0.5], threshold=[0.5], reduce_dims=["time"], check_args=False)

        assert isinstance(result.data, da.Array)
        assert "location" in result.dims
        computed = result.compute()
        assert computed.shape == (5, 1, 1)

    def test_dask_performance_no_premature_compute(self):
        """Ensure Dask arrays aren't prematurely computed during setup."""
        # Large array that would be expensive to compute
        large_fcst = xr.DataArray(da.random.random((1000, 1000), chunks=(100, 100)), dims=["time", "location"])
        large_obs = xr.DataArray(da.random.randint(0, 2, (1000, 1000), chunks=(100, 100)), dims=["time", "location"])

        # This should be fast (no computation)
        import time

        start = time.time()
        result = relative_economic_value(large_fcst, large_obs, [0.5], threshold=[0.5], check_args=False)
        elapsed = time.time() - start

        # Should take < 1 second (just graph construction)
        assert elapsed < 1.0, "Graph construction took too long - possible premature compute"
        assert isinstance(result.data, da.Array)

    def test_weighted_rev_with_dask_deferred_validation(self):
        """Critical test: weighted REV with invalid weights should defer error."""
        fcst = xr.DataArray(da.from_array([0, 1, 1, 0], chunks=2), dims=["time"])
        obs = xr.DataArray(da.from_array([0, 1, 0, 1], chunks=2), dims=["time"])
        bad_weights = xr.DataArray(da.from_array([-1.0, 1.0, 1.0, 1.0], chunks=2), dims=["time"])

        # Should not raise immediately
        result = relative_economic_value(fcst, obs, [0.5], weights=bad_weights, check_args=False)
        assert isinstance(result.data, da.Array)

        # Should raise on compute
        with pytest.raises(ValueError, match=ERROR_INVALID_WEIGHTS):
            result.compute()

    def test_weighted_rev_with_valid_dask_weights(self):
        """Test that valid weighted REV works with Dask."""
        fcst = xr.DataArray(da.from_array([0, 1, 1, 0, 1], chunks=2), dims=["time"])
        obs = xr.DataArray(da.from_array([0, 1, 0, 0, 1], chunks=2), dims=["time"])
        weights = xr.DataArray(da.from_array([1.0, 2.0, 1.0, 1.0, 2.0], chunks=2), dims=["time"])

        result = relative_economic_value(fcst, obs, [0.5], weights=weights, check_args=False)
        assert isinstance(result.data, da.Array)

        computed = result.compute()
        # Just verify it completes without error
        assert not np.isnan(computed.values).all()
