"""
Contains unit tests for scores.plotdata.rev_impl
"""

import re
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from scores.plotdata import (
    relative_economic_value,
    relative_economic_value_from_rates,
)
from scores.plotdata.rev_impl import (
    _calculate_rev_core,
    _create_output_dataset,
    _validate_dimensions,
    calculate_base_rate,
    check_monotonic_array,
)
from scores.utils import ERROR_INVALID_WEIGHTS
from tests.plotdata import rev_test_data as rtd


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
                [
                    [1.0],
                    [-2.0],
                    [-1.0],
                    [1.0],
                    [-2.0],
                ],  # REV for each space point after reducing time
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
                [
                    [1.0],
                    [-0.5],
                    [-1],
                    [1.0],
                    [-0.5],
                ],  # REV for each space point after reducing time
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

        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], reduce_dims="time")

        expected = xr.DataArray(
            expected_rev,
            dims=["space", "cost_loss_ratio"],
            coords={"space": space_coord, "cost_loss_ratio": [0.5]},
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        "fcst_dims,obs_dims,fcst_data,obs_data,expected_rev",
        [
            # obs missing space dimension - broadcasts over space
            (
                ["time", "space"],
                ["time"],
                [
                    [1, 0, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                ],  # time=0, space=[0,1,2,3,4]  # time=1  # time=2
                [1, 1, 0],  # time=[0,1,2]
                [[1.0], [-1.0], [-1.0], [1.0], [-1.0]],  # REV for each space point
            ),
            # fcst missing space dimension - broadcasts over space
            (
                ["time"],
                ["time", "space"],
                [1, 1, 0],  # time=[0,1,2]
                [
                    [1, 0, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                ],  # time=0, space=[0,1,2,3,4]  # time=1  # time=2
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

        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], preserve_dims="space")

        expected = xr.DataArray(
            expected_rev,
            dims=["space", "cost_loss_ratio"],
            coords={"space": space_coord, "cost_loss_ratio": [0.5]},
        )

        xr.testing.assert_allclose(actual, expected)


class TestCalculateBaseRate:
    """Tests for base rate calculation."""

    def test_simple_mean(self):
        """Test basic mean calculation."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        actual = calculate_base_rate(obs)
        expected = xr.DataArray(data=0.5)
        xr.testing.assert_allclose(expected, actual)

    def test_with_weights_spanning_all_dims(self):
        """Test weighted mean when weights span all reduction dims."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, 2, 1, 2], dims=["time"])

        actual = calculate_base_rate(obs, weights=weights)
        expected_val = (0 * 1 + 1 * 2 + 0 * 1 + 1 * 2) / (1 + 2 + 1 + 2)
        expected = xr.DataArray(expected_val)

        xr.testing.assert_allclose(actual, expected)

    def test_base_rate_ignores_weights_when_dims_not_matching(self):
        """Test that weights are ignored when their dims don't match reduce_dims."""
        # Create obs with a dimension 'time'
        obs = xr.DataArray([1, 0, 1, 1], dims="time")

        # Create weights with a different dimension, 'lat'
        weights = xr.DataArray([0.1, 0.2], dims="lat")

        # Reduce over 'time' (default)
        actual = calculate_base_rate(obs, weights=weights)

        # Should be simple mean of obs since weights dims don't match reduce_dims
        expected = xr.DataArray(3 / 4)

        xr.testing.assert_allclose(actual, expected)

    def test_multidimensional_with_partial_weights(self):
        """Test when weights only span some dims being reduced."""
        obs = xr.DataArray([[0, 1], [1, 0], [1, 1]], dims=["time", "space"])
        weights = xr.DataArray([1, 2], dims=["space"])

        # Reduce all dims
        actual = calculate_base_rate(obs, reduce_dims=["time", "space"], weights=weights)

        # Weighted mean over space, then mean over time
        expected_val = np.mean([(0 * 1 + 1 * 2) / 3, (1 * 1 + 0 * 2) / 3, (1 * 1 + 1 * 2) / 3])
        expected = xr.DataArray(expected_val)

        xr.testing.assert_allclose(actual, expected)

    def test_preserve_dims(self):
        """Test preserve_dims parameter."""
        obs = xr.DataArray([[0, 1], [1, 0]], dims=["time", "space"])

        actual = calculate_base_rate(obs, preserve_dims="space")
        expected = xr.DataArray([0.5, 0.5], dims=["space"])
        xr.testing.assert_allclose(actual, expected)


class TestScienceCalculations:
    """Tests for core scientific calculations in REV."""

    def test_perfect_forecast(self, make_contingency_data):
        """Perfect forecast (all correct) has REV=1 at any cost-loss ratio."""
        fcst, obs = make_contingency_data(hits=2, misses=0, false_alarms=0, correct_negatives=2)

        for alpha in [0.2, 0.5, 0.8]:
            actual = relative_economic_value(fcst, obs, cost_loss_ratios=[alpha])
            expected = xr.DataArray([1.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [alpha]})
            xr.testing.assert_allclose(actual, expected)

    def test_always_no_forecast(self, make_contingency_data):
        """Always predicting 'no' gives REV=0 at alpha=obar, negative otherwise."""
        fcst, obs = make_contingency_data(0, 2, 0, 2)

        xr.testing.assert_allclose(
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.5]),
            xr.DataArray([0.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]}),
        )
        xr.testing.assert_allclose(
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.2]),
            xr.DataArray([-3.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.2]}),
        )

    def test_always_yes_forecast(self, make_contingency_data):
        """Always predicting 'yes' gives REV=0 at alpha=obar, negative otherwise."""
        fcst, obs = make_contingency_data(2, 0, 2, 0)

        xr.testing.assert_allclose(
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.5]),
            xr.DataArray([0.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]}),
        )
        xr.testing.assert_allclose(
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.8]),
            xr.DataArray([-3.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.8]}),
        )

    def test_anti_correlated_forecast(self):
        """Forecast that's systematically wrong has REV < -1."""
        # This doesn't fit contingency model - fcst and obs are opposite
        fcst = xr.DataArray([0, 1, 0, 1], dims=["time"])
        obs = xr.DataArray([1, 0, 1, 0], dims=["time"])

        # At alpha=0.5: REV = -1.0
        assert relative_economic_value(fcst, obs, cost_loss_ratios=[0.5]).item() == -1.0
        # At extreme alphas: even worse
        assert relative_economic_value(fcst, obs, cost_loss_ratios=[0.2]).item() == -4.0

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
        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[alpha])
        assert actual.item() == pytest.approx(expected)

    def test_undefined_when_obar_is_zero_or_one(self, make_contingency_data):
        """REV undefined when base_rate is 0 or 1 (no variance in obs)."""
        # obar = 0: no events
        fcst, obs = make_contingency_data(0, 0, 2, 2)
        assert np.isnan(relative_economic_value(fcst, obs, cost_loss_ratios=[0.2]).item())

        # obar = 1: all events
        fcst, obs = make_contingency_data(2, 2, 0, 0)
        assert np.isnan(relative_economic_value(fcst, obs, cost_loss_ratios=[0.2]).item())

    def test_undefined_at_extreme_cost_loss(self, make_contingency_data):
        """REV undefined at cost_loss_ratio = 0 or 1."""
        fcst, obs = make_contingency_data(2, 0, 0, 2)  # perfect forecast

        assert np.isnan(relative_economic_value(fcst, obs, cost_loss_ratios=[0.0]).item())
        assert np.isnan(relative_economic_value(fcst, obs, cost_loss_ratios=[1.0]).item())

    def test_nan_values_excluded_from_calculation(self):
        """NaN values in fcst or obs are excluded pairwise."""
        # After removing NaNs: fcst=[1,1,0,0], obs=[1,0,1,0]
        # -> H=1, M=1, FA=1, CN=1 -> no skill -> REV=0 at alpha=0.5
        fcst = xr.DataArray([1, 1, 0, 0, 1, np.nan], dims=["time"])
        obs = xr.DataArray([1, 0, 1, 0, np.nan, 0], dims=["time"])

        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5])
        assert actual.item() == 0.0

    def test_multiple_cost_loss_ratios(self, make_contingency_data):
        """Test with multiple cost-loss ratios spanning the full range."""
        # Simple, verifiable data:
        # 10 samples: 6 hits, 2 misses, 1 false alarm, 1 correct negative
        # POD = 6/8 = 0.75, POFD = 1/2 = 0.5, base_rate = 8/10 = 0.8
        binary_fcst, obs = make_contingency_data(6, 2, 1, 1)

        cost_loss_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        actual = _calculate_rev_core(
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

        xr.testing.assert_allclose(actual, expected)

    def test_single_float_cost_loss_is_converted_to_length_one_coordinate(self):
        """
        Passing a single float for cost_loss_ratios should produce an output whose
        cost_loss_ratio coordinate has length 1 and contains that value.
        """
        fcst = rtd.PROB_FCST_DA
        obs = rtd.BINARY_DA
        single_alpha = 0.3

        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=single_alpha,
            probability_threshold=[0.5],  # to trigger probabilistic branch
            check_args=True,
        )

        expected = xr.DataArray(
            [[1.0]],
            dims=["probability_threshold", "cost_loss_ratio"],
            coords={"probability_threshold": [0.5], "cost_loss_ratio": [0.3]},
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize("scalar_value", [0, 0.0, 1, 1.0])  # test a mix of int and float values
    def test_cost_loss_is_converted_to_length_one_coordinate(self, scalar_value):
        """
        Passing an integer cost_loss_ratio should be converted to a length-1 coordinate.
        Also covers int -> list handling branch.
        """
        fcst = rtd.BINARY_DA
        obs = fcst.copy()
        actual = relative_economic_value(fcst, obs, cost_loss_ratios=scalar_value)

        expected = xr.DataArray(
            [np.nan],
            dims=["cost_loss_ratio"],
            coords={"cost_loss_ratio": [scalar_value]},
        )

        xr.testing.assert_identical(actual, expected)


class TestREVSpecialFeatures:
    """Tests for special features of the REV implementation."""

    def test_default_cost_loss_ratios(self):
        """When cost_loss_ratios is omitted, defaults to 0.01..0.99 (99 values)."""
        fcst = xr.DataArray([0, 1, 1, 0, 1], dims=["time"])
        obs = xr.DataArray([0, 1, 0, 0, 1], dims=["time"])

        result = relative_economic_value(fcst, obs)

        expected_ratios = list(np.arange(0.01, 1.0, 0.01))
        assert result.dims == ("cost_loss_ratio",)
        assert len(result.cost_loss_ratio) == 99
        np.testing.assert_allclose(result.cost_loss_ratio.values, expected_ratios)

    def test_probabilistic_single_threshold(self):
        """Test with single threshold"""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.1, 0.9], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0, 1], dims=["time"])
        threshold = 0.5
        cost_loss_ratios = [0.2, 0.5, 0.8]

        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            probability_threshold=threshold,
            probability_threshold_outputs=[threshold],
        )
        expected = xr.Dataset(
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

        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            probability_threshold=thresholds,
            probability_threshold_outputs=[0.5],
        )
        expected = xr.Dataset(
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
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            probability_threshold=thresholds,
            probability_threshold_outputs=[0.4, 0.8],
        )
        expected = xr.Dataset(
            data_vars={
                "threshold_0_4": (["cost_loss_ratio"], np.array([1.0, 1.0])),
                "threshold_0_8": (["cost_loss_ratio"], np.array([-4 / 3, 0.5])),
            },
            coords={"cost_loss_ratio": np.array(cost_loss_ratios)},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_probabilistic_maximum_output(self):
        """Test maximum value output"""
        fcst = xr.DataArray(
            [0.75] * 4 + [0.25] * 3 + [0.75] * 2 + [0.25] * 1,
            dims=["time"],
            coords={"time": np.arange(10)},
        )
        obs = xr.DataArray(
            [1] * 4 + [0] * 3 + [0] * 2 + [1] * 1,
            dims=["time"],
            coords={"time": np.arange(10)},
        )
        thresholds = np.arange(0.1, 1.0, 0.1)
        cost_loss_ratios = [0.2, 0.4, 0.6, 0.8]

        actual_full_result = relative_economic_value(
            fcst, obs, cost_loss_ratios=cost_loss_ratios, probability_threshold=thresholds
        )

        actual_max_result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            probability_threshold=thresholds,
            generate_maximum_rev=True,
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
            dims=["probability_threshold", "cost_loss_ratio"],
            coords={
                "probability_threshold": np.arange(0.1, 1.0, 0.1),
                "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8],
            },
        )

        xr.testing.assert_allclose(expected_full_result, actual_full_result)

        expected_max_result = xr.Dataset(
            data_vars={"maximum": (["cost_loss_ratio"], [0.0, 0.3, 0.2, 0.0])},
            coords={"cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )

        xr.testing.assert_allclose(expected_max_result, actual_max_result)

    def test_probabilistic_equilibrium_point_output(self):
        """Test maximum and equilibrium point output extraction (diagonal extraction)"""
        fcst = xr.DataArray(
            [0.75] * 4 + [0.25] * 3 + [0.75] * 2 + [0.25] * 1,
            dims=["time"],
            coords={"time": np.arange(10)},
        )
        obs = xr.DataArray(
            [1] * 4 + [0] * 3 + [0] * 2 + [1] * 1,
            dims=["time"],
            coords={"time": np.arange(10)},
        )
        thresholds = [0.2, 0.4, 0.6, 0.8]
        cost_loss_ratios = [0.2, 0.4, 0.6, 0.8]

        actual_full_result = relative_economic_value(
            fcst, obs, cost_loss_ratios=cost_loss_ratios, probability_threshold=thresholds
        )

        actual_rational_result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            probability_threshold=thresholds,
            generate_maximum_rev=True,
            generate_equilibrium_point_rev=True,
        )

        expected_full_result_values = np.array(
            [
                [0.0, 0.0, -0.5, -3.0],
                [-0.2, 0.3, 0.2, -0.8],
                [-0.2, 0.3, 0.2, -0.8],
                [-3.0, -0.5, 0.0, 0.0],
            ]
        )

        expected_full_result = xr.DataArray(
            expected_full_result_values,
            dims=["probability_threshold", "cost_loss_ratio"],
            coords={
                "probability_threshold": [0.2, 0.4, 0.6, 0.8],
                "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8],
            },
        )

        expected_rational_result = xr.Dataset(
            data_vars={
                "maximum": (["cost_loss_ratio"], [0.0, 0.3, 0.2, 0.0]),
                "equilibrium_point": (["cost_loss_ratio"], [0.0, 0.3, 0.2, 0.0]),
            },
            coords={
                "probability_threshold": (["cost_loss_ratio"], [0.2, 0.4, 0.6, 0.8]),
                "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8],
            },
        )
        xr.testing.assert_allclose(expected_full_result, actual_full_result)
        xr.testing.assert_allclose(expected_rational_result, actual_rational_result)

    def test_equilibrium_point_no_threshold_in_coords(self):
        """Test equilibrium_point path where threshold coord somehow not present"""
        # Create a REV array with threshold as dimension but use isel instead of sel
        # to avoid coordinate preservation
        rev = xr.DataArray(
            np.array([[0.2, 0.5], [0.7, 0.9]]),
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.1, 0.2], "cost_loss_ratio": [0.1, 0.2]},
        )

        with mock.patch("xarray.concat") as mock_concat:
            # Make concat return a DataArray without threshold in coords
            mock_result = xr.DataArray(
                [0.2, 0.9],
                dims=["cost_loss_ratio"],
                coords={"cost_loss_ratio": [0.1, 0.2]},
            )
            mock_concat.return_value = mock_result

            actual = _create_output_dataset(
                rev=rev,
                thresholds=[0.1, 0.2],
                cost_loss_ratios=[0.1, 0.2],
                generate_maximum_rev=False,
                generate_equilibrium_point_rev=True,
                threshold_outputs=None,
                threshold_dim="threshold",
                cost_loss_dim="cost_loss_ratio",
            )

            assert "equilibrium_point" in actual

    @pytest.mark.parametrize(
        "threshold_dim, cost_loss_dim, kwargs, expected_dims_by_var",
        [
            ("my_threshold", "cost_loss_ratio", {}, {"result": ("my_threshold", "cost_loss_ratio")}),
            ("threshold", "alpha", {}, {"result": ("threshold", "alpha")}),
            ("decision_threshold", "alpha", {}, {"result": ("decision_threshold", "alpha")}),
            (
                "decision_threshold",
                "alpha",
                {"generate_maximum_rev": True, "generate_equilibrium_point_rev": True},
                {"maximum": ("alpha",), "equilibrium_point": ("alpha",)},
            ),
            ("decision_threshold", "alpha", {"probability_threshold_outputs": [0.5]}, {"threshold_0_5": ("alpha",)}),
        ],
    )
    def test_custom_dimension_names(self, threshold_dim, cost_loss_dim, kwargs, expected_dims_by_var):
        """Test that custom dimension names appear correctly in all output forms."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])
        matching_values = [0.3, 0.5, 0.7]

        result = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=matching_values,
            probability_threshold=matching_values,
            probability_threshold_dim=threshold_dim,
            cost_loss_dim=cost_loss_dim,
            **kwargs,
        )

        for var, expected_dims in expected_dims_by_var.items():
            actual = result if var == "result" else result[var]
            assert actual.dims == expected_dims


class TestWeights:
    """Tests for handling of time/spatial weights in REV calculations."""

    def test_equal_weights_same_as_unweighted(self, make_contingency_data):
        """When all weights are equal, weighted result should match unweighted."""
        fcst, obs = make_contingency_data(2, 2, 2, 2)

        weights = xr.DataArray([18.342] * 8, dims=["time"])

        rev_weighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights)
        rev_unweighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5])

        xr.testing.assert_allclose(rev_weighted, rev_unweighted)

    def test_spatial_weights_broadcast(self):
        """Test that latitude weights broadcast correctly over time dimension."""
        # Two latitudes, 4 timesteps each - small enough to verify by hand
        #
        # Lat 60°: Perfect forecast (2 hits, 2 correct negatives)
        # Lat 30°: No skill (1 hit, 1 miss, 1 FA, 1 CN)

        fcst = xr.DataArray(
            [
                [1, 0, 1, 0],
                [1, 0, 1, 0],
            ],  # lat 60: fcst matches obs perfectly  # lat 30: fcst uncorrelated with obs
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
            unweighted,
            xr.DataArray([0.5], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]}),
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
            [
                [[1, 0], [0, 1], [1, 0], [0, 1]],
                [[1, 0], [0, 1], [1, 0], [0, 1]],
            ],  # lat 60  # lat 30
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

        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights, preserve_dims=["lon"])

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
            [[0.366], [-0.366]],
            dims=["lon", "cost_loss_ratio"],
            coords={"lon": [0, 180], "cost_loss_ratio": [0.5]},
        )
        xr.testing.assert_allclose(actual, expected, atol=0.001)


class TestDatasetInputs:
    """Test that REV works with xr.Dataset inputs."""

    def test_forecast_as_dataset(self):
        """Test with forecast as Dataset."""
        fcst_ds = xr.Dataset(
            {
                "ecmwf": xr.DataArray([0, 1, 1, 0], dims=["time"]),
                "access": xr.DataArray([1, 0, 0, 1], dims=["time"]),
            }
        )
        obs = xr.DataArray([0, 1, 1, 0], dims=["time"])

        actual = relative_economic_value(fcst_ds, obs, cost_loss_ratios=[0.5])

        expected = xr.Dataset(
            data_vars={
                "ecmwf": (["cost_loss_ratio"], [1.0]),
                "access": (["cost_loss_ratio"], [-1.0]),
            },
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

        actual = relative_economic_value(fcst, obs_ds, cost_loss_ratios=[0.3, 0.7], probability_threshold=[0.5])
        expected = xr.Dataset(
            data_vars={
                "station_data": (["probability_threshold", "cost_loss_ratio"], [[1.0, 1.0]]),
                "radar_data": (["probability_threshold", "cost_loss_ratio"], [[-7 / 3, -7 / 3]]),
            },
            coords={"probability_threshold": [0.5], "cost_loss_ratio": [0.3, 0.7]},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_both_as_dataset(self):
        """Test with both as Dataset."""
        fcst_ds = xr.Dataset(
            {
                "ecmwf": xr.DataArray([1, 1, 1, 0], dims=["time"]),
                "access": xr.DataArray([0, 0, 0, 1], dims=["time"]),
            }
        )
        obs_ds = xr.Dataset(
            {
                "station_data": xr.DataArray([0, 0, 1, 1], dims=["time"]),
                "radar_data": xr.DataArray([1, 1, 0, 0], dims=["time"]),
            }
        )

        actual = relative_economic_value(fcst_ds, obs_ds, cost_loss_ratios=[0.3, 0.7], probability_threshold=[0.5])

        expected = xr.Dataset(
            data_vars={
                "access__vs__radar_data": (
                    ["probability_threshold", "cost_loss_ratio"],
                    np.array([[-11 / 6, -7 / 6]]),
                ),
                "access__vs__station_data": (
                    ["probability_threshold", "cost_loss_ratio"],
                    np.array([[-1 / 6, 0.5]]),
                ),
                "ecmwf__vs__radar_data": (
                    ["probability_threshold", "cost_loss_ratio"],
                    np.array([[0.5, -1 / 6]]),
                ),
                "ecmwf__vs__station_data": (
                    ["probability_threshold", "cost_loss_ratio"],
                    np.array([[-7 / 6, -11 / 6]]),
                ),
            },
            coords={
                "probability_threshold": [0.5],
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
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights_ds)

    def test_pod_as_dataset(self):
        """Test with POD as Dataset."""
        pod_ds = xr.Dataset(
            {
                "model_a": xr.DataArray([0.8, 0.6], dims=["threshold"]),
                "model_b": xr.DataArray([0.9, 0.5], dims=["threshold"]),
            }
        )
        pofd_ds = xr.Dataset(
            {
                "model_a": xr.DataArray([0.2, 0.1], dims=["threshold"]),
                "model_b": xr.DataArray([0.2, 0.1], dims=["threshold"]),
            }
        )

        base_rate = xr.DataArray(0.3)

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, base_rate, [0.3, 0.7])

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
        base_rate = xr.DataArray(0.4)

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, base_rate, [0.5])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"model_a", "model_b"}

    def test_relative_economic_value_from_rates_type_error(self):
        """
        Tests the error when mixing datasets and dataarrays
        """
        # Create a DataArray for POD
        pod = xr.DataArray([0.8], dims=["threshold"], coords={"threshold": [0.5]})

        # Create a Dataset for POFD (mismatched type)
        pofd = xr.Dataset({"var": (["threshold"], [0.2])}, coords={"threshold": [0.5]})

        base_rate = xr.DataArray(0.3)
        cost_loss_ratios = [0.5]

        # Verify that mixing DataArray and Dataset raises TypeError
        with pytest.raises(
            TypeError,
            match="Both pod and pofd must be either xarray DataArrays or xarray Datasets",
        ):
            relative_economic_value_from_rates(pod, pofd, base_rate, cost_loss_ratios)

    def test_pod_and_base_rate_as_dataset(self):
        """Test with POD and base_rate as Datasets."""
        pod_ds = xr.Dataset(
            {
                "region_1": xr.DataArray([0.8], dims=["threshold"]),
                "region_2": xr.DataArray([0.6], dims=["threshold"]),
            }
        )
        pofd_ds = xr.Dataset(
            {
                "region_1": xr.DataArray([0.15], dims=["threshold"]),
                "region_2": xr.DataArray([0.15], dims=["threshold"]),
            }
        )
        base_rate_ds = xr.Dataset({"region_1": xr.DataArray(0.3), "region_2": xr.DataArray(0.45)})

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, base_rate_ds, [0.3, 0.7])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"region_1", "region_2"}

    def test_all_as_dataset(self):
        """Test with POD, POFD, and base_rate all as Datasets."""
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
        base_rate_ds = xr.Dataset({"var_1": xr.DataArray(0.25), "var_2": xr.DataArray(0.35)})

        actual = relative_economic_value_from_rates(pod_ds, pofd_ds, base_rate_ds, [0.2, 0.5, 0.8])

        assert isinstance(actual, xr.Dataset)
        assert set(actual.data_vars) == {"var_1", "var_2"}
        assert actual["var_1"].dims == ("cost_loss_ratio", "threshold")
        assert len(actual.cost_loss_ratio) == 3

    def test_dataset_preserves_numeric_results(self):
        """Test that Dataset processing produces correct numeric values."""
        # Use simple values where we can verify the math
        pod_scalar = xr.DataArray(1.0)  # Perfect detection
        pofd_scalar = xr.DataArray(0.0)  # No false alarms
        base_rate_scalar = xr.DataArray(0.5)

        # Calculate with scalars
        actual_scalar = relative_economic_value_from_rates(pod_scalar, pofd_scalar, base_rate_scalar, [0.5])

        # Calculate with Dataset
        pod_ds = xr.Dataset({"test": pod_scalar})
        pofd_ds = xr.Dataset({"test": pofd_scalar})
        actual_dataset = relative_economic_value_from_rates(pod_ds, pofd_ds, base_rate_scalar, [0.5])

        xr.testing.assert_allclose(actual_dataset["test"], actual_scalar)


class TestErrorHandling:
    """Tests that check that error handling is done correctly"""

    @pytest.mark.parametrize(
        "fcst_data,obs_data,cost_loss_ratios,probability_threshold,probability_threshold_outputs,expected_error",
        [
            # Probabilistic forecasts without probability_threshold
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], None, None, "contains values that are not in the set {0, 1, np.nan}"),
            # Cost-loss ratios out of range
            ([0, 1, 1], [0, 1, 0], [-0.1, 0.5, 1.2], None, None, "between 0 and 1"),
            # Cost-loss ratios not monotonic
            ([0, 1, 1], [0, 1, 0], [0.5, 0.3, 0.7], None, None, "monotonically increasing"),
            # Threshold values out of range
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], [-0.1, 0.5], None, "between 0 and 1"),
            # Threshold values not monotonic
            ([0.2, 0.8, 0.6], [0, 1, 1], [0.5], [0.7, 0.3], None, "monotonically increasing"),
            # Invalid observation values
            ([0, 1, 1], [0, 1, 2], [0.5], None, None, "contains values that are not in the set {0, 1, np.nan}"),
            # Invalid forecast values
            ([0, 1, 2], [0, 1, 1], [0.5], None, None, "contains values that are not in the set {0, 1, np.nan}"),
            # probability_threshold_outputs not in probability_threshold
            (
                [0.2, 0.8],
                [0, 1],
                [0.5],
                [0.2, 0.5],
                [0.7],
                "values in probability_threshold_outputs must be in the supplied probability_threshold parameter",
            ),
            # probability_threshold_outputs without probability_threshold
            (
                [0, 1],
                [0, 1],
                [0.5],
                None,
                [0.5],
                "probability_threshold_outputs can only be used when probability_threshold parameter is provided",
            ),
            # Forecast outside [0,1] when probability_threshold provided
            (
                [-0.1, 1.2],
                [0, 1],
                [0.1, 0.5],
                [0.5],
                None,
                "When probability_threshold is provided, fcst must contain values between 0 and 1",
            ),
        ],
        ids=[
            "probabilistic_without_threshold",
            "cost_loss_out_of_range",
            "cost_loss_not_monotonic",
            "threshold_out_of_range",
            "threshold_not_monotonic",
            "invalid_obs_values",
            "invalid_fcst_values",
            "threshold_outputs_not_in_threshold",
            "threshold_outputs_without_threshold",
            "fcst_outside_01_with_threshold",
        ],
    )
    def test_input_validation(
        self,
        fcst_data,
        obs_data,
        cost_loss_ratios,
        probability_threshold,
        probability_threshold_outputs,
        expected_error,
    ):
        """
        Test that relative_economic_value validates inputs correctly.

        Validates that the function raises ValueError for:
        - Probabilistic forecasts (values between 0 and 1) without probability_threshold parameter
        - Cost-loss ratios None, outside [0, 1] range or not strictly monotonically increasing
        - Threshold values outside [0, 1] range or not strictly monotonically increasing
        - Forecast or observation values outside {0, 1, NaN}
        """
        fcst = xr.DataArray(fcst_data, dims=["time"])
        obs = xr.DataArray(obs_data, dims=["time"])

        with pytest.raises(ValueError, match=expected_error):
            relative_economic_value(
                fcst,
                obs,
                cost_loss_ratios=cost_loss_ratios,
                probability_threshold=probability_threshold,
                probability_threshold_outputs=probability_threshold_outputs,
            )

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

    @pytest.mark.parametrize(
        "array_name,forbidden_dim",
        [
            ("fcst", "probability_threshold"),
            ("fcst", "cost_loss_ratio"),
            ("obs", "probability_threshold"),
            ("obs", "cost_loss_ratio"),
            ("weights", "probability_threshold"),
            ("weights", "cost_loss_ratio"),
        ],
    )
    def test_forbidden_dimensions(self, array_name, forbidden_dim):
        """Test that 'probability_threshold' and 'cost_loss_ratio' cannot be dimensions in fcst, obs, or weights."""
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
            relative_economic_value(fcst, obs, cost_loss_ratios=[0.5], weights=weights)

    def test_value_without_matching_thresholds(self):
        """Test that 'equilibrium_point' output requires matching thresholds"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        with pytest.raises(ValueError, match="identical"):
            relative_economic_value(
                fcst,
                obs,
                cost_loss_ratios=[0.3, 0.5],
                probability_threshold=[0.2, 0.5],
                generate_equilibrium_point_rev=True,
            )

    @pytest.mark.parametrize(
        "arr, expected_exception, expected_error",
        [
            ([-0.1, 0.5], ValueError, "array values should be between 0 and 1."),
            ([0.0, 1.5], ValueError, "array values should be between 0 and 1."),
            (["not", "numbers"], TypeError, "could not convert array into a numpy ndarray of floats"),
            ([[0.1, 0.2], [0.3, 0.4]], ValueError, "array must be one-dimensional"),
        ],
    )
    def test_check_monotonic_array_invalid_inputs(self, arr, expected_exception, expected_error):
        """Test the monotonic array checker rejects invalid inputs."""
        with pytest.raises(expected_exception, match=expected_error):
            check_monotonic_array(arr)

    @pytest.mark.parametrize("which_input", ["pod", "pofd", "base_rate"])
    def test_cost_loss_ratio_dim_in_inputs_raises(self, which_input):
        """create a DataArray that contains the forbidden dimension name"""
        da_with_forbidden_dim = xr.DataArray(
            np.array([0.5, 0.5]),
            dims=("cost_loss_ratio",),
            coords={"cost_loss_ratio": [0.1, 0.9]},
        )

        # other inputs are normal scalar-like arrays
        pod = rtd.SCALAR_DA
        pofd = pod.copy()
        base_rate = pod.copy()

        # replace the selected input with the bad one
        if which_input == "pod":
            pod = da_with_forbidden_dim
        elif which_input == "pofd":
            pofd = da_with_forbidden_dim
        else:
            base_rate = da_with_forbidden_dim

        good_cost_loss = np.array([0.1, 0.5, 0.9])  # valid monotonic ratios

        with pytest.raises(ValueError) as excinfo:
            relative_economic_value_from_rates(pod, pofd, base_rate, good_cost_loss)

        assert "dimension 'cost_loss_ratio' must not be in input data" in str(excinfo.value)

    def test_check_args_false_skips_validation_allows_probabilistic_without_threshold(
        self,
    ):
        """
        When check_args=False, the function should skip input validation and proceed.
        This test asserts no ValueError is raised for a probabilistic fcst without threshold.
        """
        fcst = rtd.PROB_FCST_DA
        obs = rtd.BINARY_DA

        # Nothing should raise; we expect an xr.DataArray back (or Dataset depending on other args)
        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=[0.1, 0.5],
            probability_threshold=None,
            check_args=False,  # skip validation
        )

        expected = xr.DataArray(
            [np.nan, np.nan],
            dims=["cost_loss_ratio"],
            coords={"cost_loss_ratio": [0.1, 0.5]},
        )
        xr.testing.assert_allclose(actual, expected)

    def test_equilibrium_point_without_threshold_raises(self, make_contingency_data):
        """Test derived metrics 'equilibrium_point' without threshold raises ValueError"""
        fcst, obs = make_contingency_data(1, 1, 1, 1)

        with pytest.raises(
            ValueError,
            match="generate_equilibrium_point_rev=True can only be used when probability_threshold parameter",
        ):
            relative_economic_value(
                fcst=fcst,
                obs=obs,
                cost_loss_ratios=[0.2, 0.5],
                generate_equilibrium_point_rev=True,
            )

    def test_valid_no_weights(self):
        """No error when inputs have no forbidden dims and weights is None."""
        fcst = xr.DataArray([0, 1], dims=["time"])
        obs = xr.DataArray([0, 1], dims=["time"])
        _validate_dimensions(fcst, obs, None, "threshold", "cost_loss_ratio")

    def test_valid_with_weights(self):
        """No error when weights present but carries no forbidden dims."""
        fcst = xr.DataArray([0, 1], dims=["time"])
        obs = xr.DataArray([0, 1], dims=["time"])
        weights = xr.DataArray([1.0, 1.0], dims=["time"])
        _validate_dimensions(fcst, obs, weights, "threshold", "cost_loss_ratio")

    @pytest.mark.parametrize(
        "which_input,forbidden_dim",
        [
            ("fcst", "threshold"),
            ("fcst", "cost_loss_ratio"),
            ("obs", "threshold"),
            ("obs", "cost_loss_ratio"),
            ("weights", "threshold"),
            ("weights", "cost_loss_ratio"),
        ],
        ids=[
            "fcst_has_threshold_dim",
            "fcst_has_cost_loss_dim",
            "obs_has_threshold_dim",
            "obs_has_cost_loss_dim",
            "weights_has_threshold_dim",
            "weights_has_cost_loss_dim",
        ],
    )
    def test_forbidden_dim_raises(self, which_input, forbidden_dim):
        """Raises ValueError when threshold_dim or cost_loss_dim appears in fcst, obs, or weights."""
        fcst = xr.DataArray([0, 1], dims=["time"])
        obs = xr.DataArray([0, 1], dims=["time"])
        weights = xr.DataArray([1.0, 1.0], dims=["time"])

        if which_input == "fcst":
            fcst = xr.DataArray([0, 1], dims=[forbidden_dim])
        elif which_input == "obs":
            obs = xr.DataArray([0, 1], dims=[forbidden_dim])
        else:
            weights = xr.DataArray([1.0, 1.0], dims=[forbidden_dim])

        with pytest.raises(ValueError, match=f"'{forbidden_dim}' cannot be a dimension in {which_input}"):
            _validate_dimensions(fcst, obs, weights, "threshold", "cost_loss_ratio")


class TestLegacyJive:
    """Tests that were part of the original Jive code"""

    @pytest.mark.parametrize(
        (
            "fcst",
            "obs",
            "thresholds",
            "cost_loss_ratios",
            "preserve_dims",
            "generate_maximum_rev",
            "threshold_outputs",
            "expected",
        ),
        [
            # 0: 3-D, keep one dim, mask_extreme_values=False
            (
                rtd.FCST_2X3X2_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.3, 1],
                [0, 0.2, 0.5, 0.8, 1],
                ["lead_day"],
                True,
                [0, 0.3, 1],
                rtd.EXP_PREV_CASE0,
            ),
            # 2: 3-D, keep no dims, mask_extreme_values=False
            (
                rtd.FCST_2X3X2_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.3, 1],
                [0, 0.2, 0.5, 0.8, 1],
                None,
                True,
                None,
                rtd.EXP_PREV_CASE2,
            ),
            # 3: 3-D, keep one dim, one threshold, one cost_loss_ratio
            # SPOT-CHECKED by DG
            (
                rtd.FCST_2X3X2_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0.3],
                [0.5],
                ["lead_day"],
                False,
                [0.3],
                rtd.EXP_PREV_CASE3,
            ),
            # 4: mis-aligned fcst & obs
            (
                rtd.FCST_2X3X2_WITH_NAN_MISALIGNED,
                rtd.OBS_3X3_WITH_NAN_MISALIGNED,
                [0.3],
                [0.5],
                ["lead_day"],
                False,
                [0.3],
                rtd.EXP_PREV_CASE3,
            ),
            (
                rtd.FCST_2X3X2_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.3, 1],
                [0, 0.2, 0.5, 0.8, 1],
                {"lead_day"},
                True,
                [0, 0.3, 1],
                rtd.EXP_PREV_CASE0,
            ),
        ],
    )
    def test_jive_probabilistic_relative_economic_value(
        self,
        fcst,
        obs,
        thresholds,
        cost_loss_ratios,
        preserve_dims,
        generate_maximum_rev,
        threshold_outputs,
        expected,
    ):
        """
        Tests that probabilistic_relative_economic_value returns correct result
        """
        actual = relative_economic_value(
            fcst,
            obs,
            probability_threshold=thresholds,
            cost_loss_ratios=cost_loss_ratios,
            preserve_dims=preserve_dims,
            generate_maximum_rev=generate_maximum_rev,
            probability_threshold_outputs=threshold_outputs,
        )
        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        (
            "fcst",
            "obs",
            "cost_loss_ratios",
            "preserve_dims",
            "expected",
        ),
        [
            # See jive.tests.metrics.standard.test_probabilistic.test_probabilistic_relative_
            # economic_value for SPOT-CHECKED by DG result.
            # 0: 4-D, keep two dims, mask_extreme_values=False
            (
                rtd.DISCRETE_FCST_2X3X2X3_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.2, 0.5, 0.8, 1],
                ["lead_day", "binary_threshold"],
                rtd.EXP_REV_CASE0,
            ),
            # 2: 4-D, keep one dim, mask_extreme_values=False
            (
                rtd.DISCRETE_FCST_2X3X2X3_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.2, 0.5, 0.8, 1],
                ["binary_threshold"],
                rtd.EXP_REV_CASE2,
            ),
            # 3: 3-D, keep no dims, mask_extreme_values=False
            (
                rtd.DISCRETE_FCST_2X3X2_WITH_NAN,
                rtd.OBS_3X3_WITH_NAN,
                [0, 0.2, 0.5, 0.8, 1],
                None,
                rtd.EXP_REV_CASE3,
            ),
            # 4: as no.3, but misaligned coordinates
            (
                rtd.DISCRETE_FCST_2X3X2_WITH_NAN_MISALIGNED,
                rtd.OBS_3X3_WITH_NAN_MISALIGNED,
                [0, 0.2, 0.5, 0.8, 1],
                None,
                rtd.EXP_REV_CASE3,
            ),
            # 5: integer fcst & obs
            (
                rtd.DISCRETE_FCST_3X5_INT,
                rtd.OBS_3X5_INT,
                [0, 0.2, 0.5, 0.8, 1],
                None,
                rtd.EXP_REV_CASE3,
            ),
        ],
    )
    def test_jive_relative_economic_value(self, fcst, obs, cost_loss_ratios, preserve_dims, expected):
        """Tests that relative_economic value returns the correct result"""
        actual = relative_economic_value(
            fcst,
            obs,
            cost_loss_ratios=cost_loss_ratios,
            preserve_dims=preserve_dims,
        )
        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        (
            "pod",
            "pofd",
            "base_rate",
            "cost_loss_ratios",
            "expected",
        ),
        [
            # 0: 3-D, keep one dim
            (
                rtd.HIT_RATE_REV_LEADDAY,
                rtd.FALSE_ALARM_RATE_REV_LEADDAY,
                rtd.OBAR_REV_LEADDAY,
                [0, 0.2, 0.5, 0.8, 1],
                rtd.EXP_REV_CASE0.transpose("lead_day", "cost_loss_ratio", "binary_threshold"),
            ),
            # 2: 3-D, keep no dims
            (
                rtd.HIT_RATE_REV_NONE,
                rtd.FALSE_ALARM_RATE_REV_NONE,
                rtd.OBAR_REV_NONE,
                [0, 0.2, 0.5, 0.8, 1],
                rtd.EXP_REV_CASE2.transpose("cost_loss_ratio", "binary_threshold"),
            ),
        ],
    )
    def test_jive_relative_economic_value_from_rates(self, pod, pofd, base_rate, cost_loss_ratios, expected):
        """Tests that relative_economic_value_from_rates returns the correct result"""
        actual = relative_economic_value_from_rates(
            pod,
            pofd,
            base_rate,
            cost_loss_ratios,
        )
        xr.testing.assert_allclose(actual, expected)
