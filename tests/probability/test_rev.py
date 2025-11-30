# pylint: disable=too-many-lines
"""
Contains unit tests for scores.probability.rev_impl
"""

try:
    import dask.array as da

    HAS_DASK = True  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    HAS_DASK = False  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover


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

# Module-level test data
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


class TestBroadcastingAndDimensionHandling:
    """Tests for broadcasting and dimension handling in REV calculations."""

    @pytest.mark.parametrize(
        "fcst_dims,obs_dims,rev_values",
        [
            (["time", "space"], ["time"], [[0.0], [-0.5], [-0.75], [0.5], [0.0]]),
            (["time"], ["time", "space"], [[-0.666667], [-1], [0], [0.5], [-1]]),
        ],
        ids=["obs_missing_space", "fcst_missing_space"],
    )
    def test_broadcasting_reducing(self, fcst_dims, obs_dims, rev_values):
        """Test that broadcasting works when reducing over dimensions."""
        np.random.seed(42)

        time_coord = np.arange(10)
        space_coord = np.arange(5)

        fcst_shape = tuple(10 if d == "time" else 5 for d in fcst_dims)
        obs_shape = tuple(10 if d == "time" else 5 for d in obs_dims)

        fcst_coords = {d: time_coord if d == "time" else space_coord for d in fcst_dims}
        obs_coords = {d: time_coord if d == "time" else space_coord for d in obs_dims}

        fcst = xr.DataArray(np.random.binomial(1, 0.7, fcst_shape), dims=fcst_dims, coords=fcst_coords)
        obs = xr.DataArray(np.random.binomial(1, 0.5, obs_shape), dims=obs_dims, coords=obs_coords)

        actual = relative_economic_value(fcst, obs, [0.5], reduce_dims="time")
        expected = xr.DataArray(
            rev_values, dims=["space", "cost_loss_ratio"], coords={"space": [0, 1, 2, 3, 4], "cost_loss_ratio": [0.5]}
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        "fcst_dims,obs_dims,rev_values",
        [
            (["time", "space"], ["time"], [[0.0], [-0.5], [-0.75], [0.5], [0.0]]),
            (["time"], ["time", "space"], [[-0.666667], [-1], [0], [0.5], [-1]]),
        ],
        ids=["obs_missing_space", "fcst_missing_space"],
    )
    def test_broadcasting_keeping(self, fcst_dims, obs_dims, rev_values):
        """Test that broadcasting works when keeping dimensions."""
        np.random.seed(42)

        time_coord = np.arange(10)
        space_coord = np.arange(5)

        fcst_shape = tuple(10 if d == "time" else 5 for d in fcst_dims)
        obs_shape = tuple(10 if d == "time" else 5 for d in obs_dims)

        fcst_coords = {d: time_coord if d == "time" else space_coord for d in fcst_dims}
        obs_coords = {d: time_coord if d == "time" else space_coord for d in obs_dims}

        fcst = xr.DataArray(np.random.binomial(1, 0.7, fcst_shape), dims=fcst_dims, coords=fcst_coords)
        obs = xr.DataArray(np.random.binomial(1, 0.5, obs_shape), dims=obs_dims, coords=obs_coords)

        actual = relative_economic_value(fcst, obs, [0.5], preserve_dims="space")
        expected = xr.DataArray(
            rev_values, dims=["space", "cost_loss_ratio"], coords={"space": [0, 1, 2, 3, 4], "cost_loss_ratio": [0.5]}
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        "reduce_dims,preserve_dims,final_dims",
        [
            (["time"], None, ["lat", "lon", "cost_loss_ratio"]),
            (["lat"], None, ["time", "lon", "cost_loss_ratio"]),
            (["lon"], None, ["time", "lat", "cost_loss_ratio"]),
            (None, ["time"], ["time", "cost_loss_ratio"]),
            (None, ["lat"], ["lat", "cost_loss_ratio"]),
            (None, ["lon"], ["lon", "cost_loss_ratio"]),
            (None, None, ["cost_loss_ratio"]),
            (None, "all", ["time", "lat", "lon", "cost_loss_ratio"]),  # can't be in list
        ],
        ids=[
            "reduce time",
            "reduce lat",
            "reduce lon",
            "preserve time",
            "preserve lat",
            "preserve lon",
            "reduce all",
            "reduce none",
        ],
    )
    def test_reduce_preserve_dims(self, reduce_dims, preserve_dims, final_dims):
        """Test that broadcasting works over a variety of combinations of reducing and preserving."""

        np.random.seed(42)
        fcst = xr.DataArray(np.random.binomial(1, 0.7, (10, 5, 3)), dims=["time", "lat", "lon"])
        obs = xr.DataArray(np.random.binomial(1, 0.5, (10, 5, 3)), dims=["time", "lat", "lon"])

        rev = relative_economic_value(fcst, obs, [0.3, 0.5], reduce_dims=reduce_dims, preserve_dims=preserve_dims)

        assert set(rev.dims) == set(final_dims)


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

    def test_negative_weights_raises_error(self):
        """Test that negative weights raise ValueError when check_args=True."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, -2, 1, 2], dims=["time"])

        with pytest.raises(ValueError, match="'weights' contains negative values"):
            calculate_climatology(obs, weights=weights, check_args=True)

    def test_nan_weights_raises_error(self):
        """Test that NaN weights raise ValueError when check_args=True."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, np.nan, 1, 2], dims=["time"])

        with pytest.raises(ValueError, match="'weights' contains NaN values"):
            calculate_climatology(obs, weights=weights, check_args=True)

    def test_negative_weights_allowed_when_check_args_false(self):
        """Test that negative weights don't raise error when check_args=False."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, -2, 1, 2], dims=["time"])

        # Should not raise
        actual = calculate_climatology(obs, weights=weights, check_args=False)
        expected = xr.DataArray(0.0)
        xr.testing.assert_allclose(actual, expected)

    def test_nan_weights_allowed_when_check_args_false(self):
        """Test that NaN weights don't raise error when check_args=False."""
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])
        weights = xr.DataArray([1, np.nan, 1, 1], dims=["time"])

        # Should not raise.
        # Treats np.nan as 0 weight,
        # although it's debatable what it should be in this case.
        actual = calculate_climatology(obs, weights=weights, check_args=False)
        expected = xr.DataArray(1 / 3)
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

    @pytest.mark.parametrize(
        "fcst,obs,cost_loss,expected",
        [
            # Perfect
            ([0, 1, 0, 1], [0, 1, 0, 1], 0.2, 1.0),
            ([0, 1, 0, 1], [0, 1, 0, 1], 0.5, 1.0),
            ([0, 1, 0, 1], [0, 1, 0, 1], 0.8, 1.0),
            # Bad
            ([0, 1, 0, 1], [1, 0, 1, 0], 0.2, -4.0),
            ([0, 1, 0, 1], [1, 0, 1, 0], 0.8, -4.0),
            ([0, 0, 0, 0], [0, 1, 0, 1], 0.2, -3.0),
            ([1, 1, 1, 1], [0, 1, 0, 1], 0.8, -3.0),
            ([0, 1, 0, 1], [1, 0, 1, 0], 0.5, -1.0),
            ([1, 0, 1, 0], [0, 1, 0, 1], 0.5, -1.0),
            # No skill
            ([0, 0, 0, 0], [0, 1, 0, 1], 0.5, 0),
            ([1, 1, 1, 1], [0, 1, 0, 1], 0.5, 0),
            ([0, 0, 0, 0], [0, 1, 0, 1], 0.8, 0),
            ([1, 1, 1, 1], [0, 1, 0, 1], 0.2, 0),
            # Undefined skill (because obar = 0 or 1)
            ([1, 1, 1, 1], [1, 1, 1, 1], 0.2, np.nan),
            ([0, 0, 0, 0], [0, 0, 0, 0], 0.2, np.nan),
            ([1, 1, 1, 1], [0, 0, 0, 0], 0.2, np.nan),
            ([0, 0, 0, 0], [1, 1, 1, 1], 0.2, np.nan),
            ([0, 1, 0, 1], [1, 1, 1, 1], 0.2, np.nan),
            ([0, 1, 0, 1], [0, 0, 0, 0], 0.2, np.nan),
            # Undefined skill (because cost loss = 0 or 1)
            ([0, 1, 0, 1], [0, 1, 0, 1], 0.0, np.nan),
            ([0, 1, 0, 1], [0, 1, 0, 1], 1.0, np.nan),
            # Single valued entries (undefined because obar = 0 or 1)
            ([1], [0], 0.2, np.nan),
            ([1], [0], 0.2, np.nan),
            ([0], [1], 0.2, np.nan),
            ([1], [1], 0.2, np.nan),
            # Complicated cases
            ([1] * 5 + [0] * 5 + [1] * 10 + [0] * 80, [1] * 5 + [1] * 5 + [0] * 10 + [0] * 80, 0.2, 0.25),
            ([1] * 7 + [0] * 3 + [1] * 8 + [0] * 82, [1] * 7 + [1] * 3 + [0] * 8 + [0] * 82, 0.2, 0.5),
            ([1] * 9 + [0] * 1 + [1] * 6 + [0] * 84, [1] * 9 + [1] * 1 + [0] * 6 + [0] * 84, 0.2, 0.75),
        ],
    )
    def test_rev_combinations(self, fcst, obs, cost_loss, expected):
        """Tests both the core and wrapper for known cases"""

        cost_loss_ratio = [cost_loss]

        fcst_xr = xr.DataArray(fcst, dims=["time"])
        obs_xr = xr.DataArray(obs, dims=["time"])
        expected_xr = xr.DataArray(expected, dims=["cost_loss_ratio"], coords={"cost_loss_ratio": cost_loss_ratio})

        actual_xr = relative_economic_value(fcst_xr, obs_xr, cost_loss_ratio)
        xr.testing.assert_allclose(actual_xr, expected_xr)

        actual_xr = _calculate_rev_core(binary_fcst=fcst_xr, obs=obs_xr, cost_loss_ratios=cost_loss_ratio)

        xr.testing.assert_allclose(expected_xr, actual_xr)

    def test_with_nans(self):
        """Test handling of NaN values in input data."""
        binary_fcst = xr.DataArray([1, 1, 0, 0, 1, np.nan], dims=["time"], coords={"time": range(6)})
        obs = xr.DataArray([1, 0, 1, 0, np.nan, 0], dims=["time"], coords={"time": range(6)})

        cost_loss_ratios = [0.5]

        result = _calculate_rev_core(
            binary_fcst=binary_fcst,
            obs=obs,
            cost_loss_ratios=cost_loss_ratios,
            dims_to_reduce="all",
            weights=None,
        )

        expected = xr.DataArray([0.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})

        xr.testing.assert_identical(result, expected)

    def test_multiple_cost_loss_ratios(self):
        """Test with multiple cost-loss ratios spanning the full range."""
        np.random.seed(42)
        binary_fcst = xr.DataArray(np.random.randint(0, 2, size=100), dims=["sample"])
        obs = xr.DataArray(np.random.randint(0, 2, size=100), dims=["sample"])

        cost_loss_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        result = _calculate_rev_core(
            binary_fcst=binary_fcst,
            obs=obs,
            cost_loss_ratios=cost_loss_ratios,
            dims_to_reduce="all",
            weights=None,
        )

        expected = xr.DataArray(
            [np.nan, -0.57142857, -0.13636364, -1.54545455, np.nan],
            dims=["cost_loss_ratio"],
            coords={"cost_loss_ratio": cost_loss_ratios},
        )

        xr.testing.assert_allclose(result, expected)


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

        data = np.array(
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
            data,
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": np.arange(0.1, 1.0, 0.1), "cost_loss_ratio": [0.2, 0.4, 0.6, 0.8]},
        )

        xr.testing.assert_allclose(expected_full_result, actual_full_result)

        data = np.array(
            [0.0, 0.3, 0.2, 0.0],
        )

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

        data = np.array(
            [[0.0, 0.0, -0.5, -3.0], [-0.2, 0.3, 0.2, -0.8], [-0.2, 0.3, 0.2, -0.8], [-3.0, -0.5, 0.0, 0.0]]
        )

        expected_full_result = xr.DataArray(
            data,
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
    def test_equal_weights_same_as_unweighted(self, weight_value):
        """When all weights are equal, weighted result should match unweighted."""
        fcst_values = [1] * 7 + [0] * 3 + [1] * 8 + [0] * 82
        obs_values = [1] * 7 + [1] * 3 + [0] * 8 + [0] * 82
        weight_values = [weight_value] * 100

        fcst = xr.DataArray(fcst_values, dims=["time"])
        obs = xr.DataArray(obs_values, dims=["time"])
        weights = xr.DataArray(weight_values, dims=["time"])

        rev_weighted = relative_economic_value(fcst, obs, cost_loss_ratios=[0.2], weights=weights)
        rev_unweighted = relative_economic_value(fcst, obs, [0.2])

        xr.testing.assert_allclose(rev_weighted, rev_unweighted)

    def test_weights_nonuniform(self):
        """Test when time weights vary"""
        fcst_values = [1] * 7 + [0] * 3 + [1] * 8 + [0] * 82
        obs_values = [1] * 7 + [1] * 3 + [0] * 8 + [0] * 82
        weight_values = [0.5] * 7 + [0.25] * 3 + [0.5] * 8 + [1] * 82

        fcst = xr.DataArray(fcst_values, dims=["time"])
        obs = xr.DataArray(obs_values, dims=["time"])
        weights = xr.DataArray(weight_values, dims=["time"])

        cost_loss_ratios = [0.2]
        actual = relative_economic_value(fcst, obs, cost_loss_ratios=cost_loss_ratios, weights=weights)

        expected = xr.DataArray(
            [0.58823529],
            dims=["cost_loss_ratio"],
            coords={"cost_loss_ratio": cost_loss_ratios},
        )

        xr.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        "use_weights,expected_rev",
        [
            (False, 0.5),
            (True, 0.455861),
        ],
        ids=["no_weights", "with_cosine_weights"],
    )
    def test_known_rev_values_spatial(self, use_weights, expected_rev):
        """Test known REV values across different latitudes.
        Also tests that weights can have fewer coordinates than fcst/obs and it broadcasts fine"""

        # Three forecast quality scenarios at different latitudes
        # Format: (hits, false_alarms, misses, correct_negatives) -> REV
        # Lat 30: (5, 10, 5, 80) -> REV = 0.25
        # Lat 45: (7, 8, 3, 82) -> REV = 0.5
        # Lat 60: (9, 6, 1, 84) -> REV = 0.75
        fcst_data = np.array(
            [
                [1] * 5 + [0] * 5 + [1] * 10 + [0] * 80,  # lat 30
                [1] * 7 + [0] * 3 + [1] * 8 + [0] * 82,  # lat 45
                [1] * 9 + [0] * 1 + [1] * 6 + [0] * 84,  # lat 60
            ]
        ).T[:, :, np.newaxis]

        obs_data = np.array(
            [
                [1] * 5 + [1] * 5 + [0] * 10 + [0] * 80,  # lat 30
                [1] * 7 + [1] * 3 + [0] * 8 + [0] * 82,  # lat 45
                [1] * 9 + [1] * 1 + [0] * 6 + [0] * 84,  # lat 60
            ]
        ).T[:, :, np.newaxis]

        binary_fcst = xr.DataArray(
            fcst_data, dims=["time", "lat", "lon"], coords={"time": range(100), "lat": [30, 45, 60], "lon": [0]}
        )

        obs = xr.DataArray(
            obs_data, dims=["time", "lat", "lon"], coords={"time": range(100), "lat": [30, 45, 60], "lon": [0]}
        )

        weights = (
            xr.DataArray(np.cos(np.radians([30, 45, 60])), dims=["lat"], coords={"lat": [30, 45, 60]})
            if use_weights
            else None
        )

        result = _calculate_rev_core(
            binary_fcst=binary_fcst,
            obs=obs,
            cost_loss_ratios=[0.2],
            dims_to_reduce="all",
            weights=weights,
        )

        expected = xr.DataArray([expected_rev], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.2]})
        xr.testing.assert_allclose(result, expected)

    def test_known_rev_values_spatial_with_weights_two_lons(self):
        """Test known REV values across different latitudes and longitudes with weights."""

        # Three forecast quality scenarios at different latitudes
        # Format: (hits, false_alarms, misses, correct_negatives) -> REV
        # Lat 30: (5, 10, 5, 80) -> REV = 0.25
        # Lat 45: (7, 8, 3, 82) -> REV = 0.5
        # Lat 60: (9, 6, 1, 84) -> REV = 0.75
        fcst_2d = np.array(
            [
                [1] * 5 + [0] * 5 + [1] * 10 + [0] * 80,  # lat 30
                [1] * 7 + [0] * 3 + [1] * 8 + [0] * 82,  # lat 45
                [1] * 9 + [0] * 1 + [1] * 6 + [0] * 84,  # lat 60
            ]
        ).T

        obs_2d = np.array(
            [
                [1] * 5 + [1] * 5 + [0] * 10 + [0] * 80,  # lat 30
                [1] * 7 + [1] * 3 + [0] * 8 + [0] * 82,  # lat 45
                [1] * 9 + [1] * 1 + [0] * 6 + [0] * 84,  # lat 60
            ]
        ).T

        # Expand to 2 longitudes: lon=0 uses original obs, lon=180 uses inverted obs
        fcst_data = np.stack([fcst_2d, fcst_2d], axis=2)
        obs_data = np.stack([obs_2d, 1 - obs_2d], axis=2)

        binary_fcst = xr.DataArray(
            fcst_data, dims=["time", "lat", "lon"], coords={"time": range(100), "lat": [30, 45, 60], "lon": [0, 180]}
        )

        obs = xr.DataArray(
            obs_data, dims=["time", "lat", "lon"], coords={"time": range(100), "lat": [30, 45, 60], "lon": [0, 180]}
        )

        # Weights vary by latitude only (cosine weighting)
        weights = xr.DataArray(np.cos(np.radians([30, 45, 60])), dims=["lat"], coords={"lat": [30, 45, 60]})

        result = _calculate_rev_core(
            binary_fcst=binary_fcst,
            obs=obs,
            cost_loss_ratios=[0.2],
            dims_to_reduce=["time", "lat"],
            weights=weights,
        )

        expected = xr.DataArray(
            [[0.455861], [-32.323443]],
            dims=["lon", "cost_loss_ratio"],
            coords={"lon": [0, 180], "cost_loss_ratio": [0.2]},
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_non_negative(self):
        """Test that negative weights raise a ValueError"""
        fcst = xr.DataArray([0, 1])
        obs = xr.DataArray([0, 1])

        # Negative weights should raise
        with pytest.raises(ValueError, match="weights must be non-negative"):
            _validate_rev_inputs(
                fcst,
                obs,
                cost_loss_ratios=[0.2, 0.5],
                threshold=[0.3, 0.7],
                threshold_dim="threshold",
                cost_loss_dim="cost_loss",
                weights=xr.DataArray([1, -1]),
                derived_metrics=None,
                threshold_outputs=None,
            )

        # Zero or positive weights should pass
        _validate_rev_inputs(
            fcst,
            obs,
            cost_loss_ratios=[0.2, 0.5],
            threshold=[0.3, 0.7],
            threshold_dim="threshold",
            cost_loss_dim="cost_loss",
            weights=xr.DataArray([0, 1]),
            derived_metrics=None,
            threshold_outputs=None,
        )


class TestDatasetInputs:
    """Test that REV works with xr.Dataset inputs."""

    def test_forecast_as_dataset(self):
        """Test with forecast as Dataset."""
        # hashtag ECMWF_Winning
        fcst_ds = xr.Dataset(
            {"ecmwf": xr.DataArray([0, 1, 1, 0], dims=["time"]), "access": xr.DataArray([1, 0, 1, 1], dims=["time"])}
        )
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])

        actual = relative_economic_value(fcst_ds, obs, [0.5])

        expected = xr.Dataset(
            data_vars={"ecmwf": (["cost_loss_ratio"], [0.0]), "access": (["cost_loss_ratio"], [-0.5])},
            coords={"cost_loss_ratio": [0.5]},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_obs_as_dataset(self):
        """Test with observations as Dataset."""
        fcst = xr.DataArray([0.2, 0.8, 0.6, 0.4], dims=["time"])
        obs_ds = xr.Dataset(
            {
                "station_data": xr.DataArray([0, 1, 1, 0], dims=["time"]),
                "radar_data": xr.DataArray([0, 0, 1, 0], dims=["time"]),
            }
        )

        actual = relative_economic_value(fcst, obs_ds, [0.3, 0.7], threshold=[0.5])
        expected = xr.Dataset(
            data_vars={
                "station_data": (["threshold", "cost_loss_ratio"], [[1.0, 1.0]]),
                "radar_data": (["threshold", "cost_loss_ratio"], [[0.571429, -1.33333]]),
            },
            coords={"threshold": [0.5], "cost_loss_ratio": [0.3, 0.7]},
        )

        xr.testing.assert_allclose(actual, expected)

    def test_both_as_dataset(self):
        """Test with both as Dataset."""
        fcst = xr.Dataset(
            {
                "access": xr.DataArray([0.7, 0.3, 0.6, 0.1], dims=["time"]),
                "ecmwf": xr.DataArray([0.8, 0.2, 0.7, 0.05], dims=["time"]),
            }
        )
        obs_ds = xr.Dataset(
            {
                "station_data": xr.DataArray([0, 1, 1, 0], dims=["time"]),
                "radar_data": xr.DataArray([0, 0, 1, 0], dims=["time"]),
            }
        )

        with pytest.raises(ValueError, match="Both fcst and obs cannot be Datasets."):
            relative_economic_value(fcst, obs_ds, [0.3, 0.7], threshold=[0.5])

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


class TestErrorHandlingGroupA:
    """Tests that check that error handling is done correctly (batch A)."""

    def test_probabilistic_without_threshold(self):
        """Test that probabilistic forecasts require threshold"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "0, 1, or NaN" in str(e)

    def test_invalid_cost_loss_ratios_range(self):
        """Test validation of cost-loss ratios - out of range"""
        fcst = xr.DataArray([0, 1, 1], dims=["time"])
        obs = xr.DataArray([0, 1, 0], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [-0.1, 0.5, 1.2])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "between 0 and 1" in str(e)

    def test_invalid_cost_loss_ratios_monotonic(self):
        """Test validation of cost-loss ratios - not monotonic"""
        fcst = xr.DataArray([0, 1, 1], dims=["time"])
        obs = xr.DataArray([0, 1, 0], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5, 0.3, 0.7])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "monotonically increasing" in str(e)

    def test_invalid_threshold_values_range(self):
        """Test validation of threshold values - out of range"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5], threshold=[-0.1, 0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "between 0 and 1" in str(e)

    def test_invalid_threshold_values_monotonic(self):
        """Test validation of threshold values - not monotonic"""
        fcst = xr.DataArray([0.2, 0.8, 0.6], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5], threshold=[0.7, 0.3])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "monotonically increasing" in str(e)

    def test_invalid_obs_values(self):
        """Test validation of observation values"""
        fcst = xr.DataArray([0, 1, 1], dims=["time"])
        obs = xr.DataArray([0, 1, 2], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "0, 1, or NaN" in str(e)

    def test_invalid_fcst_values(self):
        """Test validation of forecast values"""
        fcst = xr.DataArray([0, 1, 2], dims=["time"])
        obs = xr.DataArray([0, 1, 1], dims=["time"])

        try:
            relative_economic_value(fcst, obs, [0.5])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "0, 1, or NaN" in str(e)

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
        fcst = xr.DataArray([0, 1, 1], dims=["time"])
        obs = xr.DataArray([0, 1, 0], dims=["time"])
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


class TestErrorHandlingGroupB:
    """Tests that check that error handling is done correctly (batch B)."""

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

    def test_rational_user_without_threshold_raises(self):
        """Test derived metrics 'rational_user' without threshold raises ValueError"""
        fcst = xr.DataArray([0, 1, 0], dims=["time"])
        obs = xr.DataArray([1, 0, 1], dims=["time"])

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

    def test_binary_forecast_with_dask(self):
        """Test basic REV calculation with Dask arrays."""

        fcst_np = np.array([1, 1, 0, 0, 1])
        obs_np = np.array([1, 1, 0, 0, 1])

        fcst = xr.DataArray(da.from_array(fcst_np, chunks=2), dims=["time"])
        obs = xr.DataArray(da.from_array(obs_np, chunks=2), dims=["time"])

        result = relative_economic_value(fcst, obs, [0.5], check_args=False)

        # Verify result is still lazy
        assert isinstance(result.data, da.Array)

        # Verify computation produces correct result
        computed = result.compute()
        expected = xr.DataArray([1.0], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.5]})
        xr.testing.assert_allclose(computed, expected)

    def test_probabilistic_forecast_with_dask(self):
        """Test probabilistic REV with Dask arrays."""

        np.random.seed(42)
        fcst_np = np.random.uniform(0, 1, 50)
        obs_np = np.random.binomial(1, 0.3, 50)

        fcst = xr.DataArray(da.from_array(fcst_np, chunks=10), dims=["time"])
        obs = xr.DataArray(da.from_array(obs_np, chunks=10), dims=["time"])

        result = relative_economic_value(fcst, obs, [0.2, 0.5], threshold=[0.3, 0.7], check_args=False)

        assert isinstance(result.data, da.Array)
        computed = result.compute()

        expected = xr.DataArray(
            [[-0.19354839, -0.15789474], [-1.16129032, -0.15789474]],
            dims=["threshold", "cost_loss_ratio"],
            coords={"threshold": [0.3, 0.7], "cost_loss_ratio": [0.2, 0.5]},
        )
        xr.testing.assert_allclose(computed, expected)

    @pytest.mark.parametrize("check_args,should_warn", [(True, True), (False, False)])
    def test_dask_array_warning_behavior(self, check_args, should_warn):
        """Test Dask array warning based on check_args parameter."""
        pytest.importorskip("dask")

        fcst_dask = xr.DataArray(da.from_array([0, 1, 1, 0], chunks=2), dims=["time"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])

        if should_warn:
            with pytest.warns(UserWarning, match="check_args=True will force computation on Dask arrays"):
                relative_economic_value(fcst_dask, obs, [0.5], check_args=check_args)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                relative_economic_value(fcst_dask, obs, [0.5], check_args=check_args)

    def test_no_dask_warning_when_dask_unavailable(self, monkeypatch):
        """Test that no warning is issued when dask is not available."""
        # Temporarily hide dask module

        monkeypatch.setitem(sys.modules, "dask", None)
        monkeypatch.setitem(sys.modules, "dask.array", None)

        fcst = xr.DataArray([0, 1, 1, 0], dims=["time"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["time"])

        # Should not raise any warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = relative_economic_value(fcst, obs, [0.5], check_args=True)

        assert result is not None


class TestOther:
    """Other tests that don't fit elsewhere."""

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

    @pytest.mark.parametrize(
        "fcst_has_nan,obs_has_nan",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
        ids=["fcst_nans", "obs_nans", "both_nans"],
    )
    def test_nan_handling(self, fcst_has_nan, obs_has_nan):
        """Test that NaN values are handled correctly."""
        base_fcst = [1] * 5 + [0] * 3 + [1] * 8 + [0] * 80
        base_obs = [1] * 5 + [1] * 3 + [0] * 8 + [0] * 80

        fcst_values = base_fcst + ([np.nan] * 2 if fcst_has_nan else [1] * 2) + [1] * 2 + [0] * 2
        obs_values = base_obs + ([np.nan] * 2 if obs_has_nan else [1] * 2) + [1] * 2 + [0] * 2

        fcst = xr.DataArray(fcst_values, dims=["time"])
        obs = xr.DataArray(obs_values, dims=["time"])

        actual = relative_economic_value(fcst, obs, cost_loss_ratios=[0.2])

        expected = xr.DataArray([0.5], dims=["cost_loss_ratio"], coords={"cost_loss_ratio": [0.2]})

        xr.testing.assert_allclose(actual, expected)
