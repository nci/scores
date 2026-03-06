"""
Contains unit tests for scores.probability.auc_impl.roc_auc
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.probability import roc_auc


# ──────────────────────────── basic correctness ────────────────────────────


class TestRocAucBasic:
    """Tests for basic ROC AUC correctness."""

    def test_perfect_discrimination(self):
        """AUC should be 1.0 when forecasts perfectly separate events from non-events."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 1.0

    def test_no_discrimination(self):
        """AUC should be 0.5 when forecasts are identical for events and non-events."""
        fcst = xr.DataArray([0.5, 0.5, 0.5, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 0.5

    def test_reversed_discrimination(self):
        """AUC should be 0.0 when forecasts are perfectly reversed."""
        fcst = xr.DataArray([0.1, 0.2, 0.8, 0.9], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 0.0

    def test_known_auc_value(self):
        """Test against a manually computed AUC value."""
        # 2 positives with ranks 3, 4 out of 4 samples
        # R1 = 3 + 4 = 7, U = 7 - 2*3/2 = 4, AUC = 4 / (2*2) = 1.0
        fcst = xr.DataArray([0.1, 0.2, 0.7, 0.9], dims=["sample"])
        obs = xr.DataArray([0, 0, 1, 1], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 1.0

    def test_partial_discrimination(self):
        """Test a case with partial discrimination."""
        # Forecasts: [0.2, 0.6, 0.4, 0.8], Obs: [0, 1, 0, 1]
        # Sorted fcst values: 0.2(neg), 0.4(neg), 0.6(pos), 0.8(pos)
        # Ranks of positives: 3, 4 -> R1 = 7
        # U = 7 - 2*3/2 = 4, AUC = 4/(2*2) = 1.0
        fcst = xr.DataArray([0.2, 0.6, 0.4, 0.8], dims=["sample"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 1.0

    def test_with_ties(self):
        """Test that ties in forecast values are handled correctly via average ranking."""
        # Forecasts: [0.5, 0.5, 0.5, 0.5], Obs: [1, 0, 1, 0]
        # All ranks are averaged to 2.5
        # R1 = 2.5 + 2.5 = 5.0, U = 5.0 - 2*3/2 = 2.0, AUC = 2.0/(2*2) = 0.5
        fcst = xr.DataArray([0.5, 0.5, 0.5, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 0, 1, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 0.5

    def test_single_positive_single_negative(self):
        """AUC with one positive and one negative observation."""
        fcst = xr.DataArray([0.8, 0.2], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert float(result) == 1.0


# ──────────────────────────── NaN handling ────────────────────────────


class TestRocAucNaN:
    """Tests for NaN handling."""

    def test_nan_in_fcst(self):
        """NaN in fcst should be excluded pairwise."""
        fcst = xr.DataArray([0.9, np.nan, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        # After removing NaN: fcst=[0.9, 0.3, 0.1], obs=[1, 0, 0]
        # Ranks: 0.1->1, 0.3->2, 0.9->3. Positive rank sum = 3
        # U = 3 - 1*2/2 = 2, AUC = 2/(1*2) = 1.0
        assert float(result) == 1.0

    def test_nan_in_obs(self):
        """NaN in obs should be excluded pairwise."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, np.nan, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        # After removing NaN: fcst=[0.9, 0.3, 0.1], obs=[1, 0, 0]
        # AUC = 1.0
        assert float(result) == 1.0

    def test_all_nan_obs(self):
        """If all obs are NaN, result should be NaN."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([np.nan, np.nan, np.nan, np.nan], dims=["sample"])
        result = roc_auc(fcst, obs, check_args=False)
        assert np.isnan(float(result))

    def test_all_positive_obs(self):
        """If all obs are 1 (no negatives), result should be NaN."""
        fcst = xr.DataArray([0.9, 0.8, 0.7, 0.6], dims=["sample"])
        obs = xr.DataArray([1, 1, 1, 1], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert np.isnan(float(result))

    def test_all_negative_obs(self):
        """If all obs are 0 (no positives), result should be NaN."""
        fcst = xr.DataArray([0.9, 0.8, 0.7, 0.6], dims=["sample"])
        obs = xr.DataArray([0, 0, 0, 0], dims=["sample"])
        result = roc_auc(fcst, obs)
        assert np.isnan(float(result))


# ──────────────────────────── dimension handling ────────────────────────────


class TestRocAucDimensions:
    """Tests for reduce_dims and preserve_dims."""

    def test_reduce_all_dims(self):
        """When reduce_dims and preserve_dims are both None, all dims are reduced."""
        fcst = xr.DataArray(
            [[0.9, 0.8], [0.3, 0.1]],
            dims=["station", "time"],
            coords={"station": [0, 1], "time": [0, 1]},
        )
        obs = xr.DataArray(
            [[1, 1], [0, 0]],
            dims=["station", "time"],
            coords={"station": [0, 1], "time": [0, 1]},
        )
        result = roc_auc(fcst, obs)
        assert result.dims == ()
        assert float(result) == 1.0

    def test_preserve_dims(self):
        """Preserving a dimension should produce AUC per slice."""
        fcst = xr.DataArray(
            [[0.9, 0.8, 0.3, 0.1], [0.1, 0.3, 0.8, 0.9]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        obs = xr.DataArray(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        result = roc_auc(fcst, obs, preserve_dims=["station"])
        assert result.dims == ("station",)
        assert float(result.sel(station="A")) == 1.0
        assert float(result.sel(station="B")) == 1.0

    def test_reduce_dims(self):
        """Specifying reduce_dims should reduce only those dimensions."""
        fcst = xr.DataArray(
            [[0.9, 0.8, 0.3, 0.1], [0.1, 0.3, 0.8, 0.9]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        obs = xr.DataArray(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        result = roc_auc(fcst, obs, reduce_dims=["time"])
        assert result.dims == ("station",)
        assert float(result.sel(station="A")) == 1.0
        assert float(result.sel(station="B")) == 1.0

    def test_preserve_all_dims(self):
        """preserve_dims='all' should return NaN everywhere (scalar AUC is undefined)."""
        fcst = xr.DataArray([0.9, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        result = roc_auc(fcst, obs, preserve_dims="all")
        assert result.dims == ("sample",)
        # With preserve_dims='all', no dims to reduce — degenerate case
        assert all(np.isnan(result.values))

    def test_multidimensional_preserve(self):
        """Test with 3 dimensions, preserving one."""
        np.random.seed(42)
        n_stations, n_lead, n_time = 2, 3, 50
        fcst_data = np.random.rand(n_stations, n_lead, n_time)
        obs_data = np.random.randint(0, 2, size=(n_stations, n_lead, n_time)).astype(float)
        fcst = xr.DataArray(
            fcst_data,
            dims=["station", "lead", "time"],
        )
        obs = xr.DataArray(
            obs_data,
            dims=["station", "lead", "time"],
        )
        result = roc_auc(fcst, obs, preserve_dims=["station"])
        assert result.dims == ("station",)
        assert result.shape == (n_stations,)
        # Values should be finite (with enough random data both classes are present)
        assert all(np.isfinite(result.values))


# ──────────────────────────── input validation ────────────────────────────


class TestRocAucValidation:
    """Tests for input validation."""

    def test_fcst_out_of_range_high(self):
        """Raises ValueError when fcst > 1."""
        fcst = xr.DataArray([1.1, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        with pytest.raises(ValueError, match="fcst"):
            roc_auc(fcst, obs)

    def test_fcst_out_of_range_low(self):
        """Raises ValueError when fcst < 0."""
        fcst = xr.DataArray([-0.1, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        with pytest.raises(ValueError, match="fcst"):
            roc_auc(fcst, obs)

    def test_obs_not_binary(self):
        """Raises ValueError when obs has values other than 0, 1, NaN."""
        fcst = xr.DataArray([0.9, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 2], dims=["sample"])
        with pytest.raises(ValueError, match="obs"):
            roc_auc(fcst, obs)

    def test_check_args_false_skips_validation(self):
        """Setting check_args=False should skip validation and not raise."""
        fcst = xr.DataArray([1.5, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        # Should not raise
        roc_auc(fcst, obs, check_args=False)

    def test_both_dims_raises(self):
        """Specifying both reduce_dims and preserve_dims should raise ValueError."""
        fcst = xr.DataArray([0.9, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 0], dims=["sample"])
        with pytest.raises(ValueError):
            roc_auc(fcst, obs, reduce_dims=["sample"], preserve_dims=["sample"])


# ──────────────────────────── Dataset support ────────────────────────────


class TestRocAucDataset:
    """Tests for xr.Dataset support."""

    def test_dataset_input(self):
        """roc_auc should work with xr.Dataset inputs."""
        fcst = xr.Dataset(
            {
                "model_a": xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"]),
                "model_b": xr.DataArray([0.1, 0.3, 0.8, 0.9], dims=["sample"]),
            }
        )
        obs = xr.Dataset(
            {
                "model_a": xr.DataArray([1, 1, 0, 0], dims=["sample"]),
                "model_b": xr.DataArray([0, 0, 1, 1], dims=["sample"]),
            }
        )
        result = roc_auc(fcst, obs)
        assert isinstance(result, xr.Dataset)
        np.testing.assert_almost_equal(float(result["model_a"]), 1.0)
        np.testing.assert_almost_equal(float(result["model_b"]), 1.0)


# ──────────────────────────── consistency with ROC curve ────────────────────────────


class TestRocAucConsistency:
    """Tests that roc_auc is consistent with the trapezoidal AUC from the ROC curve."""

    def test_consistent_with_roc_curve_data(self):
        """AUC from roc_auc should match AUC from roc_curve_data."""
        from scores.probability import roc_curve_data

        np.random.seed(123)
        fcst = xr.DataArray(np.random.rand(100), dims=["sample"])
        obs = xr.DataArray(np.random.randint(0, 2, size=100), dims=["sample"])

        auc_mann_whitney = float(roc_auc(fcst, obs))
        roc_result = roc_curve_data(fcst, obs)
        auc_trapezoidal = float(roc_result["AUC"])

        np.testing.assert_almost_equal(auc_mann_whitney, auc_trapezoidal, decimal=10)

    def test_consistent_with_roc_curve_data_with_ties(self):
        """AUC consistency when there are tied forecast values."""
        from scores.probability import roc_curve_data

        fcst = xr.DataArray([0.5, 0.5, 0.5, 0.8, 0.8, 0.2, 0.2, 0.2], dims=["sample"])
        obs = xr.DataArray([1, 0, 1, 1, 0, 0, 1, 0], dims=["sample"])

        auc_mann_whitney = float(roc_auc(fcst, obs))
        roc_result = roc_curve_data(fcst, obs)
        auc_trapezoidal = float(roc_result["AUC"])

        np.testing.assert_almost_equal(auc_mann_whitney, auc_trapezoidal, decimal=10)

    def test_consistent_preserve_dims(self):
        """AUC consistency when preserving dimensions."""
        from scores.probability import roc_curve_data

        np.random.seed(456)
        fcst = xr.DataArray(
            np.random.rand(3, 80),
            dims=["station", "time"],
            coords={"station": ["A", "B", "C"]},
        )
        obs = xr.DataArray(
            np.random.randint(0, 2, size=(3, 80)),
            dims=["station", "time"],
            coords={"station": ["A", "B", "C"]},
        )

        result_mw = roc_auc(fcst, obs, preserve_dims=["station"])
        result_roc = roc_curve_data(fcst, obs, preserve_dims=["station"])

        for station in ["A", "B", "C"]:
            np.testing.assert_almost_equal(
                float(result_mw.sel(station=station)),
                float(result_roc["AUC"].sel(station=station)),
                decimal=10,
            )


# ──────────────────────────── Dask support ────────────────────────────


class TestRocAucDask:
    """Tests for Dask-backed arrays."""

    def test_dask_basic(self):
        """roc_auc should work with Dask-backed arrays."""
        if dask == "Unavailable":  # pragma: no cover
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"]).chunk({"sample": 4})
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"]).chunk({"sample": 4})
        result = roc_auc(fcst, obs, check_args=False)
        assert float(result) == 1.0

    def test_dask_warns_on_check_args(self):
        """A warning should be issued when check_args=True with Dask arrays."""
        if dask == "Unavailable":  # pragma: no cover
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"]).chunk({"sample": 4})
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"]).chunk({"sample": 4})
        with pytest.warns(UserWarning, match="Dask"):
            roc_auc(fcst, obs, check_args=True)

    def test_dask_preserve_dims(self):
        """roc_auc with Dask-backed arrays and preserve_dims."""
        if dask == "Unavailable":  # pragma: no cover
            pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

        fcst = xr.DataArray(
            [[0.9, 0.8, 0.3, 0.1], [0.1, 0.3, 0.8, 0.9]],
            dims=["station", "time"],
        ).chunk({"station": 1, "time": 4})
        obs = xr.DataArray(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            dims=["station", "time"],
        ).chunk({"station": 1, "time": 4})
        result = roc_auc(fcst, obs, preserve_dims=["station"], check_args=False)
        np.testing.assert_array_almost_equal(result.values, [1.0, 1.0])


# ──────────────────────────── weights ────────────────────────────


class TestRocAucWeighted:
    """Tests for the weighted ROC AUC."""

    def test_uniform_weights_equals_unweighted(self):
        """Uniform weights should produce the same result as no weights."""
        fcst = xr.DataArray([0.9, 0.8, 0.6, 0.4, 0.2, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 1, 0, 0, 0], dims=["sample"])
        weights = xr.DataArray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dims=["sample"])
        result_weighted = float(roc_auc(fcst, obs, weights=weights))
        result_unweighted = float(roc_auc(fcst, obs))
        np.testing.assert_almost_equal(result_weighted, result_unweighted, decimal=12)

    def test_perfect_discrimination_with_weights(self):
        """Weighted AUC should be 1.0 for perfect forecasts regardless of weights."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        weights = xr.DataArray([2.0, 1.0, 3.0, 0.5], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        assert result == 1.0

    def test_reversed_discrimination_with_weights(self):
        """Weighted AUC should be 0.0 for perfectly reversed forecasts."""
        fcst = xr.DataArray([0.1, 0.2, 0.8, 0.9], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        weights = xr.DataArray([2.0, 1.0, 3.0, 0.5], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        assert result == 0.0

    def test_weighted_known_value(self):
        """Test a manually computed weighted AUC.

        fcst: [0.2, 0.8, 0.4, 0.9], obs: [0, 1, 0, 1], weights: [1, 2, 1, 1]
        Sorted: (0.2,neg,w=1), (0.4,neg,w=1), (0.8,pos,w=2), (0.9,pos,w=1)
        Sweep:
          i=0 (0.2, neg): cum_neg stays 0 -> cum_neg becomes 1
          i=1 (0.4, neg): cum_neg stays 1 -> cum_neg becomes 2
          i=2 (0.8, pos, w=2): U += 2 * (2 + 0.5*0) = 4
          i=3 (0.9, pos, w=1): U += 1 * (2 + 0.5*0) = 2
        U_w = 6, W_+ = 3, W_- = 2, AUC = 6 / (3*2) = 1.0
        """
        fcst = xr.DataArray([0.2, 0.8, 0.4, 0.9], dims=["sample"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["sample"])
        weights = xr.DataArray([1.0, 2.0, 1.0, 1.0], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        assert result == 1.0

    def test_weights_with_ties(self):
        """Weighted AUC with tied forecast values should handle half-concordance correctly.

        fcst: [0.5, 0.5, 0.5, 0.5], obs: [1, 0, 1, 0], weights: [2, 1, 1, 2]
        All in one tie group.
        tie_neg = 1 + 2 = 3, pos weights = [2, 1], W_+ = 3, W_- = 3
        U_w = (2 + 1) * (0 + 0.5 * 3) = 3 * 1.5 = 4.5
        AUC = 4.5 / (3 * 3) = 0.5
        """
        fcst = xr.DataArray([0.5, 0.5, 0.5, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 0, 1, 0], dims=["sample"])
        weights = xr.DataArray([2.0, 1.0, 1.0, 2.0], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        np.testing.assert_almost_equal(result, 0.5, decimal=12)

    def test_zero_weight_excludes_sample(self):
        """A sample with weight=0 should be effectively excluded.

        With weight=0 on the single negative that causes errors, AUC should be NaN
        because there are no effective negatives.
        """
        fcst = xr.DataArray([0.9, 0.1, 0.5], dims=["sample"])
        obs = xr.DataArray([1, 1, 0], dims=["sample"])
        # Zero-weight the only negative -> no effective negatives -> NaN
        weights = xr.DataArray([1.0, 1.0, 0.0], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        assert np.isnan(result)

    def test_weighted_partially_concordant(self):
        """Weighted AUC for a partially concordant case.

        fcst: [0.3, 0.7, 0.6, 0.4], obs: [0, 1, 0, 1], weights: [1, 1, 1, 1]
        Sorted: (0.3,neg,w=1), (0.4,pos,w=1), (0.6,neg,w=1), (0.7,pos,w=1)
        Sweep:
          (0.3,neg): cum_neg -> 1
          (0.4,pos,w=1): U += 1*(1 + 0) = 1, cum_neg stays 1
          (0.6,neg): cum_neg -> 2
          (0.7,pos,w=1): U += 1*(2 + 0) = 2
        U_w = 3, W_+ = 2, W_- = 2, AUC = 3/4 = 0.75
        Should match unweighted (uniform weights).
        """
        fcst = xr.DataArray([0.3, 0.7, 0.6, 0.4], dims=["sample"])
        obs = xr.DataArray([0, 1, 0, 1], dims=["sample"])
        weights = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["sample"])
        result = float(roc_auc(fcst, obs, weights=weights))
        np.testing.assert_almost_equal(result, 0.75, decimal=12)

    def test_weights_broadcast_from_lat_dim(self):
        """Latitude-style weights broadcast correctly over a time dimension."""
        np.random.seed(7)
        fcst = xr.DataArray(
            np.random.rand(3, 20),
            dims=["lat", "time"],
            coords={"lat": [20.0, 45.0, 70.0]},
        )
        obs = xr.DataArray(
            np.random.randint(0, 2, size=(3, 20)).astype(float),
            dims=["lat", "time"],
            coords={"lat": [20.0, 45.0, 70.0]},
        )
        lat_weights = xr.DataArray(
            np.cos(np.deg2rad([20.0, 45.0, 70.0])),
            dims=["lat"],
            coords={"lat": [20.0, 45.0, 70.0]},
        )
        result = roc_auc(fcst, obs, weights=lat_weights)
        assert result.dims == ()
        assert np.isfinite(float(result))

    def test_weights_preserve_dims(self):
        """Weights should work correctly when preserving a dimension."""
        fcst = xr.DataArray(
            [[0.9, 0.8, 0.3, 0.1], [0.1, 0.3, 0.8, 0.9]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        obs = xr.DataArray(
            [[1, 1, 0, 0], [0, 0, 1, 1]],
            dims=["station", "time"],
            coords={"station": ["A", "B"], "time": [0, 1, 2, 3]},
        )
        weights = xr.DataArray([1.0, 2.0, 1.0, 2.0], dims=["time"], coords={"time": [0, 1, 2, 3]})
        result = roc_auc(fcst, obs, weights=weights, preserve_dims=["station"])
        assert result.dims == ("station",)
        assert float(result.sel(station="A")) == 1.0
        assert float(result.sel(station="B")) == 1.0

    def test_invalid_weights_negative_raises(self):
        """Negative weights should raise a ValueError."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        weights = xr.DataArray([1.0, -1.0, 1.0, 1.0], dims=["sample"])
        with pytest.raises(ValueError):
            roc_auc(fcst, obs, weights=weights)

    def test_invalid_weights_nan_raises(self):
        """NaN weights should raise a ValueError."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        weights = xr.DataArray([1.0, np.nan, 1.0, 1.0], dims=["sample"])
        with pytest.raises(ValueError):
            roc_auc(fcst, obs, weights=weights)

    def test_check_args_false_skips_weight_validation(self):
        """check_args=False should NOT skip weight validation (weights are always checked)."""
        fcst = xr.DataArray([0.9, 0.8, 0.3, 0.1], dims=["sample"])
        obs = xr.DataArray([1, 1, 0, 0], dims=["sample"])
        weights = xr.DataArray([1.0, -1.0, 1.0, 1.0], dims=["sample"])
        # Weight validation is independent of check_args
        with pytest.raises(ValueError):
            roc_auc(fcst, obs, weights=weights, check_args=False)
