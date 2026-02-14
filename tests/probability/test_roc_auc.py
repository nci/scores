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
