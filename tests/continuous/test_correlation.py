"""
Tests for correlation calculations
"""

import numpy as np
import pytest
import xarray as xr

from scores.continuous.correlation import anomaly_correlation_coefficient, pearsonr, spearmanr

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

DA1_CORR = xr.DataArray(
    np.array([[1, 2, 3], [0, 1, 0], [0.5, -0.5, 0.5], [3, 6, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA2_CORR = xr.DataArray(
    np.array([[2, 4, 6], [6, 5, 6], [3, 4, 5], [3, np.nan, 3]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)

DA3_CORR = xr.DataArray(
    np.array([[1, 2, 3], [3, 2.5, 3], [1.5, 2, 2.5], [1.5, np.nan, 1.5]]),
    dims=("space", "time"),
    coords=[
        ("space", ["w", "x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA4_CORR = xr.DataArray(
    np.array([[1, 3, 7], [2, 2, 8], [3, 1, 7]]),
    dims=("space", "time"),
    coords=[
        ("space", ["x", "y", "z"]),
        ("time", [1, 2, 3]),
    ],
)
DA5_CORR = xr.DataArray(
    np.array([1, 2, 3]),
    dims=("space"),
    coords=[("space", ["x", "y", "z"])],
)

EXP_CORR_KEEP_SPACE_DIM = xr.DataArray(
    np.array([1.0, -1.0, 0.0, np.nan]),
    dims=("space"),
    coords=[("space", ["w", "x", "y", "z"])],
)

EXP_CORR_REDUCE_ALL = xr.DataArray(1.0)

EXP_CORR_DIFF_SIZE = xr.DataArray(
    np.array([1.0, -1.0, 0.0]),
    dims=("time"),
    coords=[("time", [1, 2, 3])],
)

# Adding testing for divergence between Pearson and Spearman

# Generate non-linear monotonic data using a logistic function
np.random.seed(42)
X = np.linspace(0, 10, 100)
Y = 1 / (1 + np.exp(-X))  # Logistic relationship

# Convert to xarray.DataArray
X_DA = xr.DataArray(X, dims="sample", name="x")
Y_DA = xr.DataArray(Y, dims="sample", name="y")
PEARSON_OUTPUT = 0.76
SPEARMAN_OUTPUT = 1.0


@pytest.mark.parametrize(
    ("da1", "da2", "reduce_dims", "preserve_dims", "expected"),
    [
        # Check reduce dim arg
        (DA1_CORR, DA2_CORR, None, "space", EXP_CORR_KEEP_SPACE_DIM),
        # Check preserve dim arg
        (DA1_CORR, DA2_CORR, "time", None, EXP_CORR_KEEP_SPACE_DIM),
        # Check reduce all
        (DA3_CORR, DA2_CORR, None, None, EXP_CORR_REDUCE_ALL),
        # Check different size arrays as input
        (DA4_CORR, DA5_CORR, "space", None, EXP_CORR_DIFF_SIZE),
        # Check Dataset
        (
            xr.Dataset({"a": DA1_CORR, "b": DA2_CORR}),
            xr.Dataset({"a": DA2_CORR, "b": DA1_CORR}),
            None,
            "space",
            xr.Dataset({"a": EXP_CORR_KEEP_SPACE_DIM, "b": EXP_CORR_KEEP_SPACE_DIM}),
        ),
    ],
)
def test_pearson_correlation(da1, da2, reduce_dims, preserve_dims, expected):
    """
    Tests continuous.correlation.pearsonr
    """
    result = pearsonr(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("da1", "da2", "preserve_dims", "err", "err_msg"),
    [
        # Check preserve_dims = "all"
        (DA1_CORR, DA2_CORR, "all", ValueError, "You cannot preserve all dimensions with"),
        # Check preserve_dims = all the dims
        (DA1_CORR, DA2_CORR, ["time", "space"], ValueError, "You cannot preserve all dimensions with"),
        # Check xr.Datasets with different variables
        (
            xr.Dataset({"var1": DA1_CORR}),
            xr.Dataset({"var2": DA2_CORR}),
            None,
            ValueError,
            "Both datasets must contain the same variables",
        ),
        # Check mixing Datasets with DataArrays
        (
            xr.Dataset({"var1": DA1_CORR}),
            DA2_CORR,
            None,
            TypeError,
            "Both fcst and obs must be either xarray DataArrays or xarray Datasets",
        ),
    ],
)
def test_pearson_correlation_raises(da1, da2, preserve_dims, err, err_msg):
    """
    Tests continuous.correlation.pearsonr raises the correct errors
    """
    with pytest.raises(err, match=err_msg):
        pearsonr(da1, da2, preserve_dims=preserve_dims)


def test_correlation_dask():
    """
    Tests continuous.correlation works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = pearsonr(DA3_CORR.chunk(), DA2_CORR.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, EXP_CORR_REDUCE_ALL)


@pytest.mark.parametrize(
    ("da1", "da2", "reduce_dims", "preserve_dims", "expected"),
    [
        # Check reduce dim arg
        (DA1_CORR, DA2_CORR, None, "space", EXP_CORR_KEEP_SPACE_DIM),
        # Check preserve dim arg
        (DA1_CORR, DA2_CORR, "time", None, EXP_CORR_KEEP_SPACE_DIM),
        # Check reduce all
        (DA3_CORR, DA2_CORR, None, None, EXP_CORR_REDUCE_ALL),
        # Check different size arrays as input
        (DA4_CORR, DA5_CORR, "space", None, EXP_CORR_DIFF_SIZE),
    ],
)
def test_spearman_correlation(da1, da2, reduce_dims, preserve_dims, expected):
    """
    Tests continuous.correlation.spearmanr
    """
    result = spearmanr(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("da1", "da2", "preserve_dims", "err", "err_msg"),
    [
        # Check preserve_dims = "all"
        (DA1_CORR, DA2_CORR, "all", ValueError, "You cannot preserve all dimensions with"),
        # Check preserve_dims = all the dims
        (DA1_CORR, DA2_CORR, ["time", "space"], ValueError, "You cannot preserve all dimensions with"),
        # Check xr.Datasets with different variables
        (
            xr.Dataset({"var1": DA1_CORR}),
            xr.Dataset({"var2": DA2_CORR}),
            None,
            ValueError,
            "Both datasets must contain the same variables",
        ),
        # Check mixing Datasets with DataArrays
        (
            xr.Dataset({"var1": DA1_CORR}),
            DA2_CORR,
            None,
            TypeError,
            "Both fcst and obs must be either xarray DataArrays or xarray Datasets",
        ),
    ],
)
def test_spearman_correlation_raises(da1, da2, preserve_dims, err, err_msg):
    """
    Tests continuous.correlation.spearmanr raises the correct errors
    """
    with pytest.raises(err, match=err_msg):
        spearmanr(da1, da2, preserve_dims=preserve_dims)


def test_spearman_correlation_dask():
    """
    Tests continuous.correlation.spearmanr works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = spearmanr(DA3_CORR.chunk(), DA2_CORR.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, (np.ndarray, np.generic))
    xr.testing.assert_allclose(result, EXP_CORR_REDUCE_ALL)


@pytest.mark.parametrize(
    ("da1", "da2", "reduce_dims", "preserve_dims", "expected", "corr"),
    [
        # Check non-linear monotonic relationship
        (X_DA, Y_DA, None, None, PEARSON_OUTPUT, "pearson"),
        (X_DA, Y_DA, None, None, SPEARMAN_OUTPUT, "spearman"),
    ],
)
def test_divergence(da1, da2, reduce_dims, preserve_dims, expected, corr):
    if corr == "spearman":
        result = spearmanr(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        assert result.item() == pytest.approx(expected)
    else:
        result = pearsonr(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
        assert np.round(result.item(), 2) == expected


@pytest.mark.parametrize(
    ("fcst_ds", "obs_ds", "reduce_dims", "preserve_dims"),
    [
        (
            xr.Dataset({"var1": DA1_CORR, "var2": DA3_CORR}),
            xr.Dataset({"var1": DA2_CORR, "var2": DA2_CORR}),
            "time",
            None,
        ),
        (
            xr.Dataset({"var1": DA1_CORR}),
            xr.Dataset({"var1": DA2_CORR}),
            None,
            "space",
        ),
    ],
)
def test_spearman_correlation_dataset(fcst_ds, obs_ds, reduce_dims, preserve_dims):
    """
    Tests continuous.correlation.spearmanr with xarray.Dataset inputs.
    """
    result = spearmanr(fcst_ds, obs_ds, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == set(fcst_ds.data_vars)


ACC_FCST_ANOMALIES = xr.DataArray(
    [[1.0, 2.0, 3.0, 4.0, np.nan], [2.0, 0.0, -2.0, -4.0, 1]],
    dims=("station", "time"),
    coords={"station": ["a", "b"], "time": [0, 1, 2, 3, 4]},
)
ACC_OBS_ANOMALIES = xr.DataArray(
    [[2.0, 1.0, 4.0, 3.0, 1.0], [1.0, -1.0, -3.0, -5.0, np.nan]],
    dims=("station", "time"),
    coords=ACC_FCST_ANOMALIES.coords,
)
ACC_CLIMATOLOGY = xr.DataArray(
    [[10.0, 20.0, 30.0, 40.0, 50.0], [5.0, 10.0, 15.0, 20.0, 25.0]],
    dims=("station", "time"),
    coords=ACC_FCST_ANOMALIES.coords,
)
ACC_FCST = ACC_CLIMATOLOGY + ACC_FCST_ANOMALIES
ACC_OBS = ACC_CLIMATOLOGY + ACC_OBS_ANOMALIES
ACC_WEIGHTS = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": [0, 1, 2, 3]})

EXP_ACC_BY_STATION = xr.DataArray(
    [28 / np.sqrt(30 * 30), 28 / np.sqrt(24 * 36)],
    dims="station",
    coords={"station": ["a", "b"]},
)
EXP_WEIGHTED_ACC_BY_STATION = xr.DataArray(
    [90 / np.sqrt(100 * 90), 100 / np.sqrt(80 * 130)],
    dims="station",
    coords={"station": ["a", "b"]},
)


@pytest.mark.parametrize(
    ("reduce_dims", "preserve_dims", "weights", "expected"),
    [
        (None, None, None, xr.DataArray(56 / np.sqrt(54 * 66))),
        ("time", None, None, EXP_ACC_BY_STATION),
        (None, "station", None, EXP_ACC_BY_STATION),
        ("time", None, ACC_WEIGHTS, EXP_WEIGHTED_ACC_BY_STATION),
    ],
)
def test_anomaly_correlation_coefficient(reduce_dims, preserve_dims, weights, expected):
    """Check the formula, dimension handling and broadcasting of weights."""
    result = anomaly_correlation_coefficient(
        ACC_FCST,
        ACC_OBS,
        ACC_CLIMATOLOGY,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(result, expected)


def test_anomaly_correlation_coefficient_broadcasting_and_alignment():
    """Broadcast climatology and observations, and use only shared coordinates."""
    climatology = ACC_CLIMATOLOGY.isel(station=0, drop=True)
    obs = (climatology + ACC_OBS_ANOMALIES.isel(station=0, drop=True)).sel(time=[2, 1, 0])
    result = anomaly_correlation_coefficient(
        climatology + ACC_FCST_ANOMALIES,
        obs,
        climatology,
        reduce_dims="time",
    )
    expected = xr.DataArray(
        [16 / np.sqrt(14 * 21), -4 / np.sqrt(8 * 21)],
        dims="station",
        coords={"station": ["a", "b"]},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("centered", "expected_by_reference", "expected_all"),
    [(False, [5 / 7, 1 / 5], 11 / 19), (True, [-1.0, -1.0], -5 / 11)],
)
def test_anomaly_correlation_coefficient_extra_climatology_dimension(centered, expected_by_reference, expected_all):
    """A dimension introduced by climatology can be preserved or reduced by default."""
    fcst = xr.DataArray([1.0, 2.0, 3.0], dims="time")
    obs = xr.DataArray([3.0, 2.0, 1.0], dims="time")
    climatology = xr.DataArray([0.0, 1.0], dims="reference", coords={"reference": ["baseline", "shifted"]})

    result = anomaly_correlation_coefficient(fcst, obs, climatology, centered=centered, preserve_dims="reference")
    expected = xr.DataArray(expected_by_reference, dims="reference", coords=climatology.coords)
    xr.testing.assert_allclose(result, expected)

    result = anomaly_correlation_coefficient(fcst, obs, climatology, centered=centered)
    xr.testing.assert_allclose(result, xr.DataArray(expected_all))


@pytest.mark.parametrize(
    ("fcst", "obs", "climatology", "weights", "expected"),
    [
        # Missing forecast, observation or climatology excludes the point from every weighted sum.
        (
            [11, np.nan, 13, 14, 12],
            [12, 12, np.nan, 14, 11],
            [10, 10, 10, np.nan, 10],
            [1, 2, 3, 4, 2],
            6 / np.sqrt(9 * 6),
        ),
        # Zero forecast anomaly magnitude makes ACC undefined, even with valid observations.
        ([0, 0], [1, 2], [0, 0], [1, 1], np.nan),
        # Zero observed anomaly magnitude makes ACC undefined, even with valid forecasts.
        ([1, 2], [0, 0], [0, 0], [1, 1], np.nan),
        # All forecasts are missing, leaving no valid pairs despite positive weights.
        ([np.nan, np.nan], [1, 2], [0, 0], [1, 1], np.nan),
        # The only valid pair has zero weight; the positive weight belongs to a missing pair.
        ([1, np.nan], [1, 2], [0, 0], [0, 1], np.nan),
    ],
)
def test_anomaly_correlation_coefficient_missing_and_zero(fcst, obs, climatology, weights, expected):
    """Use the same valid samples in all sums; undefined correlations are NaN."""
    fcst, obs, climatology, weights = [
        xr.DataArray(values, dims="time") for values in (fcst, obs, climatology, weights)
    ]
    result = anomaly_correlation_coefficient(fcst, obs, climatology, weights=weights)
    xr.testing.assert_allclose(result, xr.DataArray(expected))


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (None, xr.Dataset({"a": EXP_ACC_BY_STATION, "b": EXP_ACC_BY_STATION})),
        (ACC_WEIGHTS, xr.Dataset({"a": EXP_WEIGHTED_ACC_BY_STATION, "b": EXP_WEIGHTED_ACC_BY_STATION})),
        (
            xr.Dataset({"a": ACC_WEIGHTS, "b": xr.ones_like(ACC_WEIGHTS)}),
            xr.Dataset({"a": EXP_WEIGHTED_ACC_BY_STATION, "b": EXP_ACC_BY_STATION}),
        ),
    ],
)
def test_anomaly_correlation_coefficient_dataset(weights, expected):
    """Dataset variables are scored independently with optional shared or per-variable weights."""
    fcst = xr.Dataset({"a": ACC_FCST, "b": ACC_CLIMATOLOGY + 2 * ACC_FCST_ANOMALIES})
    obs = xr.Dataset({"a": ACC_OBS, "b": ACC_OBS})
    climatology = xr.Dataset({"a": ACC_CLIMATOLOGY, "b": ACC_CLIMATOLOGY})
    result = anomaly_correlation_coefficient(fcst, obs, climatology, reduce_dims="time", weights=weights)
    xr.testing.assert_allclose(result, expected)


def test_anomaly_correlation_coefficient_dataset_requires_reduction_per_variable():
    """Every Dataset variable must have at least one reduced dimension."""
    climatology = xr.Dataset({"a": ("x", [0, 0]), "b": ("y", [0, 0])})
    with pytest.raises(ValueError, match="At least one dimension must be reduced for every data variable"):
        anomaly_correlation_coefficient(climatology + 1, climatology + 1, climatology, preserve_dims="x")


@pytest.mark.parametrize(
    "weights",
    [-ACC_WEIGHTS, xr.zeros_like(ACC_WEIGHTS), xr.full_like(ACC_WEIGHTS, np.nan)],
    ids=["negative", "all_zero", "all_nan"],
)
def test_anomaly_correlation_coefficient_invalid_weights(weights):
    """ACC calls the shared weight validation."""
    with pytest.raises(ValueError):
        anomaly_correlation_coefficient(ACC_FCST, ACC_OBS, ACC_CLIMATOLOGY, weights=weights)


@pytest.mark.parametrize(
    ("fcst", "obs", "climatology", "weights"),
    [
        (ACC_FCST.values, ACC_OBS, ACC_CLIMATOLOGY, None),
        (xr.Dataset({"a": ACC_FCST}), ACC_OBS, ACC_CLIMATOLOGY, None),
        (ACC_FCST, ACC_OBS, ACC_CLIMATOLOGY, ACC_WEIGHTS.values),
        (ACC_FCST, ACC_OBS, ACC_CLIMATOLOGY, xr.Dataset({"a": ACC_WEIGHTS})),
    ],
)
def test_anomaly_correlation_coefficient_input_types_raise(fcst, obs, climatology, weights):
    """Reject incompatible input types."""
    with pytest.raises(TypeError):
        anomaly_correlation_coefficient(fcst, obs, climatology, weights=weights)


@pytest.mark.parametrize("mismatched_input", ["obs", "climatology", "weights"])
def test_anomaly_correlation_coefficient_dataset_variables_raise(mismatched_input):
    """All Dataset inputs must contain the same data variables."""
    inputs = {
        "fcst": xr.Dataset({"a": ACC_FCST}),
        "obs": xr.Dataset({"a": ACC_OBS}),
        "climatology": xr.Dataset({"a": ACC_CLIMATOLOGY}),
        "weights": xr.Dataset({"a": ACC_WEIGHTS}),
    }
    inputs[mismatched_input] = inputs[mismatched_input].rename({"a": "b"})
    with pytest.raises(ValueError, match="same variables"):
        anomaly_correlation_coefficient(**inputs)


@pytest.mark.parametrize("dimension_args", [{"preserve_dims": "all"}, {"reduce_dims": []}])
def test_anomaly_correlation_coefficient_no_reduction(dimension_args):
    """ACC requires a reduction dimension."""
    with pytest.raises(ValueError, match="You cannot preserve all dimensions"):
        anomaly_correlation_coefficient(ACC_FCST, ACC_OBS, ACC_CLIMATOLOGY, **dimension_args)


@pytest.mark.parametrize("centered", [False, True])
def test_anomaly_correlation_coefficient_dask(centered):
    """The weighted result stays lazy, including groups with undefined ACC."""
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst = ACC_FCST.where(ACC_FCST.station == "a", ACC_CLIMATOLOGY)
    result = anomaly_correlation_coefficient(
        fcst.chunk({"time": 2}),
        ACC_OBS.chunk({"time": 2}),
        ACC_CLIMATOLOGY.chunk({"time": 2}),
        reduce_dims="time",
        weights=ACC_WEIGHTS.chunk({"time": 2}),
        centered=centered,
    )
    assert isinstance(result.data, dask.array.Array)
    expected = EXP_WEIGHTED_ACC_BY_STATION.where(ACC_FCST.station == "a")
    if centered:
        expected = xr.DataArray([6 / np.sqrt(116), np.nan], dims="station", coords=expected.coords)
    xr.testing.assert_allclose(result.compute(), expected)


@pytest.mark.parametrize(
    ("reduce_dims", "preserve_dims", "weights", "expected"),
    [
        ("time", None, None, [0.6, 1.0]),
        (None, "station", None, [0.6, 1.0]),
        ("time", None, ACC_WEIGHTS, [6 / np.sqrt(116), 1.0]),
    ],
)
def test_anomaly_correlation_coefficient_centered(reduce_dims, preserve_dims, weights, expected):
    """Centre anomalies independently within each preserved group, using weighted means."""
    result = anomaly_correlation_coefficient(
        ACC_FCST,
        ACC_OBS,
        ACC_CLIMATOLOGY,
        centered=True,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        weights=weights,
    )
    xr.testing.assert_allclose(result, xr.DataArray(expected, dims="station", coords={"station": ["a", "b"]}))


@pytest.mark.parametrize(
    ("fcst", "obs", "climatology", "weights", "expected"),
    [
        # Centring gives perfect negative correlation for oppositely varying anomalies.
        ([1, 2, 3], [3, 2, 1], [0, 0, 0], [1, 1, 1], -1),
        # Zero covariance with nonzero anomaly variances gives zero ACC, not NaN.
        ([1, 0, -1], [1, -2, 1], [0, 0, 0], [1, 1, 1], 0),
        # Constant forecast anomalies have zero variance after centring, so ACC is undefined.
        ([1, 1, 1], [1, 2, 3], [0, 0, 0], [1, 1, 1], np.nan),
        # Constant observed anomalies have zero variance after centring, so ACC is undefined.
        ([1, 2, 3], [1, 1, 1], [0, 0, 0], [1, 1, 1], np.nan),
        # All forecasts are missing, leaving no valid pairs for the centred means.
        ([np.nan, np.nan], [1, 2], [0, 0], [1, 1], np.nan),
        # The only valid pair has zero weight, leaving the centred means undefined.
        ([1, np.nan], [1, 2], [0, 0], [0, 1], np.nan),
        # One positively weighted pair defines the means but leaves both centred variances zero.
        ([1, 2], [3, 4], [0, 0], [1, 0], np.nan),
        # Missing inputs and zero weights exclude points from both means and sums;
        # only the first, second and last pairs contribute.
        (
            [11, 12, np.nan, 100, 100, 100, 13],
            [12, 11, 100, np.nan, 100, 100, 14],
            [10, 10, 10, 10, np.nan, 10, 10],
            [1, 2, 3, 4, 5, 0, 3],
            14 / np.sqrt(340),
        ),
    ],
)
def test_anomaly_correlation_coefficient_centered_missing_and_zero(fcst, obs, climatology, weights, expected):
    """Missing triples and zero weights are excluded from the means as well as the correlation."""
    fcst, obs, climatology, weights = [
        xr.DataArray(values, dims="time") for values in (fcst, obs, climatology, weights)
    ]
    result = anomaly_correlation_coefficient(fcst, obs, climatology, centered=True, weights=weights)
    xr.testing.assert_allclose(result, xr.DataArray(expected))


@pytest.mark.parametrize("as_dataset", [False, True])
def test_anomaly_correlation_coefficient_centered_alignment(as_dataset):
    """Means use shared coordinates, including weights, with broadcast observations and climatology."""
    fcst = ACC_FCST
    obs = ACC_OBS.isel(station=0, drop=True).sel(time=[2, 1, 0])
    climatology = ACC_CLIMATOLOGY.isel(station=0, drop=True)
    weights = ACC_WEIGHTS.sel(time=[3, 2, 0])
    expected = xr.DataArray([1.0, -1.0], dims="station", coords={"station": ["a", "b"]})
    if as_dataset:
        fcst, obs, climatology = [xr.Dataset({"a": value}) for value in (fcst, obs, climatology)]
        expected = xr.Dataset({"a": expected})
    result = anomaly_correlation_coefficient(fcst, obs, climatology, weights=weights, centered=True, reduce_dims="time")
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("weights", [None, ACC_WEIGHTS])
def test_anomaly_correlation_coefficient_centered_matches_pearson(weights):
    """Centred ACC is Pearson correlation of anomalies, invariant to uniform anomaly offsets."""
    fcst = ACC_FCST.where(ACC_FCST.time != 1)
    expected = xr.corr(fcst - ACC_CLIMATOLOGY, ACC_OBS_ANOMALIES, dim=["station", "time"], weights=weights)
    result = anomaly_correlation_coefficient(fcst + 8, ACC_OBS - 3, ACC_CLIMATOLOGY, centered=True, weights=weights)
    xr.testing.assert_allclose(result, expected)
