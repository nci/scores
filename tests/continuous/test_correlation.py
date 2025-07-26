"""
Tests for correlation calculations
"""

import numpy as np
import pytest
import xarray as xr

from scores.continuous.correlation import pearsonr, spearmanr

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore # pylint: disable=invalid-name  # pragma: no cover

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
        assert result.item() == expected
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
