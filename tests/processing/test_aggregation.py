"""
Contains tests for weighting scores appropriately.
"""

import numpy as np
import pytest
import xarray as xr

from scores.processing import agg

DA_3x3 = xr.DataArray(
    [[0, 0.5, 1], [2, np.nan, 1], [97, 1, 1]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)

EXP_DA_X = xr.DataArray(
    [33, 0.75, 1],
    dims=["y"],
    coords={"y": [0, 1, 2]},
)
EXP_DA_X_SUM = xr.DataArray(
    [99, 1.5, 3],
    dims=["y"],
    coords={"y": [0, 1, 2]},
)
EXP_DA_Y = xr.DataArray(
    [0.5, 1.5, 33],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_Y_SUM = xr.DataArray(
    [1.5, 3, 99],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_NAN = xr.DataArray(
    [np.nan, np.nan, np.nan],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_0 = xr.DataArray(
    [0, 0, 0],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_VARY = xr.DataArray(
    [2 / 3, (2 * 3 + 1) / 4, np.nan],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_VARY_SUM = xr.DataArray(
    [4, 7, 0],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_NEW_DIM = xr.DataArray(
    [[33.0, 33.0], [0.75, 0.75], [1.0, 1.0]],
    dims=["y", "z"],
    coords={"z": [0, 1], "y": [0, 1, 2]},
)
EXP_DA_Y_BROADCAST = xr.DataArray(
    [1 / 3, 2, 33],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_VARY_XY = xr.DataArray(1.1)
WEIGHTS1 = xr.DataArray(
    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)
WEIGHTS2 = WEIGHTS1 * 0
WEIGHTS3 = xr.DataArray(
    [[1, 2, 3], [3, 2, 1], [0, 0, 0]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)
WEIGHTS4 = xr.DataArray(
    [1, 2],
    dims=["z"],
    coords={"z": [0, 1]},
)
WEIGHTS5 = xr.DataArray(
    [1, 2, 0],
    dims=["y"],
    coords={"y": [0, 1, 3]},
)

NEGATIVE_WEIGHTS = xr.DataArray(
    [[1, -1, 1], [1, 1, 1], [1, 1, 1]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)

NAN_WEIGHTS = xr.DataArray(
    [[2, 2, 2], [2, 2, 2], [2, np.nan, 2]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)


@pytest.mark.parametrize(
    ("values", "reduce_dims", "weights", "expected"),
    [
        # No reduction
        (DA_3x3, None, None, DA_3x3),
        # Unweighted mean over x
        (DA_3x3, ["x"], None, EXP_DA_X),
        # Unweighted mean over y
        (DA_3x3, ["y"], None, EXP_DA_Y),
        # Equal weights over y
        (DA_3x3, ["y"], WEIGHTS1, EXP_DA_Y),
        # Zero weights over y
        (DA_3x3, ["y"], WEIGHTS2, EXP_DA_NAN),
        # Varying weights over y
        (DA_3x3, ["y"], WEIGHTS3, EXP_DA_VARY),
        # Extra dim in weights
        (DA_3x3, ["x"], WEIGHTS4, EXP_DA_NEW_DIM),
        # Weights only for 1 dim to test broadcasting, with NaN
        (DA_3x3, ["y"], WEIGHTS5, EXP_DA_Y_BROADCAST),
        # Varying weights over x and y
        (DA_3x3, ["x", "y"], WEIGHTS3, EXP_DA_VARY_XY),
        # Varying weights over y, but input error is NaN
        (DA_3x3 * np.nan, ["y"], WEIGHTS3, EXP_DA_VARY * np.nan),
    ],
)
def test_agg_mean(values, reduce_dims, weights, expected):
    """
    Tests scores.functions.agg with method=mean
    """
    # Check with xr.DataArray
    result = agg(values, reduce_dims=reduce_dims, weights=weights)
    xr.testing.assert_equal(result, expected)

    # Check with xr.Dataset
    values_ds = xr.Dataset(({"var1": values, "var2": values}))
    result_ds = agg(values_ds, reduce_dims=reduce_dims, weights=weights)
    expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
    xr.testing.assert_equal(result_ds, expected_ds)


@pytest.mark.parametrize(
    ("values", "reduce_dims", "weights", "expected"),
    [
        # No sum
        (DA_3x3, None, None, DA_3x3),
        # # Unweighted sum over x
        (DA_3x3, ["x"], None, EXP_DA_X_SUM),
        # Unweighted sum over y
        (DA_3x3, ["y"], None, EXP_DA_Y_SUM),
        # Equal weights over y
        # (DA_3x3, ["y"], WEIGHTS1, EXP_DA_Y_SUM),
        # Zero weights over y
        (DA_3x3, ["y"], WEIGHTS2, EXP_DA_0),
        # Varying weights over y
        (DA_3x3, ["y"], WEIGHTS3, EXP_DA_VARY_SUM),
        # # Extra dim in weights
        # (DA_3x3, ["x"], WEIGHTS4, EXP_DA_NEW_DIM),
        # # Weights only for 1 dim to test broadcasting, with NaN
        # (DA_3x3, ["y"], WEIGHTS5, EXP_DA_Y_BROADCAST),
        # # Varying weights over x and y
        # (DA_3x3, ["x", "y"], WEIGHTS3, EXP_DA_VARY_XY),
        # # Varying weights over y, but input error is NaN
        # (DA_3x3 * np.nan, ["y"], WEIGHTS3, EXP_DA_VARY * np.nan),
    ],
)
def test_agg_sum(values, reduce_dims, weights, expected):
    """
    Tests scores.functions.weighted_agg with method=sum
    """
    # Check with xr.DataArray
    result = agg(values, reduce_dims=reduce_dims, weights=weights, method="sum")
    xr.testing.assert_equal(result, expected)

    # Check with xr.Dataset
    values_ds = xr.Dataset(({"var1": values, "var2": values}))
    result_ds = agg(values_ds, reduce_dims=reduce_dims, weights=weights, method="sum")
    expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
    xr.testing.assert_equal(result_ds, expected_ds)


def test_agg_warns():
    """
    Test that a warning is raised if weights are provided but reduce_dims is None.
    """
    with pytest.warns(UserWarning):
        result = agg(DA_3x3, reduce_dims=None, weights=WEIGHTS1)
    xr.testing.assert_equal(DA_3x3, result)


@pytest.mark.parametrize(
    ("weights", "method", "msg"),
    [
        # Negative weights
        (NEGATIVE_WEIGHTS, "mean", "Weights must not contain negative values."),
        # NaN weights
        (NAN_WEIGHTS, "mean", "Weights must not contain NaN values."),
        # Wrong method
        (None, "agg", "Method must be either 'mean' or 'sum', got 'agg'"),
    ],
)
def test_agg_raises(weights, method, msg):
    """
    Test that a Value error is raised if there are negative weights
    """

    with pytest.raises(ValueError, match=msg):
        agg(DA_3x3, reduce_dims=["x"], weights=weights, method=method)
