"""
Contains tests for weighting scores appropriately.
"""

import numpy as np
import pytest
import xarray as xr

from scores.processing import aggregate
from scores.utils import ERROR_INVALID_WEIGHTS

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
    [4, 7, np.nan],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_SUM_NEW_DIM = xr.DataArray(
    [[99, 198], [1.5, 3], [3, 6]],
    dims=["y", "z"],
    coords={"z": [0, 1], "y": [0, 1, 2]},
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
EXP_DA_Y_SUM_BROADCAST = xr.DataArray(
    [1, 2, 99],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_VARY_XY = xr.DataArray(1.1)
EXP_DA_SUM_VARY_XY = xr.DataArray(11)
EXP_DS_Y = xr.Dataset(({"var1": EXP_DA_Y, "var2": EXP_DA_Y}))
WEIGHTS1 = xr.DataArray(
    [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)
WEIGHTS1_DS = xr.Dataset(({"var1": WEIGHTS1, "var2": WEIGHTS1}))

WEIGHTS2 = xr.DataArray(
    [[1, 2, 3], [3, 2, 1], [0, 0, 0]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)
WEIGHTS3 = xr.DataArray(
    [1, 2],
    dims=["z"],
    coords={"z": [0, 1]},
)
WEIGHTS4 = xr.DataArray(
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
        # Varying weights over y
        (DA_3x3, ["y"], WEIGHTS2, EXP_DA_VARY),
        # Extra dim in weights
        (DA_3x3, ["x"], WEIGHTS3, EXP_DA_NEW_DIM),
        # Weights only for 1 dim to test broadcasting,
        (DA_3x3, ["y"], WEIGHTS4, EXP_DA_Y_BROADCAST),
        # Varying weights over x and y
        (DA_3x3, ["x", "y"], WEIGHTS2, EXP_DA_VARY_XY),
        # Varying weights over y, but input error is NaN
        (DA_3x3 * np.nan, ["y"], WEIGHTS2, EXP_DA_VARY * np.nan),
    ],
)
def test_aggregate_mean(values, reduce_dims, weights, expected):
    """
    Tests scores.functions.aggregate with method=mean
    """
    # Check with xr.DataArray
    result = aggregate(values, reduce_dims=reduce_dims, weights=weights)
    xr.testing.assert_equal(result, expected)

    # Check with xr.Dataset for values, but xr.DataArray for weights
    values_ds = xr.Dataset(({"var1": values, "var2": values}))
    result_ds = aggregate(values_ds, reduce_dims=reduce_dims, weights=weights)
    expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
    xr.testing.assert_equal(result_ds, expected_ds)

    # Check with xr.Dataset for both values and weights
    if weights is not None:
        weights_ds = xr.Dataset(({"var1": weights, "var2": weights}))
        result_ds = aggregate(values_ds, reduce_dims=reduce_dims, weights=weights_ds)
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
        (DA_3x3, ["y"], WEIGHTS1, 2 * EXP_DA_Y_SUM),
        # Varying weights over y
        (DA_3x3, ["y"], WEIGHTS2, EXP_DA_VARY_SUM),
        # Extra dim in weights
        (DA_3x3, ["x"], WEIGHTS3, EXP_DA_SUM_NEW_DIM),
        # Weights only for 1 dim to test broadcasting
        (DA_3x3, ["y"], WEIGHTS4, EXP_DA_Y_SUM_BROADCAST),
        # Varying weights over x and y
        (DA_3x3, ["x", "y"], WEIGHTS2, EXP_DA_SUM_VARY_XY),
        # Varying weights over y, but input error is NaN
        (DA_3x3 * np.nan, ["y"], WEIGHTS2, EXP_DA_VARY * np.nan),
    ],
)
def test_aggregate_sum(values, reduce_dims, weights, expected):
    """
    Tests scores.functions.aggregate with method=sum
    """
    # Check with xr.DataArray
    result = aggregate(values, reduce_dims=reduce_dims, weights=weights, method="sum")
    xr.testing.assert_equal(result, expected)

    # Check with xr.Dataset
    values_ds = xr.Dataset(({"var1": values, "var2": values}))
    result_ds = aggregate(values_ds, reduce_dims=reduce_dims, weights=weights, method="sum")
    expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
    xr.testing.assert_equal(result_ds, expected_ds)

    # Check with xr.Dataset for both values and weights
    if weights is not None:
        weights_ds = xr.Dataset(({"var1": weights, "var2": weights}))
        result_ds = aggregate(values_ds, reduce_dims=reduce_dims, weights=weights_ds, method="sum")
        expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
        xr.testing.assert_equal(result_ds, expected_ds)


def test_agg_warns():
    """
    Test that a warning is raised if weights are provided but reduce_dims is None.
    """
    with pytest.warns(UserWarning):
        result = aggregate(DA_3x3, reduce_dims=None, weights=WEIGHTS1)
    xr.testing.assert_equal(DA_3x3, result)


@pytest.mark.parametrize(
    ("values", "weights", "method", "msg", "err_type"),
    [
        # Negative weights, DA values
        (DA_3x3, NEGATIVE_WEIGHTS, "mean", ERROR_INVALID_WEIGHTS, ValueError),
        # Negative weights, DS values, DA weights
        (
            xr.Dataset(({"var1": DA_3x3, "var2": DA_3x3})),
            NEGATIVE_WEIGHTS,
            "mean",
            ERROR_INVALID_WEIGHTS,
            ValueError,
        ),
        # Negative weights, DS values, DS weights
        (
            xr.Dataset(({"var1": DA_3x3, "var2": DA_3x3})),
            xr.Dataset(({"var1": NEGATIVE_WEIGHTS, "var2": NEGATIVE_WEIGHTS})),
            "mean",
            ERROR_INVALID_WEIGHTS,
            ValueError,
        ),
        # NaN weights, DA values
        (DA_3x3, NAN_WEIGHTS, "mean", ERROR_INVALID_WEIGHTS, ValueError),
        # NaN weights, DS values, DA weights
        (
            xr.Dataset(({"var1": DA_3x3, "var2": DA_3x3})),
            NAN_WEIGHTS,
            "mean",
            ERROR_INVALID_WEIGHTS,
            ValueError,
        ),
        # NaN weights, DS values, DS weights
        (
            xr.Dataset(({"var1": DA_3x3, "var2": DA_3x3})),
            xr.Dataset(({"var1": NAN_WEIGHTS, "var2": NAN_WEIGHTS})),
            "mean",
            ERROR_INVALID_WEIGHTS,
            ValueError,
        ),
        # Wrong method
        (DA_3x3, None, "agg", "Method must be either 'mean' or 'sum', got 'agg'", ValueError),
        # DS weights missing data var
        (
            xr.Dataset(({"var1": DA_3x3, "var2": DA_3x3})),
            xr.Dataset(({"var1": WEIGHTS1})),
            "mean",
            "No weights provided for variable 'var2'",
            KeyError,
        ),
        # DA values, DS weights
        (
            DA_3x3,
            xr.Dataset(({"var1": WEIGHTS1, "var2": WEIGHTS1})),
            "mean",
            "`weights` cannot be an xr.Dataset when `values` is an xr.DataArray",
            ValueError,
        ),
    ],
)
def test_aggregate_raises(values, weights, method, msg, err_type):
    """
    Test that a Value error is raised if there are negative weights
    """

    with pytest.raises(err_type, match=msg):
        aggregate(values, reduce_dims=["x"], weights=weights, method=method)
