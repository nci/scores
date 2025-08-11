"""
Containts tests for weighting scores appropriately.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest
import xarray as xr

import scores.continuous
import scores.functions

# Tests for apply_weighted_mean

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
EXP_DA_Y = xr.DataArray(
    [0.5, 1.5, 33],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_NAN = xr.DataArray(
    [np.nan, np.nan, np.nan],
    dims=["x"],
    coords={"x": [0, 1, 2]},
)
EXP_DA_VARY = xr.DataArray(
    [2 / 3, (2 * 3 + 1) / 4, np.nan],
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
WEIGHTS3 = WEIGHTS1 * np.nan
WEIGHTS4 = xr.DataArray(
    [[1, 2, 3], [3, 2, 1], [0, np.nan, np.nan]],
    dims=["x", "y"],
    coords={"x": [0, 1, 2], "y": [0, 1, 2]},
)
WEIGHTS5 = xr.DataArray(
    [1, 2],
    dims=["z"],
    coords={"z": [0, 1]},
)
WEIGHTS6 = xr.DataArray(
    [1, 2, np.nan],
    dims=["y"],
    coords={"y": [0, 1, 3]},
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
        # NaN weights over y
        (DA_3x3, ["y"], WEIGHTS3, EXP_DA_NAN),
        # Varying weights over y
        (DA_3x3, ["y"], WEIGHTS4, EXP_DA_VARY),
        # Extra dim in weights
        (DA_3x3, ["x"], WEIGHTS5, EXP_DA_NEW_DIM),
        # Weights only for 1 dim to test broadcasting, with NaN
        (DA_3x3, ["y"], WEIGHTS6, EXP_DA_Y_BROADCAST),
        # Varying weights over x and y
        (DA_3x3, ["x", "y"], WEIGHTS4, EXP_DA_VARY_XY),
        # Varying weights over y, but input error is NaN
        (DA_3x3 * np.nan, ["y"], WEIGHTS4, EXP_DA_VARY * np.nan),
    ],
)
def test_apply_weighted_mean(values, reduce_dims, weights, expected):
    """
    Tests scores.functions.apply_weighted_mean
    #"""
    # Check with xr.DataArray
    result = scores.functions.apply_weighted_mean(values, reduce_dims=reduce_dims, weights=weights)
    xr.testing.assert_equal(result, expected)

    # Check with xr.Dataset
    values_ds = xr.Dataset(({"var1": values, "var2": values}))
    result_ds = scores.functions.apply_weighted_mean(values_ds, reduce_dims=reduce_dims, weights=weights)
    expected_ds = xr.Dataset(({"var1": expected, "var2": expected}))
    xr.testing.assert_equal(result_ds, expected_ds)


def test_apply_weighted_mean_warns():
    """
    Test that a warning is raised if weights are provided but reduce_dims is None.
    """
    with pytest.warns(UserWarning) as record:
        result = scores.functions.apply_weighted_mean(DA_3x3, reduce_dims=None, weights=WEIGHTS1)
    xr.testing.assert_equal(DA_3x3, result)


def test_apply_weighted_mean_raises():
    """
    Test that a Value error is raised if there are negative weights
    """
    negative_weights = xr.DataArray(
        [[1, -1, 1], [1, 1, 1], [1, 1, 1]],
        dims=["x", "y"],
        coords={"x": [0, 1, 2], "y": [0, 1, 2]},
    )

    with pytest.raises(ValueError, match="Weights must not contain negative values."):
        scores.functions.apply_weighted_mean(DA_3x3, reduce_dims=["x"], weights=negative_weights)


# Old tests
ZERO = np.array([[1 for i in range(10)] for j in range(10)])


# Standard forecast and observed test data which is static and can be used
# across tests
np.random.seed(0)
LATS = [50, 51, 52, 53]
LONS = [30, 31, 32, 33]
fcst_temperatures_2d = 15 + 8 * np.random.randn(1, 4, 4)
obs_temperatures_2d = 15 + 6 * np.random.randn(1, 4, 4)
FCST_2D = xr.DataArray(fcst_temperatures_2d[0], dims=["latitude", "longitude"], coords=[LATS, LONS])
OBS_2D = xr.DataArray(obs_temperatures_2d[0], dims=["latitude", "longitude"], coords=[LATS, LONS])
IDENTITY = np.ones((4, 4))
ZEROS = np.zeros((4, 4))


def simple_da(data):
    """
    Helper function for making a DataArray with a latitude coordinate variable roughly
    over the Australian region
    """
    lats = np.arange(-5, -45, (-45 / len(data)))
    arr = xr.DataArray.from_dict(
        {
            "coords": {
                "lat": {"data": lats, "dims": "lat"},
            },
            "data": data,
        }
    )
    return arr


# These scores will be tested for valid processing of weights
all_scores = [scores.continuous.mse, scores.continuous.mae]


def test_weights_identity():
    for score in all_scores:
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=IDENTITY)
        assert unweighted == weighted


def test_weights_zeros():
    for score in all_scores:
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=ZEROS)

        assert unweighted != weighted
        assert weighted.sum() == 0


def test_weights_latitude():
    """
    Tests the use of latitude weightings
    """

    lat_weightings_values = scores.functions.create_latitude_weights(OBS_2D.latitude)

    for score in all_scores:
        unweighted = score(FCST_2D, OBS_2D)
        weighted = score(FCST_2D, OBS_2D, weights=lat_weightings_values)
        assert unweighted != weighted

    # Latitudes in degrees, tested to 8 decimal places
    latitude_tests = [
        (90, 0),
        (89, 0.017452),
        (45, 0.707107),
        (22.5, 0.92388),
        (0, 1),
        (-22.5, 0.92388),
        (-45, 0.707107),
        (-89, 0.017452),
        (-90, 0),
    ]
    latitudes, expected = zip(*latitude_tests)
    latitudes = xr.DataArray(list(latitudes))  # Will not work from a tuple
    expected = xr.DataArray(list(expected))  # Will not work from a tuple

    found = scores.functions.create_latitude_weights(latitudes)
    decimal_places = 6
    found = found.round(decimal_places)
    expected = expected.round(decimal_places)
    assert found.equals(expected)


def test_weights_nan_matching():
    da = xr.DataArray

    fcst = da([np.nan, 0, 1, 2, 7, 0, 7, 1])
    obs = da([np.nan, np.nan, 0, 1, 7, 0, 7, 0])
    weights = da([1, 1, 1, 1, 1, 1, 0, np.nan])
    expected = da([np.nan, np.nan, 1, 1, 0, 0, 0, np.nan])

    result = scores.continuous.mae(fcst, obs, weights=weights, preserve_dims="all")
    assert isinstance(result, xr.DataArray)
    assert isinstance(expected, xr.DataArray)

    assert result.equals(expected)


def test_weights_add_dimension():
    """
    Test what happens when additional dimensions are added into weights which are not present in
    fcst or obs. Repeats some of the NaN matching but the focus is really on the dimensional
    expansion, using the same data to slowly build up the example and establish confidence.
    """

    da = simple_da  # Make a DataArray with a latitude dimension

    fcst = da([np.nan, 0, 1, 2, 7, 0, 7, 1])
    obs = da([np.nan, np.nan, 0, 1, 7, 0, 7, 0])
    simple_weights = [1, 1, 1, 1, 1, 1, 0, np.nan]
    double_weights = [2, 2, 2, 2, 2, 2, 0, np.nan]
    simple_expect = [np.nan, np.nan, 1, 1, 0, 0, 0, np.nan]
    double_expect = [np.nan, np.nan, 2, 2, 0, 0, 0, np.nan]

    simple = scores.continuous.mae(fcst, obs, weights=da(simple_weights), preserve_dims="all")
    doubled = scores.continuous.mae(fcst, obs, weights=da(double_weights), preserve_dims="all")

    assert simple.equals(da(simple_expect))  # type: ignore  # Static analysis mireports this
    assert doubled.equals(da(double_expect))  # type: ignore  # Static analysis mireports this

    composite_weights_data = [simple_weights, double_weights]
    composite_expected_data = [simple_expect, double_expect]

    composite_weights = xr.DataArray.from_dict(
        {
            "coords": {
                "method": {"data": ["simpleweight", "doubleweight"], "dims": "method"},
                "lat": {"data": list(fcst.lat), "dims": "lat"},
            },
            "data": composite_weights_data,
        }
    )

    composite_expected = xr.DataArray.from_dict(
        {
            "coords": {
                "method": {"data": ["simpleweight", "doubleweight"], "dims": "method"},
                "lat": {"data": list(fcst.lat), "dims": "lat"},
            },
            "data": composite_expected_data,
        }
    )

    composite = scores.continuous.mae(fcst, obs, weights=composite_weights, preserve_dims="all").transpose()
    composite.broadcast_equals(composite_expected)  # type: ignore  # Static analysis mireports this
