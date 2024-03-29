"""
Containts tests for weighting scores appropriately.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import xarray as xr

import scores.continuous
import scores.functions

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


def test_weights_NaN_matching():  # pylint: disable=C0103
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
