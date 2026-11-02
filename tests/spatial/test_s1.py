"""
Contains unit tests for scores.spatial.s1_impl
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=R0801

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

import scores.spatial

# S1 score

LATS_S1 = [-35, -30, -25]
LONS_S1 = [140, 150, 160]

OBS_S1 = xr.DataArray(
    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
    coords={"lat": LATS_S1, "lon": LONS_S1},
    dims=["lat", "lon"],
)
FCST_S1 = xr.DataArray(
    [[1.3, 1.8, 3.4], [1.9, 3.3, 3.8], [3.1, 4.4, 4.6]],
    coords={"lat": LATS_S1, "lon": LONS_S1},
    dims=["lat", "lon"],
)
S1_WEIGHTS = xr.DataArray(np.cos(np.deg2rad(LATS_S1)), coords={"lat": LATS_S1}, dims=["lat"])

EXP_S1_UNWEIGHTED = xr.DataArray(36.17021276595745)
EXP_S1_WEIGHTED = xr.DataArray(35.970278594111335)


@pytest.mark.parametrize(
    ("fcst", "obs", "x_dim", "y_dim", "weights", "expected"),
    [
        # Basic 2D grid, no weights
        (FCST_S1, OBS_S1, "lon", "lat", None, EXP_S1_UNWEIGHTED),
        # With cos(latitude) weighting
        (FCST_S1, OBS_S1, "lon", "lat", S1_WEIGHTS, EXP_S1_WEIGHTED),
        # A perfect forecast has an S1 score of 0
        (OBS_S1, OBS_S1, "lon", "lat", None, xr.DataArray(0.0)),
    ],
)
def test_s1(fcst, obs, x_dim, y_dim, weights, expected):
    """
    Tests scores.spatial.s1
    """
    result = scores.spatial.s1(fcst, obs, x_dim=x_dim, y_dim=y_dim, weights=weights)
    xr.testing.assert_allclose(result, expected)


TIMES_S1 = ["t1", "t2"]
EXP_S1_UNWEIGHTED_TIME = xr.DataArray([36.17021276595745, 36.17021276595745], coords={"time": TIMES_S1}, dims=["time"])


def test_s1_preserve_dims():
    """
    Tests that scores.spatial.s1 correctly preserves non-grid dimensions
    """
    fcst = xr.concat([FCST_S1, FCST_S1 * 1.1], dim="time").assign_coords(time=TIMES_S1)
    obs = xr.concat([OBS_S1, OBS_S1 * 1.1], dim="time").assign_coords(time=TIMES_S1)

    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat", preserve_dims="time")
    xr.testing.assert_allclose(result, EXP_S1_UNWEIGHTED_TIME)


def test_s1_dataset():
    """
    Tests that scores.spatial.s1 works with xr.Dataset inputs
    """
    fcst = xr.Dataset({"a": FCST_S1})
    obs = xr.Dataset({"a": OBS_S1})
    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat")
    expected = xr.Dataset({"a": EXP_S1_UNWEIGHTED})
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("x_dim", "y_dim", "preserve_dims", "expected_message"),
    [
        # x_dim not present in fcst/obs
        ("not_a_dim", "lat", None, "`x_dim` \\('not_a_dim'\\) must be a dimension of both `fcst` and `obs`"),
        # y_dim not present in fcst/obs
        ("lon", "not_a_dim", None, "`y_dim` \\('not_a_dim'\\) must be a dimension of both `fcst` and `obs`"),
        # x_dim and y_dim are the same
        ("lat", "lat", None, "`x_dim` and `y_dim` must be different dimensions"),
        # x_dim preserved rather than reduced
        ("lon", "lat", "lon", "The S1 score is computed over the verification grid"),
    ],
)
def test_s1_raises(x_dim, y_dim, preserve_dims, expected_message):
    """
    Tests that scores.spatial.s1 raises a ValueError for invalid dimension arguments
    """
    with pytest.raises(ValueError, match=expected_message):
        scores.spatial.s1(FCST_S1, OBS_S1, x_dim=x_dim, y_dim=y_dim, preserve_dims=preserve_dims)


def test_s1_dask():
    """
    Tests that scores.spatial.s1 works with dask
    """
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    fcst = FCST_S1.chunk()
    obs = OBS_S1.chunk()
    weights = S1_WEIGHTS.chunk()

    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat", weights=weights)
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, EXP_S1_WEIGHTED)


def test_s1_nan():
    """
    Tests that scores.spatial.s1 correctly handles NaNs in fcst/obs by excluding
    affected grid points from the aggregation (via `broadcast_and_match_nan`)
    """
    fcst = FCST_S1.copy()
    fcst.values[0, 1] = np.nan  # lat=-35, lon=150

    result = scores.spatial.s1(fcst, OBS_S1, x_dim="lon", y_dim="lat")
    expected = xr.DataArray(35.0)
    xr.testing.assert_allclose(result, expected)


def test_s1_broadcasting():
    """
    Tests that scores.spatial.s1 correctly broadcasts fcst and obs with mismatched dimensions
    """
    fcst = xr.concat([FCST_S1, FCST_S1], dim="time").assign_coords(time=TIMES_S1)
    # obs has no "time" dimension, so it must broadcast against fcst's "time" dimension
    obs = OBS_S1

    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat", preserve_dims="time")
    xr.testing.assert_allclose(result, EXP_S1_UNWEIGHTED_TIME)


def test_s1_reduce_dims():
    """
    Tests that scores.spatial.s1 works when `reduce_dims` is explicitly specified
    (as opposed to `preserve_dims`)
    """
    fcst = xr.concat([FCST_S1, FCST_S1 * 1.1], dim="time").assign_coords(time=TIMES_S1)
    obs = xr.concat([OBS_S1, OBS_S1 * 1.1], dim="time").assign_coords(time=TIMES_S1)

    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat", reduce_dims=["lon", "lat"])
    xr.testing.assert_allclose(result, EXP_S1_UNWEIGHTED_TIME)


def test_s1_is_invariant_to_latitude_order():
    """
    Reversing latitude order should not change S1.
    """
    result1 = scores.spatial.s1(
        FCST_S1,
        OBS_S1,
        x_dim="lon",
        y_dim="lat",
    )

    result2 = scores.spatial.s1(
        FCST_S1[::-1],
        OBS_S1[::-1],
        x_dim="lon",
        y_dim="lat",
    )

    xr.testing.assert_allclose(result1, result2)


def test_s1_corner_errors_are_included():
    """
    An error at a corner point should contribute to S1.
    """
    fcst = OBS_S1.copy()
    fcst[0, 0] = 10

    result = scores.spatial.s1(
        fcst,
        OBS_S1,
        x_dim="lon",
        y_dim="lat",
    )

    assert result > 0


def test_s1_constant_fields():
    """
    A constant field has zero gradient everywhere, so S1 is 0/0 (NaN).
    """
    fcst = xr.DataArray(
        np.full((3, 3), 5.0),
        dims=["lat", "lon"],
    )
    obs = xr.DataArray(
        np.full((3, 3), 5.0),
        dims=["lat", "lon"],
    )

    result = scores.spatial.s1(fcst, obs, x_dim="lon", y_dim="lat")

    assert np.isnan(result.item())
