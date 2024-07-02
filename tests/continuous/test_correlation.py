import numpy as np
import pytest
import xarray as xr

from scores.continuous.correlation import pearsonr

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
def test_correlation(da1, da2, reduce_dims, preserve_dims, expected):
    """
    Tests continuous.correlation
    """
    result = pearsonr(da1, da2, preserve_dims=preserve_dims, reduce_dims=reduce_dims)
    xr.testing.assert_allclose(result, expected)


def test_correlation_dask():
    """
    Tests continuous.correlation works with Dask
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = pearsonr(DA3_CORR.chunk(), DA2_CORR.chunk())
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, EXP_CORR_REDUCE_ALL)
