"""
Contains unit tests for scores.probability.continuous
"""

try:
    import dask
    import dask.array
except:  # noqa: E722 allow bare except here  # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.continuous import quantile_score
from scores.utils import DimensionError
from tests.continuous import quantile_loss_test_data as qltd


@pytest.mark.parametrize("alpha", [1.0, 1.1, 0, -0.8])
def test_qsf_value_error_alpha(alpha):
    """quantile_score raises ValueError."""
    with pytest.raises(ValueError):
        quantile_score(qltd.DA1_2X2, qltd.DA1_2X2, alpha)


@pytest.mark.parametrize(
    ("obs", "reduce_dims", "preserve_dims"),
    [
        # fcst and obs with mismatched dims
        (qltd.DA1_2X2X2, None, None),
        # dims not in fcst or obs
        (qltd.DA1_2X2, ["bananas"], None),
        (qltd.DA1_2X2, None, ["orange"]),
    ],
)
def test_qsf_exceptions(obs, reduce_dims, preserve_dims):
    """quantile_score raises DimensionError."""
    with pytest.raises(DimensionError):
        quantile_score(qltd.DA1_2X2, obs, 0.5, reduce_dims=reduce_dims, preserve_dims=preserve_dims)


@pytest.mark.parametrize(
    ("fcst", "obs", "alpha", "preserve_dims", "reduce_dims", "weights", "expected"),
    [
        (
            qltd.DA1_2X2,
            qltd.DA2_2X2,
            0.7,
            ["i", "j"],
            None,
            None,
            xr.DataArray([[0.9, 0], [0.7, 1.4]], coords=[[0, 1], [0, 1]], dims=["i", "j"]),
        ),
        (
            qltd.DA2_2X2,
            qltd.DA1_2X2,
            0.7,
            ["i", "j"],
            None,
            None,
            xr.DataArray([[2.1, 0], [0.3, 0.6]], coords=[[0, 1], [0, 1]], dims=["i", "j"]),
        ),
        (
            qltd.FCST1,
            qltd.OBS1,
            0.2,
            ["valid_start", "station_index", "lead_time"],
            None,
            None,
            qltd.EXPECTED1,
        ),
        (
            qltd.FCST1,
            qltd.OBS1,
            0.1,
            ["lead_time"],
            None,
            None,
            qltd.EXPECTED2,
        ),
        (
            qltd.FCST1,
            qltd.OBS1,
            0.1,
            None,
            ["valid_start", "station_index"],
            None,
            qltd.EXPECTED2,
        ),
        # To test weight
        (
            qltd.FCST1,
            qltd.OBS1,
            0.1,
            None,
            None,
            qltd.WEIGHTS,
            qltd.EXPECTED4,
        ),
        # To test missing data
        (qltd.FCST2, qltd.FCST2, 0.8, ["lead_time"], None, None, qltd.EXPECTED3),
        (qltd.FCST2, qltd.FCST2, 0.8, None, None, None, xr.DataArray([0.0]).squeeze()),
        # To test function can handle Dataset as input
        (
            qltd.FCST_DS,
            qltd.OBS_DS,
            0.1,
            None,
            ["valid_start", "station_index"],
            None,
            qltd.EXPECTED_DS1,
        ),
        (
            qltd.FCST_DS,
            qltd.OBS_DS,
            0.1,
            None,
            None,
            qltd.WEIGHTS_DS,
            qltd.EXPECTED_DS2,
        ),
        # To test function handles reduce_dims='all'
        (
            qltd.FCST_DS,
            qltd.OBS_DS,
            0.1,
            None,
            "all",
            None,
            qltd.EXPECTED_DS3,
        ),
    ],
)
def test_qsf_calculations(fcst, obs, alpha, preserve_dims, reduce_dims, weights, expected):
    """quantile_score returns the expected object."""
    result = quantile_score(fcst, obs, alpha, preserve_dims=preserve_dims, reduce_dims=reduce_dims, weights=weights)
    xr.testing.assert_allclose(result, expected)


def test_quantile_score_dask():
    """Tests quantile_score works with dask"""

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    result = quantile_score(
        fcst=qltd.FCST1.chunk(),
        obs=qltd.OBS1.chunk(),
        alpha=0.1,
        reduce_dims=["valid_start", "station_index"],
    )
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    assert isinstance(result.data, np.ndarray)
    xr.testing.assert_allclose(result, qltd.EXPECTED2)
