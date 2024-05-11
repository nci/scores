"""
Contains unit tests for scores.probability.fss_impl
"""
import numpy as np
import pytest
import xarray as xr

from scores import sample_data as sd
from scores.spatial.fss_impl import fss_2d, fss_2d_binary, fss_2d_single_field
from scores.utils import DimensionError
from tests.spatial import fss_test_data as ftd


@pytest.mark.parametrize(
    ("obs_pdf", "fcst_pdf", "window_size", "event_threshold", "expected"),
    [
        ((0.0, 1.0), (0.0, 1.0), (10, 10), 0.5, 0.974078),
        ((0.0, 1.0), (0.0, 2.0), (5, 5), 0.5, 0.888756),
        ((0.0, 1.0), (1.0, 1.0), (10, 10), 0.25, 0.812008),
    ],
)
def test_fss_2d_single_field(obs_pdf, fcst_pdf, window_size, event_threshold, expected):
    """
    Integration test to check that fss is generally working for a single field
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate(obs_pdf, fcst_pdf, seed=seed)
    res = fss_2d_single_field(fcst, obs, event_threshold=event_threshold, window_size=window_size, zero_padding=False)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("obs_pdf", "fcst_pdf", "window_size", "event_threshold", "expected"),
    [
        ((0.0, 1.0), (0.0, 1.0), (10, 10), 0.5, 0.970884),
        ((0.0, 1.0), (0.0, 2.0), (5, 5), 0.5, 0.90405),
        ((0.0, 1.0), (1.0, 1.0), (10, 10), 0.25, 0.811622),
    ],
)
def test_fss_2d_single_field_zero_pad(obs_pdf, fcst_pdf, window_size, event_threshold, expected):
    """
    Integration test to check that fss is generally working for a single field,
    but with zero padding instead of interior border.
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate(obs_pdf, fcst_pdf, seed=seed)
    res = fss_2d_single_field(fcst, obs, event_threshold=event_threshold, window_size=window_size, zero_padding=True)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("window_size", "event_threshold", "reduce_dims", "preserve_dims", "expected"),
    [
        ((2, 2), 2.0, None, None, xr.DataArray(0.96813032)),
        ((2, 2), 8.0, None, None, xr.DataArray(0.78144748)),
        ((5, 5), 2.0, None, None, xr.DataArray(0.99381861)),
        ((5, 5), 8.0, None, None, xr.DataArray(0.94054107)),  # output_dims: scalar
        ((2, 2), 5.0, ["time"], None, ftd.EXPECTED_TEST_FSS_2D_REDUCE_TIME),  # output_dims: 1 x lead_time
        ((2, 2), 5.0, None, ["time"], ftd.EXPECTED_TEST_FSS_2D_PRESERVE_TIME),  # output_dims: 1 x time
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            ftd.EXPECTED_TEST_FSS_2D_PRESERVE_ALL,
        ),  # output_dims: time x lead_time
    ],
)
def test_fss_2d(window_size, event_threshold, reduce_dims, preserve_dims, expected):
    """
    Tests various combinations of window_size/event_threshold/preserve_dims/reduce_dims

    Note: does not include error/assertion testing this will be done separately.
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    res = fss_2d(
        da_fcst,
        da_obs,
        event_threshold=event_threshold,
        window_size=window_size,
        spatial_dims=["lat", "lon"],
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    xr.testing.assert_allclose(res, expected)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("window_size", "event_threshold", "reduce_dims", "preserve_dims", "expected"),
    [
        ((2, 2), 2.0, None, None, xr.DataArray(0.96813032)),
        ((2, 2), 8.0, None, None, xr.DataArray(0.78144748)),
        ((5, 5), 2.0, None, None, xr.DataArray(0.99381861)),
        ((5, 5), 8.0, None, None, xr.DataArray(0.94054107)),  # output_dims: scalar
        ((2, 2), 5.0, ["time"], None, ftd.EXPECTED_TEST_FSS_2D_REDUCE_TIME),  # output_dims: 1 x lead_time
        ((2, 2), 5.0, None, ["time"], ftd.EXPECTED_TEST_FSS_2D_PRESERVE_TIME),  # output_dims: 1 x time
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            ftd.EXPECTED_TEST_FSS_2D_PRESERVE_ALL,
        ),  # output_dims: time x lead_time
    ],
)
def test_fss_2d_binary(window_size, event_threshold, reduce_dims, preserve_dims, expected):
    """
    Tests various combinations of window_size/event_threshold/preserve_dims/reduce_dims

    Note: does not include error/assertion testing this will be done separately.
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    res = fss_2d_binary(
        np.greater(da_fcst, event_threshold),
        np.greater(da_obs, event_threshold),
        window_size=window_size,
        spatial_dims=["lat", "lon"],
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )
    xr.testing.assert_allclose(res, expected)


@pytest.mark.parametrize(("large_obs"), [(True), (False)])
def test_invalid_input_dimensions(large_obs):
    """
    Compares a large input with a small input - should raise an error
    """
    # NOTE: large continuous forecast takes too long, so using continuous_observations
    # to generate the forecast dataset as well. This test only really cares about the
    # input spatial dimensions being compatible.
    da_fcst = sd.continuous_observations(large_size=large_obs)
    da_obs = sd.continuous_observations(large_size=not large_obs)
    with pytest.raises(DimensionError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=5.0,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
        )

    with pytest.raises(DimensionError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=5.0,
            window_size=(5, 5),
            spatial_dims=["lat"],
        )


@pytest.mark.parametrize(("window_size"), [(50, 10), (-1, 5), (5, -1), (-1, -1), (10, 50), (50, 50)])
def test_invalid_window_size(window_size):
    """
    Check for various invalid window_size sizes (note: assumes small dataset is used)
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    with pytest.raises(DimensionError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=5.0,
            window_size=window_size,
            spatial_dims=["lat", "lon"],
        )


def test_zero_denom_fss():
    """
    Force denominator of the input field to be 0.0 to test against divide by zero
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    da_fcst[:] = 0.0
    da_obs[:] = 0.0
    res = fss_2d(
        da_fcst,
        da_obs,
        event_threshold=5.0,
        window_size=(5, 5),
        spatial_dims=["lat", "lon"],
    )
    assert res == xr.DataArray(0.0)


def test_zero_denom_fss_single_field():
    """
    Force denominator of the input field to be 0.0 to test against divide by zero
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate((0, 0), (0, 0), seed=seed)
    res = fss_2d_single_field(fcst, obs, event_threshold=1.0, window_size=(5, 5))
    assert res == 0.0
