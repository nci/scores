"""
Contains unit tests for scores.probability.fss_impl
"""

import numpy as np
import pytest
import xarray as xr

from scores import sample_data as sd
from scores.spatial.fss_impl import (
    _aggregate_fss_decomposed,
    fss_2d,
    fss_2d_single_field,
)
from scores.utils import DimensionError, FieldTypeError
from tests.spatial import fss_test_data as ftd


def test_nan_2d_single_field_curated_nan_cross():
    """
    Check that NaNs are handled as expected.

    This also applies to ``fss_2d`` and ``_fss_2d_without_ref`` because they both use
    similar functionality as ``fss_2d_single_field``.
    """
    window_size = (ftd.FSS_CURATED_WINDOW_SIZE_3X3, ftd.FSS_CURATED_WINDOW_SIZE_3X3)
    fcst = np.array(np.copy(ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST), dtype=np.float64)
    obs = np.array(np.copy(ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS), dtype=np.float64)
    # run a cross (+) through the dataset with NaN
    fcst[2, :] = np.array(np.nan)
    fcst[:, 2] = np.array(np.nan)
    obs[2, :] = np.array(np.nan)
    obs[:, 2] = np.array(np.nan)
    res_1 = fss_2d_single_field(
        obs, fcst, event_threshold=ftd.FSS_CURATED_THRESHOLD, window_size=window_size, padding=None
    )

    fcst = np.copy(ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST)
    obs = np.copy(ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS)
    # run a cross (+) through the dataset with zero - this should
    # give the same result.
    fcst[2, :] = 0
    fcst[:, 2] = 0
    obs[2, :] = 0
    obs[:, 2] = 0
    res_2 = fss_2d_single_field(
        obs, fcst, event_threshold=ftd.FSS_CURATED_THRESHOLD, window_size=window_size, padding=None
    )
    np.testing.assert_allclose(
        res_1["FSS"],
        res_2["FSS"],
    )
    np.testing.assert_allclose(
        res_1["FBS"],
        res_2["FBS"],
    )
    np.testing.assert_allclose(
        res_1["FBS_ref"],
        res_2["FBS_ref"],
    )


def test_fss_2d_single_field_curated_1x1_window():
    """
    Check that a 1x1 window performs as expected
    """
    window_size = (1, 1)
    fcst = ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST
    obs = ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS
    th = ftd.FSS_CURATED_THRESHOLD
    obs_bin = np.array(obs > th, dtype=np.int64)
    fcst_bin = np.array(fcst > th, dtype=np.int64)
    denom = np.mean(obs_bin * obs_bin + fcst_bin * fcst_bin)
    numer = np.mean((obs_bin - fcst_bin) * (obs_bin - fcst_bin))
    expected = 1.0 - numer / denom
    res = fss_2d_single_field(obs, fcst, event_threshold=th, window_size=window_size, padding=None)
    np.testing.assert_allclose(res["FSS"], expected)
    np.testing.assert_allclose(res["FBS"], numer)
    np.testing.assert_allclose(res["FBS_ref"], denom)


def test_fss_2d_single_field_curated_window_equal_data_size():
    """
    Check that a data = window size performs as expected
    """
    window_size = (4, 5)
    total_size = window_size[0] * window_size[1]
    fcst = ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST
    obs = ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS
    th = ftd.FSS_CURATED_THRESHOLD
    obs_bin = np.sum(obs > th) / total_size
    fcst_bin = np.sum(fcst > th) / total_size
    denom = np.mean(obs_bin * obs_bin + fcst_bin * fcst_bin)
    numer = np.mean((obs_bin - fcst_bin) * (obs_bin - fcst_bin))
    expected = 1.0 - numer / denom
    res = fss_2d_single_field(obs, fcst, event_threshold=th, window_size=window_size, padding=None)
    np.testing.assert_allclose(res["FSS"], expected)
    np.testing.assert_allclose(res["FBS"], numer)
    np.testing.assert_allclose(res["FBS_ref"], denom)


@pytest.mark.parametrize(
    ("obs", "fcst", "window_size", "event_threshold", "expected"),
    [
        (
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_3X3, ftd.FSS_CURATED_WINDOW_SIZE_3X3),
            ftd.FSS_CURATED_THRESHOLD,
            ftd.EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW,
        ),
        (
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_4X4, ftd.FSS_CURATED_WINDOW_SIZE_4X4),
            ftd.FSS_CURATED_THRESHOLD,
            ftd.EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW,
        ),
    ],
)
def test_fss_2d_single_field_curated(obs, fcst, window_size, event_threshold, expected):
    """
    Odd/even sized window with hand-crafted data
    """
    res = fss_2d_single_field(obs, fcst, event_threshold=event_threshold, window_size=window_size, padding=None)
    np.testing.assert_allclose(res["FSS"], expected[0])
    np.testing.assert_allclose(res["FBS"], expected[1])
    np.testing.assert_allclose(res["FBS_ref"], expected[2])


@pytest.mark.parametrize(
    ("obs", "obs_padded", "fcst", "fcst_padded", "window_size", "event_threshold", "padding", "expected"),
    [
        (
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_OBS,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_3X3, ftd.FSS_CURATED_WINDOW_SIZE_3X3),
            ftd.FSS_CURATED_THRESHOLD,
            "zero",
            ftd.EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_ZERO_PADDED,
        ),
        (
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_OBS,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_FCST,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_4X4, ftd.FSS_CURATED_WINDOW_SIZE_4X4),
            ftd.FSS_CURATED_THRESHOLD,
            "zero",
            ftd.EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_ZERO_PADDED,
        ),
        (
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_OBS,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_FCST,
            ftd.FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_3X3, ftd.FSS_CURATED_WINDOW_SIZE_3X3),
            ftd.FSS_CURATED_THRESHOLD,
            "reflective",
            ftd.EXPECTED_FSS_CURATED_TEST_4X5_DATA_3X3_WINDOW_REFLECTIVE_PADDED,
        ),
        (
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_OBS,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_OBS,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_FCST,
            ftd.FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED_FCST,
            (ftd.FSS_CURATED_WINDOW_SIZE_4X4, ftd.FSS_CURATED_WINDOW_SIZE_4X4),
            ftd.FSS_CURATED_THRESHOLD,
            "reflective",
            ftd.EXPECTED_FSS_CURATED_TEST_5X6_DATA_4X4_WINDOW_REFLECTIVE_PADDED,
        ),
    ],
)
def test_fss_2d_single_field_curated_padded(
    obs, obs_padded, fcst, fcst_padded, window_size, event_threshold, padding, expected
):
    """
    Odd/even sized window with hand-crafted data, with zero and reflective padding.

    Also tests that pre-padded inputs match zero/reflective-padding from the algorithm.
    """
    # test in-built padding
    res_1 = fss_2d_single_field(fcst, obs, event_threshold=event_threshold, window_size=window_size, padding=padding)
    np.testing.assert_allclose(res_1["FSS"], expected[0])
    np.testing.assert_allclose(res_1["FBS"], expected[1])
    np.testing.assert_allclose(res_1["FBS_ref"], expected[2])

    # test pre-padded input data
    res_2 = fss_2d_single_field(
        fcst_padded, obs_padded, event_threshold=event_threshold, window_size=window_size, padding=None
    )
    np.testing.assert_allclose(res_2["FSS"], expected[0])
    np.testing.assert_allclose(res_2["FBS"], expected[1])
    np.testing.assert_allclose(res_2["FBS_ref"], expected[2])

    # assert that both results are the same
    np.testing.assert_allclose(res_1["FSS"], res_2["FSS"])
    np.testing.assert_allclose(res_1["FBS"], res_2["FBS"])
    np.testing.assert_allclose(res_1["FBS_ref"], res_2["FBS_ref"])


@pytest.mark.parametrize(
    ("obs_pdf", "fcst_pdf", "window_size", "event_threshold", "expected"),
    [
        ((0.0, 1.0), (0.0, 1.0), (10, 10), 0.5, [0.974733, 0.0049381, 0.195445]),
        ((0.0, 1.0), (0.0, 2.0), (5, 5), 0.5, [0.88913, 0.030689, 0.276801]),
        ((0.0, 1.0), (1.0, 1.0), (10, 10), 0.25, [0.81069, 0.144627, 0.763968]),
    ],
)
def test_fss_2d_single_field(obs_pdf, fcst_pdf, window_size, event_threshold, expected):
    """
    Integration test to check that fss is generally working for a single field
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate(obs_pdf, fcst_pdf, seed=seed)
    res = fss_2d_single_field(fcst, obs, event_threshold=event_threshold, window_size=window_size, padding=None)
    np.testing.assert_allclose(res["FSS"], expected[0], rtol=1e-5)
    np.testing.assert_allclose(res["FBS"], expected[1], rtol=1e-5)
    np.testing.assert_allclose(res["FBS_ref"], expected[2], rtol=1e-5)


@pytest.mark.parametrize(
    ("obs_pdf", "fcst_pdf", "window_size", "event_threshold", "expected"),
    [
        ((0.0, 1.0), (0.0, 1.0), (10, 10), 0.5, [0.972075, 0.0045055, 0.161344]),
        ((0.0, 1.0), (0.0, 2.0), (5, 5), 0.5, [0.887028, 0.0288287, 0.255183]),
        ((0.0, 1.0), (1.0, 1.0), (10, 10), 0.25, [0.811639, 0.116971, 0.620991]),
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
    res = fss_2d_single_field(fcst, obs, event_threshold=event_threshold, window_size=window_size, padding="zero")
    np.testing.assert_allclose(res["FSS"], expected[0], rtol=1e-5)
    np.testing.assert_allclose(res["FBS"], expected[1], rtol=1e-5)
    np.testing.assert_allclose(res["FBS_ref"], expected[2], rtol=1e-5)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    (
        "window_size",
        "event_threshold",
        "reduce_dims",
        "preserve_dims",
        "padding",
        "benchmark",
        "is_binary",
        "check_boolean",
        "expected",
    ),
    [
        ((2, 2), 2.0, None, None, None, None, False, False, ftd.EXPECTED_TEST_FSS_2D_W2X2_T2),
        ((2, 2), 8.0, None, None, None, None, False, False, ftd.EXPECTED_TEST_FSS_2D_W2X2_T8),
        ((2, 2), 8.0, None, None, None, "uniform", False, False, ftd.EXPECTED_TEST_FSS_2D_W2X2_T8_UNIFORM_BENCHMARK),
        ((2, 2), 8.0, None, None, None, "random", False, False, ftd.EXPECTED_TEST_FSS_2D_W2X2_T8_RANDOM_BENCHMARK),
        ((5, 5), 8.0, None, None, None, "both", False, False, ftd.EXPECTED_TEST_FSS_2D_W5X5_T8_BOTH_BENCHMARK),
        ((5, 5), 2.0, None, None, None, None, False, False, ftd.EXPECTED_TEST_FSS_2D_W5X5_T2),
        ((5, 5), 2.0, None, None, None, None, True, False, ftd.EXPECTED_TEST_FSS_2D_W5X5_T2),
        ((5, 5), 2.0, None, None, None, None, True, True, ftd.EXPECTED_TEST_FSS_2D_W5X5_T2),
        ((5, 5), 8.0, None, None, None, None, False, False, ftd.EXPECTED_TEST_FSS_2D_W5X5_T8),
        ((5, 5), 8.0, None, None, "zero", None, False, False, ftd.EXPECTED_TEST_FSS_2D_W5X5_T8_ZERO_PADDED),
        (
            (5, 5),
            8.0,
            None,
            None,
            "reflective",
            "both",
            False,
            False,
            ftd.EXPECTED_TEST_FSS_2D_W5X5_T8_REFLECTIVE_PADDED,
        ),
        (
            (2, 2),
            5.0,
            ["time"],
            None,
            None,
            None,
            False,
            False,
            ftd.EXPECTED_TEST_FSS_2D_REDUCE_TIME,
        ),  # output_dims: 1 x lead_time
        (
            (2, 2),
            5.0,
            None,
            ["time"],
            None,
            None,
            False,
            False,
            ftd.EXPECTED_TEST_FSS_2D_PRESERVE_TIME,
        ),  # output_dims: 1 x time
        (
            (2, 2),
            5.0,
            None,
            ["time", "lead_time"],
            None,
            None,
            False,
            False,
            ftd.EXPECTED_TEST_FSS_2D_PRESERVE_ALL,
        ),  # output_dims: time x lead_time
    ],
)
def test_fss_2d(
    window_size, event_threshold, reduce_dims, preserve_dims, padding, benchmark, is_binary, check_boolean, expected
):
    """
    Tests various combinations of window_size/event_threshold/preserve_dims/reduce_dims/benchmark/padding

    Note: does not include error/assertion testing this will be done separately.
    """
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    if is_binary:
        da_fcst = np.greater(da_fcst, event_threshold)
        da_obs = np.greater(da_obs, event_threshold)
    res = fss_2d(
        da_fcst,
        da_obs,
        event_threshold=event_threshold,
        window_size=window_size,
        spatial_dims=["lat", "lon"],
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        padding=padding,
        benchmark=benchmark,
        is_input_binary=is_binary,
        check_boolean=check_boolean,
    )
    xr.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fss_2d_binary_bool_check():
    """Tests function raises suitable error when there is issue in input data and/or specified arguments"""
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    with pytest.raises(FieldTypeError):
        fss_2d(
            da_fcst,
            np.greater(da_obs, 0.5),
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
            is_input_binary=True,
            check_boolean=True,
        )
    with pytest.raises(FieldTypeError):
        fss_2d(
            np.greater(da_fcst, 0.5),
            da_obs,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
            is_input_binary=True,
            check_boolean=True,
        )
    with pytest.raises(ValueError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=0.5,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
            is_input_binary=False,
            check_boolean=True,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fss_2d_nonbinary_without_threshold():
    """Tests function raises an error when event_threshold is not specified for a non-binary field."""
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    with pytest.raises(ValueError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=None,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
            is_input_binary=False,
        )


def test_fss_2d_with_invalid_padding():
    """Tests function raises an error when an invalid padding is requested."""
    da_fcst = sd.continuous_forecast(large_size=False, lead_days=True)
    da_obs = sd.continuous_observations(large_size=False)
    with pytest.raises(ValueError):
        fss_2d(da_fcst, da_obs, event_threshold=0.5, window_size=(5, 5), spatial_dims=["lat", "lon"], padding="one")


def test_missing_spatial_dimensions():
    """
    Test for missing spatial dimensions in input data
    """

    # missing in forecast
    da_fcst = sd.continuous_observations(large_size=False)
    da_obs = sd.continuous_observations(large_size=False)
    da_fcst = da_fcst.rename({"lat": "bat"})
    with pytest.raises(DimensionError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=5.0,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
        )

    # missing in obs
    da_fcst = sd.continuous_observations(large_size=False)
    da_obs = sd.continuous_observations(large_size=False)
    da_obs = da_fcst.rename({"lat": "mat"})
    with pytest.raises(DimensionError):
        fss_2d(
            da_fcst,
            da_obs,
            event_threshold=5.0,
            window_size=(5, 5),
            spatial_dims=["lat", "lon"],
        )


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
        fss_2d_single_field(
            da_fcst[0].values,
            da_obs[0].values,
            event_threshold=5.0,
            window_size=(5, 5),
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
    assert res == xr.Dataset(data_vars={"FSS": 0.0, "FBS": 0.0, "FBS_ref": 0.0})


def test_zero_denom_fss_single_field():
    """
    Force denominator of the input field to be 0.0 to test against divide by zero
    """
    # half the meaning of life, in order to maintain some mystery
    seed = 21
    (obs, fcst) = ftd.generate((0, 0), (0, 0), seed=seed)
    res = fss_2d_single_field(fcst, obs, event_threshold=1.0, window_size=(5, 5))
    assert res == xr.Dataset(data_vars={"FSS": 0.0, "FBS": 0.0, "FBS_ref": 0.0})


def test_empty_score_aggregation():
    """
    Test for the scenario where the aggregation logic is being performed on an empty array.

    Theoretically this shouldn't be possible... but in practicality may happen.

    Note: `ufuncs` by design (which are typically implemented in C)
        should obfuscate this and should not be reachable, think for loop:
        `for (int i = 0; i < 0; i++) {...}` - this shouldn't reach any inner computations.
    """
    fss = _aggregate_fss_decomposed(np.empty(shape=(0, 0)))
    assert fss == 0.0
