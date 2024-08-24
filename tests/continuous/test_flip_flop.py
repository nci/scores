"""
This module contains tests for scores.continuous.flip_flop
"""

try:
    import dask
except:  # noqa: E722 allow bare except here # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # type: ignore  # pylint: disable=invalid-name  # pragma: no cover

import numpy as np
import pytest
import xarray as xr

from scores.continuous.flip_flop_impl import (
    _encompassing_sector_size_np,
    _flip_flop_index,
    encompassing_sector_size,
    flip_flop_index,
    flip_flop_index_proportion_exceeding,
)
from scores.utils import DimensionError
from tests.continuous import flip_flop_test_data as ntd


@pytest.mark.parametrize(
    ("data", "sampling_dim", "is_angular", "expected"),
    [
        # 0: 1-D, length=6
        (ntd.DATA_FFI_1D_6, "letter", False, xr.DataArray(12.5)),
        # 1: 1-D, with NaN, length=6
        (ntd.DATA_FFI_1D_6_WITH_NAN, "letter", False, xr.DataArray(np.nan)),
        # 2: 2-D, with NaN, length=6
        (
            ntd.DATA_FFI_2D_6,
            "letter",
            False,
            xr.DataArray([12.5, np.nan], coords=[("number", [1, 2])]),
        ),
        # 3: 1-D, length=3, no flip-flops
        (ntd.DATA_FFI_1D_6_ABC, "letter", False, xr.DataArray(0.0)),
        # 4: 1-D, length=3
        (ntd.DATA_FFI_1D_6_BCD, "letter", False, xr.DataArray(10.0)),
        # 5: 1-D, length=4
        (ntd.DATA_FFI_1D_6_ACEF, "letter", False, xr.DataArray(25.0)),
        # 6: 1-D, with NaN, length=3
        (ntd.DATA_FFI_1D_6_WITH_NAN_ACE, "letter", False, xr.DataArray(np.nan)),
        # 7: 1-D, length=3, with floating-points
        (
            xr.DataArray([np.pi, 0, 2 * np.pi], dims="pi"),
            "pi",
            False,
            xr.DataArray(np.pi),
        ),
        # 8: 3-D, length=3, with floating-points
        (ntd.DATA_FFI_3D, "letter", False, ntd.EXP_FFI_CASE8),
        # 9: 3-D, length=3, with floating-points, directional data
        (ntd.DATA_FFI_3D_DIR, "letter", True, ntd.EXP_FFI_CASE9),
    ],
)
def test__flip_flop_index(data, sampling_dim, is_angular, expected):
    """Tests that _flip_flop_index returns the correct object"""
    calculated = _flip_flop_index(data, sampling_dim, is_angular=is_angular)
    xr.testing.assert_allclose(calculated, expected)


def test__flip_flop_index_smoke_raises():
    """Smoke tests that _flip_flop_index raises the correct exception"""
    with pytest.raises(DimensionError) as ex:
        _flip_flop_index(ntd.DATA_FFI_1D_6, "banana")
    assert "not superset to the dimensions ['banana']" in str(ex.value)


@pytest.mark.parametrize(
    ("data", "sampling_dim", "is_angular", "expected"),
    [
        # 0: 1-D, selections={}
        (ntd.DATA_FFI_1D_6, "letter", False, ntd.EXP_FFI_SUB_CASE0),
        # 1: 1-D with NaN, selections={}
        (ntd.DATA_FFI_1D_6_WITH_NAN, "letter", False, ntd.EXP_FFI_SUB_CASE1),
        # 2: 2-D, selections={}
        (ntd.DATA_FFI_2D_6, "letter", False, ntd.EXP_FFI_SUB_CASE2),
        # 3: 1-D with float values, selections = {}
        (
            xr.DataArray([np.pi, 0, 2 * np.pi], dims="pi"),
            "pi",
            False,
            ntd.EXP_FFI_SUB_CASE3,
        ),
        # 4: 2-D, is_angular=True, selections={}
        (ntd.DATA_FFI_2D_6, "letter", True, ntd.EXP_FFI_SUB_CASE2),
    ],
)
def test_flip_flop_index_no_selections(data, sampling_dim, is_angular, expected):
    """
    Tests that flip_flop_index returns the correct result  when
    **selections are not supplied
    """
    calculated = flip_flop_index(data, sampling_dim, is_angular=is_angular)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    ("data", "sampling_dim", "is_angular", "selections", "expected"),
    [
        # 0: 1-D, selections pulling out in different order
        (
            ntd.DATA_FFI_1D_6,
            "letter",
            False,
            {"back_to_front": ["e", "a", "c"]},
            ntd.EXP_FFI_SUB_CASE4,
        ),
        # 1: 2-D, selections specified (length of samples = 3)
        (
            ntd.DATA_FFI_2D_6,
            "letter",
            False,
            ntd.DICT_FFI_SUB_CASE4,
            ntd.EXP_FFI_SUB_CASE5,
        ),
        # 2: 2-D, selections specified (length of samples = 4)
        (
            ntd.DATA_FFI_2D_6,
            "letter",
            False,
            {"banana": ["a", "c", "e", "f"]},
            ntd.EXP_FFI_SUB_CASE6,
        ),
        # 3: 3-D, selections specified (length of samples = 3)
        (
            ntd.DATA_FFI_3D,
            "letter",
            False,
            {"banana": ["a", "b", "c"]},
            ntd.EXP_FFI_SUB_CASE7,
        ),
        # 4: 1-D with floats coordinates
        (
            xr.DataArray([np.pi, 0, 2 * np.pi], dims=["pi"], coords={"pi": [3.14, np.pi, np.e]}),
            "pi",
            False,
            {"irrational": [np.e, np.pi, 3.14]},
            ntd.EXP_FFI_SUB_CASE8,
        ),
        # 5: 3-D
        # SPOT-CHECKED by DG
        (
            ntd.DATA_FFI_2X2X4,
            "int",
            False,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            ntd.EXP_FFI_SUB_CASE9,
        ),
        # 10: 2-D, selections specified with different length samples
        (
            ntd.DATA_FFI_2D_6,
            "letter",
            False,
            {"3letters": ["a", "c", "e"], "4letters": ["a", "c", "e", "f"]},
            ntd.EXP_FFI_SUB_CASE10,
        ),
        # 11: 2-D, selections specified with different length samples, angular data
        (
            ntd.DATA_FFI_2D_6,
            "letter",
            True,
            {"3letters": ["a", "c", "e"], "4letters": ["a", "c", "e", "f"]},
            ntd.EXP_FFI_SUB_CASE10,
        ),
    ],
)
def test_flip_flop_index(data, sampling_dim, is_angular, selections, expected):
    """
    Tests that flip_flop_index returns the correct result when
    **selections are supplied
    """
    calculated = flip_flop_index(data, sampling_dim, is_angular=is_angular, **selections)
    xr.testing.assert_allclose(calculated, expected)


def test_flip_flop_index_raises():
    """
    Test that flip_flop_index raises the correct exception when an invalid
    selection is supplied
    """
    with pytest.raises(KeyError) as ex:
        flip_flop_index(ntd.DATA_FFI_1D_6, "letter", zero=["a", "e", "g"])
    assert "for `selections` item {'zero': ['a', 'e', 'g']}, not all values" in str(ex.value)


@pytest.mark.parametrize(
    ("data", "dims", "skipna", "expected"),
    [
        (ntd.ESS, ["i", "j"], False, ntd.EXP_ESS_DIM_K),
        (ntd.ESS, ["i", "j"], True, ntd.EXP_ESS_DIM_K_SKIPNA),
        (ntd.ESS.sel(j=[1]), ["i", "j"], False, ntd.EXP_ESS_DIM_K_1J),
    ],
)
def test_encompassing_sector_size(data, dims, skipna, expected):
    """
    Tests encompassing_sector_size
    """
    calculated = encompassing_sector_size(data, dims, skipna=skipna)
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    ("angles", "expected"),
    ntd.ENC_SIZE_NP_TESTS_SKIPNA,
)
def test_encompassing_sector_size_np_1d_skipna(angles, expected):
    """test _encompassing_sector_size_np, 1d test cases with skipna=True"""
    angles = np.array(angles)
    result = _encompassing_sector_size_np(angles, skipna=True)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("angles", "expected"),
    ntd.ENC_SIZE_NP_TESTS_SKIPNA,
)
def test_encompassing_sector_size_np_1d_skipna_noorder(angles, expected):
    """
    test _encompassing_sector_size_np
    1d test cases, skipna=True, reverse test case order to make sure
    implementation not dependent on ordering of angles.
    """
    angles = np.array(list(reversed(angles)))
    result = _encompassing_sector_size_np(angles, skipna=True)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("angles", "expected"),
    ntd.ENC_SIZE_NP_TESTS_SKIPNA,
)
def test_encompassing_sector_size_np_1d_skipna_offsets(angles, expected):
    """
    test _encompassing_sector_size_np
    1d test cases, skipna=True, with various offsets applied, to test
    check implementation works where angles aren't all in (0, 360) range and
    even if angles span 0/360/720 boundaries.
    """
    angles = np.array(angles)
    for offset in range(-720, 720, 90):
        testcase = angles + offset
    result = _encompassing_sector_size_np(testcase, skipna=True)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("angles", "expected"),
    ntd.ENC_SIZE_NP_TESTS_NOSKIPNA,
)
def test_encompassing_sector_size_np_1d_noskipna(angles, expected):
    """test _encompassing_sector_size_np, 1d test cases, skipna=True"""
    angles = np.array(angles)
    result = _encompassing_sector_size_np(angles, skipna=False)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("angles", "axis_to_collapse", "skipna", "expected"),
    [
        (ntd.ENC_SIZE_3D_TEST_AXIS0, 0, True, ntd.ENC_SIZE_3D_ANSW_AXIS0_SKIPNA),
        (ntd.ENC_SIZE_3D_TEST_AXIS0, 0, False, ntd.ENC_SIZE_3D_ANSW_AXIS0_NOSKIPNA),
        (ntd.ENC_SIZE_3D_TEST_AXIS2, 2, True, ntd.ENC_SIZE_3D_ANSW_AXIS2_SKIPNA),
        (ntd.ENC_SIZE_3D_TEST_AXIS2, 2, False, ntd.ENC_SIZE_3D_ANSW_AXIS2_NOSKIPNA),
    ],
)
def test_encompassing_sector_size_np_3d(angles, axis_to_collapse, skipna, expected):
    """
    Test _encompassing_sector_size_np with 3d test cases.
    """
    result = _encompassing_sector_size_np(angles, axis_to_collapse=axis_to_collapse, skipna=skipna)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("data", "dims", "skipna"),
    [
        (ntd.ESS, ["i"], False),
        (ntd.ESS, ["j"], True),
        (ntd.ESS, ["foobar"], True),
    ],
)
def test_encompassing_sector_size_raises(data, dims, skipna):
    """
    Tests encompassing_sector_size for cases where:
    * Too many dimensions reduced for skipna true and false
    * Dimension not in existing dataset
    """
    with pytest.raises(DimensionError):
        encompassing_sector_size(data, dims, skipna=skipna)


@pytest.mark.parametrize(
    (
        "data",
        "sampling_dim",
        "thresholds",
        "is_angular",
        "selections",
        "reduce_dims",
        "preserve_dims",
        "expected",
    ),
    [
        # 0. 3-D, preserve_dims=None, reduce_dims=None
        (
            ntd.DATA_FFI_2X2X4,
            "int",
            [0, 1, 5],
            False,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            None,
            None,
            ntd.EXP_FFI_PE_NONE,
        ),
        # 1. 3-D, preserve_dims=['char']
        (
            ntd.DATA_FFI_2X2X4,
            "int",
            [0, 1, 5],
            False,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            None,
            ["char"],
            ntd.EXP_FFI_PE_CHAR,
        ),
        # 2. 3-D, preserve_dims=['char', 'bool']
        (
            ntd.DATA_FFI_2X2X4,
            "int",
            [0, 1, 5],
            False,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            None,
            ["char", "bool"],
            ntd.EXP_FFI_PE_CHARBOOL,
        ),
        # 3. 3-D, preserve_dims=['char', 'bool'], angular data
        (
            ntd.DATA_FFI_2X2X4_DIR,
            "int",
            [0, 50, 100],
            True,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            None,
            ["char", "bool"],
            ntd.EXP_FFI_PE_CHARBOOL_DIR,
        ),
        # 4. 3-D, reduce_dims=['bool']
        (
            ntd.DATA_FFI_2X2X4,
            "int",
            [0, 1, 5],
            False,
            {"one": [1, 2, 3], "two": [2, 3, 4]},
            ["bool"],
            None,
            ntd.EXP_FFI_PE_CHAR,
        ),
    ],
)
def test_flip_flop_index_proportion_exceeding(
    data, sampling_dim, thresholds, is_angular, selections, reduce_dims, preserve_dims, expected
):
    """
    Tests that flip_flop_index_proportion_exceeding returns the correct object
    """
    calculated = flip_flop_index_proportion_exceeding(
        data,
        sampling_dim,
        thresholds,
        is_angular=is_angular,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
        **selections,
    )
    xr.testing.assert_allclose(calculated, expected)


@pytest.mark.parametrize(
    "dims_kwargs",
    [{"preserve_dims": ["letter", "banana"]}, {"reduce_dims": ["letter", "banana"]}],
    ids=["preserve_dims", "reduce_dims"],
)
def test_flip_flop_index_proportion_exceeding_raises(dims_kwargs):
    """
    Smoke tests that flip_flop_index_proportion_exceeding raises the correct
    exception when `sampling_dim` is in `dims`
    """
    with pytest.raises(DimensionError) as ex:
        flip_flop_index_proportion_exceeding(
            xr.DataArray([[1]], dims=["letter", "banana"]),
            "letter",
            [10, 20],
            **dims_kwargs,
        )
    assert "`sampling_dim`: 'letter' must not be in dimensions to " in str(ex.value)


def test_flip_flop_index_is_dask_compatible():
    """
    Test that flip flop works with dask.
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    dask_array = ntd.DATA_FFI_1D_6.chunk()
    result = flip_flop_index(dask_array, "letter")
    assert isinstance(result.data, dask.array.Array)
    result = result.compute()
    xr.testing.assert_allclose(result, ntd.EXP_FFI_SUB_CASE0)
    assert isinstance(result.data, np.ndarray | np.generic)


def test_flip_flop_index_proportion_exceeding_is_dask_compatible():
    """
    Test that flip flop proportion exceeding works with dask.
    """

    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover

    dask_array = ntd.DATA_FFI_2X2X4.chunk()
    result = flip_flop_index_proportion_exceeding(dask_array, "int", [0, 1, 5], **{"one": [1, 2, 3], "two": [2, 3, 4]})
    assert isinstance(result.data_vars["one"].data, dask.array.Array)
    result = result.compute()
    xr.testing.assert_allclose(result, ntd.EXP_FFI_PE_NONE)
    assert isinstance(result.one.data, np.ndarray)
