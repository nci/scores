"""
This module contains unit tests for scores.stats.tests.block_bootstrap
"""

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

import scores.processing.block_bootstrap_impl as block_bootstrap_module
from scores.processing.block_bootstrap_impl import (
    _block_bootstrap,
    _bootstrap,
    _expand_n_nested_random_indices,
    _get_blocked_random_indices,
    _n_nested_blocked_random_indices,
    block_bootstrap,
)
from scores.utils import dask_available

HAS_DASK = dask_available()

if HAS_DASK:
    import dask.array as da
else:
    da = None


@pytest.mark.parametrize(
    "shape, block_axis, block_size, prev_block_sizes, circular, expected_shape",
    [
        # Test case 1: block_size = 1, no previous block sizes
        ([10, 20, 30], 1, 1, [], True, (10, 20, 30)),
        # Test case 2: block_size > 1, no previous block sizes
        ([10, 20, 30], 0, 3, [], False, (10, 20, 30)),
        # Test case 3: block_size = 1, with previous block sizes
        ([10, 20, 30], 1, 1, [2, 1], True, (10, 20, 30)),
        # Test case 4: block_size > 1, with previous block sizes
        ([10, 20, 30], 0, 3, [2, 1], False, (10, 20, 30)),
        # Test case 5: block_size == length
        ([10, 20, 30], 0, 10, [2, 1], False, (10, 20, 30)),
        # Test case 6: block_size != length and circular
        ([10, 20, 30], 0, 3, [2, 1], True, (10, 20, 30)),
    ],
)
def test__get_blocked_random_indices(shape, block_axis, block_size, prev_block_sizes, circular, expected_shape):
    "Test that _get_blocked_random_indices works as expected"
    indices = _get_blocked_random_indices(shape, block_axis, block_size, prev_block_sizes, circular)
    assert indices.shape == expected_shape


@pytest.mark.parametrize(
    "sizes, n_iteration, circular, expected_shapes",
    [
        (OrderedDict([("dim1", (10, 2)), ("dim2", (5, 2))]), 3, True, [(10, 3), (10, 5, 3)]),
        (OrderedDict([("dim1", (10, 2)), ("dim2", (5, 2))]), 3, False, [(10, 3), (10, 5, 3)]),
    ],
)
def test__n_nested_blocked_random_indices(sizes, n_iteration, circular, expected_shapes):
    """Test that _n_nested_blocked_random_indices returns indices with expected shape"""
    indices = _n_nested_blocked_random_indices(sizes, n_iteration, circular)
    assert len(indices) == len(sizes)
    for (dim, _), expected_shape in zip(sizes.items(), expected_shapes):
        assert indices[dim].shape == expected_shape


@pytest.mark.parametrize(
    "indices, expected_shapes",
    [
        ([np.random.randint(0, 10, (10,)), np.random.randint(0, 5, (10, 3))], [(10,), (10, 3)]),
        (
            [np.random.randint(0, 10, (10,)), np.random.randint(0, 5, (10, 3)), np.random.randint(0, 3, (10, 3, 2))],
            [(10, 1), (10, 3), (10, 3, 2)],
        ),
    ],
)
def test__expand_n_nested_random_indices(indices, expected_shapes):
    """Test _expand_n_nested_random_indices returns indices with expected shape"""
    expanded_indices = _expand_n_nested_random_indices(indices)
    assert len(expanded_indices) == len(indices) + 1
    # Check this function returns `...`
    assert expanded_indices[0] == ...
    # Check dimensions of each array are correctly expanded
    for ind, expected_shape in zip(expanded_indices[1:], expected_shapes):
        assert ind.shape == expected_shape


@pytest.mark.parametrize(
    "objects, blocks, n_iteration, exclude_dims, circular, expected_shape",
    [
        (
            [xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
        ),
        (
            [xr.DataArray(np.random.rand(10, 5, 7), dims=["dim1", "dim2", "dim3"])],
            {"dim2": 2, "dim3": 1},
            3,
            [["dim1"]],
            True,
            (10, 5, 7, 3),
        ),
        # Test excluding 2 dims
        (
            [xr.DataArray(np.random.rand(10, 5, 7, 5), dims=["dim1", "dim2", "dim3", "dim4"])],
            {"dim2": 2, "dim3": 1},
            3,
            [["dim1", "dim4"]],
            True,
            (10, 5, 5, 7, 3),
        ),
        (
            [
                xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
                xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            False,
            (10, 5, 3),
        ),
        (
            [
                xr.Dataset({"var1": xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])}),
                xr.Dataset({"var2": xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])}),
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
        ),
    ],
)
def test__block_bootstrap(objects, blocks, n_iteration, exclude_dims, circular, expected_shape):
    """Test _block_bootstrap works as expected"""
    result = _block_bootstrap(
        objects, blocks=blocks, n_iteration=n_iteration, exclude_dims=exclude_dims, circular=circular
    )
    for res in result:
        if isinstance(res, xr.Dataset):
            for var in res.data_vars:
                assert res[var].shape == expected_shape
        else:
            assert res.shape == expected_shape


def test__bootstrap_tuple_return():
    """Test for returning a tuple from _bootstrap"""
    arrays = [np.random.rand(10, 5), np.random.rand(10, 5)]
    indices = [np.random.randint(0, 10, size=10), np.random.randint(0, 10, size=10)]
    result = _bootstrap(*arrays, indices=indices)
    assert isinstance(result, tuple)
    assert result[0].shape == result[1].shape == (10, 5)


@pytest.mark.parametrize(
    "objects, blocks, n_iteration, exclude_dims, circular, expected_exception, match",
    [
        (
            [
                xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
                xr.DataArray(np.random.rand(11, 5), dims=["dim1", "dim2"]),
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            ValueError,
            "Block dimension dim1 is not the same size on all input arrays",
        ),
        (
            [xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])],
            {"dim1": 2, "dim2": 2},
            3,
            "invalid",
            True,
            ValueError,
            "exclude_dims should be a list of lists",
        ),
        (
            [xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])],
            {"dim1": 2, "dim2": 2},
            3,
            [["lead_day"], ["lead_day"], []],
            True,
            ValueError,
            "exclude_dims should be a list of the same length as the number of arrays in array_list",
        ),
        (
            [xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])],
            {"dim1": 2, "dim3": 2},
            3,
            None,
            True,
            ValueError,
            "At least one input array must contain all dimensions in blocks.keys()",
        ),
    ],
)
def test__block_bootstrap_exceptions(objects, blocks, n_iteration, exclude_dims, circular, expected_exception, match):
    """Test _block_bootstrap correctly raises errors"""
    with pytest.raises(expected_exception=expected_exception, match=match):
        _block_bootstrap(objects, blocks=blocks, n_iteration=n_iteration, exclude_dims=exclude_dims, circular=circular)


@pytest.mark.parametrize(
    "objects, blocks, n_iteration, exclude_dims, circular, expected_shape, expected_type",
    [
        # Single array bootstrap
        (
            xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
            xr.DataArray,
        ),
        # Multiple arrays bootstrap. Also test it works with NaNs
        (
            [
                xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
                np.nan * xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
            tuple,
        ),
        # Exclude dimensions
        (
            [xr.DataArray(np.random.rand(10, 5, 7), dims=["dim1", "dim2", "dim3"])],
            {"dim2": 2, "dim3": 2},
            3,
            [["dim1"]],
            True,
            (10, 5, 7, 3),
            xr.DataArray,
        ),
        # Dataset bootstrap. Also test it works with NaNs
        (
            [
                xr.Dataset(
                    {
                        "var1": xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
                        "var2": np.nan * xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]),
                    }
                )
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
            xr.Dataset,
        ),
    ],
)
def test_block_bootstrap(objects, blocks, n_iteration, exclude_dims, circular, expected_shape, expected_type):
    """Test block_bootstrap works as expected"""
    result = block_bootstrap(
        objects, blocks=blocks, n_iteration=n_iteration, exclude_dims=exclude_dims, circular=circular
    )
    if expected_type == tuple:
        assert isinstance(result, tuple)
        assert all(isinstance(res, xr.DataArray) for res in result)
        for res in result:
            assert res.shape == expected_shape
    elif expected_type == xr.Dataset:
        assert isinstance(result, xr.Dataset)
        for var in result.data_vars:
            assert result[var].shape == expected_shape
    else:
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape


dask_bb_scenarios = [[None, None, None, None, None, None]]
if HAS_DASK:
    dask_bb_scenarios = [
        (
            [xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"]).chunk()],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 5, 3),
        ),
        # Dask arrays to meet block_size < 1
        (
            [xr.DataArray(da.random.random((100, 100, 30), chunks=dict({"dim1": -1})), dims=["dim1", "dim2", "dim3"])],
            {"dim1": 2, "dim2": 2},
            2,
            None,
            True,
            (30, 100, 100, 2),
        ),
        # Dask arrays for a case with leftover != 0
        (
            [xr.DataArray(da.random.random((100, 100, 10), chunks=dict({"dim1": -1})), dims=["dim1", "dim2", "dim3"])],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (10, 100, 100, 3),
        ),
        # Dataset with dask arrays
        (
            [
                xr.Dataset(
                    {
                        "var1": xr.DataArray(
                            da.random.random((100, 100, 30), chunks=dict({"dim1": -1})), dims=["dim1", "dim2", "dim3"]
                        ),
                        "var2": xr.DataArray(
                            da.random.random((100, 100, 30), chunks=dict({"dim1": -1})), dims=["dim1", "dim2", "dim3"]
                        ),
                    }
                )
            ],
            {"dim1": 2, "dim2": 2},
            3,
            None,
            True,
            (30, 100, 100, 3),
        ),
    ]


@pytest.mark.parametrize("objects, blocks, n_iteration, exclude_dims, circular, expected_shape", dask_bb_scenarios)
def test_block_bootstrap_dask(monkeypatch, objects, blocks, n_iteration, exclude_dims, circular, expected_shape):
    """Test block_bootstrap can work with dask arrays"""
    if not HAS_DASK:  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover
    # We mock MAX_BATCH_SIZE so that we don't need to pass in large arrays which
    # slow down the tests
    monkeypatch.setattr(block_bootstrap_module, "MAX_BATCH_SIZE_MB", 2)
    result = block_bootstrap(
        objects, blocks=blocks, n_iteration=n_iteration, exclude_dims=exclude_dims, circular=circular
    )
    if isinstance(result, xr.DataArray):
        assert isinstance(result.data, da.Array)
        result = result.compute()
        assert result.shape == expected_shape
        assert isinstance(result.data, np.ndarray)
    else:
        for var in result.data_vars:
            assert isinstance(result[var].data, da.Array)
        result = result.compute()
        for var in result.data_vars:
            assert isinstance(result[var].data, np.ndarray)
