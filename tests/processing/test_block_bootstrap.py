"""
This module contains unit tests for scores.stats.tests.block_bootstrap
"""

try:
    import dask
    import dask.array as da
except:  # noqa: E722 allow bare except here  # pylint: disable=bare-except  # pragma: no cover
    dask = "Unavailable"  # pylint: disable=invalid-name  # pragma: no cover

from collections import OrderedDict
from copy import deepcopy

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
    indices = _get_blocked_random_indices(
        shape, block_axis, block_size, prev_block_sizes, circular, draw_integers=np.random.default_rng(100).integers
    )
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
    indices = _n_nested_blocked_random_indices(
        sizes, n_iteration, circular, draw_integers=np.random.default_rng(100).integers
    )
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
        objects,
        blocks=blocks,
        n_iteration=n_iteration,
        exclude_dims=exclude_dims,
        circular=circular,
        draw_integers=np.random.default_rng(100).integers,
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
        _block_bootstrap(
            objects,
            blocks=blocks,
            n_iteration=n_iteration,
            exclude_dims=exclude_dims,
            circular=circular,
            draw_integers=np.random.default_rng(100).integers,
        )


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
    if expected_type is tuple:
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
if not dask == "Unavailable":
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
            [xr.DataArray(da.random.random((100, 100, 30), chunks=dict(dim1=-1)), dims=["dim1", "dim2", "dim3"])],
            {"dim1": 2, "dim2": 2},
            2,
            None,
            True,
            (30, 100, 100, 2),
        ),
        # Dask arrays for a case with leftover != 0
        (
            [xr.DataArray(da.random.random((100, 100, 10), chunks=dict(dim1=-1)), dims=["dim1", "dim2", "dim3"])],
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
                            da.random.random((100, 100, 30), chunks=dict(dim1=-1)), dims=["dim1", "dim2", "dim3"]
                        ),
                        "var2": xr.DataArray(
                            da.random.random((100, 100, 30), chunks=dict(dim1=-1)), dims=["dim1", "dim2", "dim3"]
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
    if dask == "Unavailable":  # pragma: no cover
        pytest.skip("Dask unavailable, could not run test")  # pragma: no cover
    # We mock MAX_BATCH_SIZE so that we don't need to pass in large arrays which
    # slow down the tests
    monkeypatch.setattr(block_bootstrap_module, "MAX_BATCH_SIZE_MB", 2)
    result = block_bootstrap(
        objects, blocks=blocks, n_iteration=n_iteration, exclude_dims=exclude_dims, circular=circular
    )
    if isinstance(result, xr.DataArray):
        assert isinstance(result.data, dask.array.Array)
        result = result.compute()
        assert result.shape == expected_shape
        assert isinstance(result.data, np.ndarray)
    else:
        for var in result.data_vars:
            assert isinstance(result[var].data, dask.array.Array)
        result = result.compute()
        for var in result.data_vars:
            assert isinstance(result[var].data, np.ndarray)


def test_block_bootstrap_legacy_case():
    """Tests that the block_bootstrap function is backwards compatible."""
    data = xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])
    blocks = {"dim1": 2, "dim2": 2}
    np.random.seed(42)
    first = block_bootstrap(data, blocks=blocks, n_iteration=5)

    np.random.seed(42)
    second = block_bootstrap(data, blocks=blocks, n_iteration=5, rng=None)

    xr.testing.assert_identical(first, second)


def test_block_bootstrap_works_with_rng():
    """Tests that the block_bootstrap function works correctly when a random number generator is provided."""
    data = xr.DataArray(np.random.rand(10, 5), dims=["dim1", "dim2"])
    blocks = {"dim1": 2, "dim2": 2}
    np.random.seed(1)
    first = block_bootstrap(data, blocks=blocks, n_iteration=5, rng=100)

    np.random.seed(2)
    second = block_bootstrap(data, blocks=blocks, n_iteration=5, rng=100)

    xr.testing.assert_identical(first, second)


def test_block_bootstrap_legacy_samples():
    """Keep the original global-seed samples, rather than just their repeatability."""
    data = xr.DataArray(np.arange(24).reshape(6, 4), dims=["time", "station"])
    np.random.seed(42)
    result = block_bootstrap(data, blocks={"time": 3, "station": 2}, n_iteration=2)
    expected = np.array(
        [
            [[12, 9], [13, 10], [14, 10], [15, 11]],
            [[16, 13], [17, 14], [18, 14], [19, 15]],
            [[20, 17], [21, 18], [22, 18], [23, 19]],
            [[18, 18], [19, 19], [18, 19], [19, 16]],
            [[22, 22], [23, 23], [22, 23], [23, 20]],
            [[2, 2], [3, 3], [2, 3], [3, 0]],
        ]
    )
    np.testing.assert_array_equal(result.values, expected)


def test_block_bootstrap_supplied_generator():
    """Continue the supplied stream without changing global state or paired sampling."""
    data = xr.DataArray(np.arange(24).reshape(6, 4), dims=["time", "station"])
    kwargs = {"blocks": {"time": 2, "station": 2}, "n_iteration": 3}
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)
    global_state = np.random.get_state()

    for _ in range(2):
        previous_state = deepcopy(rng.bit_generator.state)
        actual, paired = block_bootstrap([data, data + 10], rng=rng, **kwargs)
        expected = block_bootstrap(data, rng=reference_rng, **kwargs)
        xr.testing.assert_identical(actual, expected)
        np.testing.assert_array_equal(paired.values, actual.values + 10)
        assert rng.bit_generator.state != previous_state

    for before, after in zip(global_state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("circular", [False, True])
def test_block_bootstrap_nested_blocks(circular):
    """Preserve consecutive blocks and shared inner sampling within outer blocks."""
    data = xr.DataArray(100 * np.arange(7)[:, None] + np.arange(5), dims=["time", "station"])
    result = block_bootstrap(data, blocks={"time": 3, "station": 2}, n_iteration=5, circular=circular, rng=42)
    times = result.values // 100
    stations = result.values % 100
    for start in range(0, 7, 3):
        time_block = times[start : start + 3]
        differences = np.diff(time_block, axis=0)
        np.testing.assert_array_equal(differences % 7 if circular else differences, 1)
        for row in range(start + 1, min(start + 3, 7)):
            np.testing.assert_array_equal(stations[row], stations[start])
    for start in range(0, 5, 2):
        differences = np.diff(stations[:, start : start + 2], axis=1)
        np.testing.assert_array_equal(differences % 5 if circular else differences, 1)


@pytest.mark.skipif(dask == "Unavailable", reason="Dask unavailable")
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("rng_kind", ["seed", "generator"])
@pytest.mark.parametrize(
    "blocks",
    [{"time": 1}, {"time": 3}, {"time": 1, "station": 1}, {"time": 3, "station": 2}, {"time": 7, "station": 5}],
)
def test_block_bootstrap_dask_matches_numpy(monkeypatch, batch_size, circular, rng_kind, blocks):
    """Chunking and leftover iteration batches must not change modern samples."""
    data = xr.DataArray(np.arange(105).reshape(7, 5, 3), dims=["time", "station", "member"])
    # Bootstrap axes start split; the largest output chunk retains one member.
    chunked = data.chunk({"time": 2, "station": 2, "member": 1})
    chunk_bytes = 7 * (5 if "station" in blocks else 2) * data.dtype.itemsize
    monkeypatch.setattr(block_bootstrap_module, "MAX_BATCH_SIZE_MB", batch_size * chunk_bytes / 1024**2)
    kwargs = {"blocks": blocks, "n_iteration": 5, "circular": circular}
    expected = block_bootstrap(data, rng=42 if rng_kind == "seed" else np.random.default_rng(42), **kwargs)
    actual = block_bootstrap(chunked, rng=42 if rng_kind == "seed" else np.random.default_rng(42), **kwargs)

    assert isinstance(actual.data, da.Array)
    assert max(actual.chunksizes["iteration"]) == batch_size
    assert actual.chunksizes["member"] == (1, 1, 1)
    xr.testing.assert_identical(actual.compute(scheduler="threads"), expected)


@pytest.mark.skipif(dask == "Unavailable", reason="Dask unavailable")
@pytest.mark.parametrize("dask_first", [False, True])
def test_block_bootstrap_mixed_inputs_and_exclusions(monkeypatch, dask_first):
    """Detect lazy inputs anywhere and avoid mutating excluded dimensions between batches."""
    data = xr.DataArray(np.arange(35).reshape(7, 5), dims=["time", "station"])
    arrays = [data, (data + 10).chunk({"time": 2, "station": 2})]
    exclusions = [["station"], []]
    if dask_first:
        arrays.reverse()
        exclusions.reverse()
    original_dims = [obj.dims for obj in arrays]
    kwargs = {"blocks": {"time": 3, "station": 2}, "n_iteration": 5, "exclude_dims": exclusions, "rng": 42}
    expected = block_bootstrap([obj.compute() for obj in arrays], **kwargs)
    monkeypatch.setattr(block_bootstrap_module, "MAX_BATCH_SIZE_MB", 2 * data.nbytes / 1024**2)
    actual = block_bootstrap(arrays, **kwargs)

    assert [obj.dims for obj in arrays] == original_dims
    for result, reference in zip(actual, expected):
        xr.testing.assert_identical(result.compute(), reference)
    lazy_result = actual[0 if dask_first else 1]
    assert lazy_result.chunksizes["iteration"] == (2, 2, 1)


@pytest.mark.skipif(dask == "Unavailable", reason="Dask unavailable")
def test_block_bootstrap_mixed_dataset(monkeypatch):
    """Mixed dtypes, eager variables and differing chunk layouts remain supported."""
    integers = xr.DataArray(np.arange(35, dtype=np.int16).reshape(7, 5), dims=["time", "station"])
    data = xr.Dataset(
        {
            "integer": integers.chunk({"time": 2, "station": 2}),
            "float": (integers / 10).chunk({"time": 3, "station": 1}),
            "eager": integers + 10,
        }
    )
    kwargs = {"blocks": {"time": 3, "station": 2}, "n_iteration": 5, "rng": 42}
    expected = block_bootstrap(data.compute(), **kwargs)
    monkeypatch.setattr(block_bootstrap_module, "MAX_BATCH_SIZE_MB", 2 * data["float"].nbytes / 1024**2)
    result = block_bootstrap(data, **kwargs)

    for name in data.data_vars:
        assert result[name].dtype == data[name].dtype
    assert result["float"].chunksizes["iteration"] == (2, 2, 1)
    xr.testing.assert_identical(result.compute(), expected)


@pytest.mark.skipif(dask == "Unavailable", reason="Dask unavailable")
def test_block_bootstrap_dask_computation_does_not_draw_again():
    """Random sampling happens before task execution, and graphs can be recomputed."""
    data = xr.DataArray(np.arange(35).reshape(7, 5), dims=["time", "station"]).chunk({"time": 2, "station": 2})
    rng = np.random.default_rng(42)
    result = block_bootstrap(data, blocks={"time": 3, "station": 2}, n_iteration=5, rng=rng)
    state = deepcopy(rng.bit_generator.state)
    first = result.compute(scheduler="threads")
    second = result.compute(scheduler="single-threaded")
    xr.testing.assert_identical(first, second)
    assert rng.bit_generator.state == state
