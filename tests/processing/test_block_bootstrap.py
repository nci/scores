import itertools

import dask.array
import numpy as np
import pytest
import xarray as xr

from xbootstrap.core import (
    _expand_n_nested_random_indices,
    _n_nested_blocked_random_indices,
    block_bootstrap,
)


@pytest.mark.parametrize("shape", [(1,), (2, 50, 6)])
@pytest.mark.parametrize("n_iteration", [1, 5])
@pytest.mark.parametrize("circular", [True, False])
def test_random_indices_shape(shape, n_iteration, circular):
    """
    Test that _n_nested_blocked_random_indices and
    _expand_n_nested_random_indices produce outputs with the right shape
    """
    blocks = (1,) * len(shape)
    axes = [f"a{i}" for i in range(len(shape))]
    data = np.zeros(shape, dtype="<U16")
    for i in itertools.product(*[range(i) for i in shape]):
        data[i] = "".join([f"{axes[j]}{i[j]}" for j in range(len(i))])
    nested_indexes = _n_nested_blocked_random_indices(
        dict(zip(axes, zip(shape, blocks))), n_iteration, circular
    )
    indexes = _expand_n_nested_random_indices(
        [nested_indexes[k] for k in axes],
    )
    assert data[indexes].shape == shape + (n_iteration,)


@pytest.mark.parametrize("shape", [(9, 8, 7, 6, 5), (5, 5, 5, 5, 5)])
@pytest.mark.parametrize("blocks", [(1, 2, 3, 4, 5), (5, 5, 5, 5, 5)])
@pytest.mark.parametrize("n_iteration", [1, 10, 20])
@pytest.mark.parametrize("circular", [True, False])
def test_bootstrap_nesting(shape, blocks, n_iteration, circular):
    """Test that block bootstrap nests correctly"""

    def block_iterate(seq, size):
        """Iterate in blocks"""
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    def check_and_get_block_indices(block, shape, axes):
        """
        Return the block indices for each axes, checking that they are
        consecutive and the same for each nested level
        """

        def __get_indices(s, ax):
            return s.split(ax)[1][0]

        _get_indices = np.vectorize(__get_indices)

        indices = []
        for ax, i in zip(axes, range(block.ndim, 0, -1)):
            nested_block = block[(...,) + (0,) * (i - 1)]
            nested_block_indices = (
                _get_indices(nested_block, ax)
                .reshape(-1, nested_block.shape[-1])
                .astype(int)
            )

            # Assert all nested blocks within a block are the same
            assert (nested_block_indices[0] == nested_block_indices).all()

            # Assert block is consecutive
            diff = np.diff(nested_block_indices[0])
            assert np.logical_or(diff == 1, diff == -shape[-i] + 1).all()

            indices.append(nested_block_indices[0])
        return indices

    # Generate some test data with data that makes it location clear
    axes = ["a", "b", "c", "d", "e"]
    data = np.zeros(shape, dtype="<U16")
    for i in itertools.product(*[range(i) for i in shape]):
        data[i] = "".join([f"{axes[j]}{i[j]}" for j in range(len(i))])

    # Randomly resample the test data
    nested_indexes = _n_nested_blocked_random_indices(
        dict(zip(axes, zip(shape, blocks))), n_iteration, circular
    )
    indexes = _expand_n_nested_random_indices(
        [nested_indexes[k] for k in axes],
    )
    bootstrapped_data = data[indexes]

    # Check that blocks have correct nesting
    for pi in range(n_iteration):
        for ai in block_iterate(range(shape[0]), blocks[0]):
            inner_b_block_indices = []
            for bi in block_iterate(range(shape[1]), blocks[1]):
                inner_c_block_indices = []
                for ci in block_iterate(range(shape[2]), blocks[2]):
                    inner_d_block_indices = []
                    for di in block_iterate(range(shape[3]), blocks[3]):
                        inner_e_block_indices = []
                        for ei in block_iterate(range(shape[4]), blocks[4]):
                            block = bootstrapped_data[
                                ai.start : ai.stop,
                                bi.start : bi.stop,
                                ci.start : ci.stop,
                                di.start : di.stop,
                                ei.start : ei.stop,
                                pi,
                            ]

                            indices = check_and_get_block_indices(
                                block,
                                shape,
                                axes,
                            )
                            inner_b_block_indices.append(indices[:-4])
                            inner_c_block_indices.append(indices[:-3])
                            inner_d_block_indices.append(indices[:-2])
                            inner_e_block_indices.append(indices[:-1])

                        # Assert that there is no randomization within an outer
                        # block
                        assert all(
                            (x == y).all()
                            for b in inner_e_block_indices[1:]
                            for x, y in zip(inner_e_block_indices[0], b)
                        )
                    assert all(
                        (x == y).all()
                        for b in inner_d_block_indices[1:]
                        for x, y in zip(inner_d_block_indices[0], b)
                    )
                assert all(
                    (x == y).all()
                    for b in inner_c_block_indices[1:]
                    for x, y in zip(inner_c_block_indices[0], b)
                )
            assert all(
                (x == y).all()
                for b in inner_b_block_indices[1:]
                for x, y in zip(inner_b_block_indices[0], b)
            )


@pytest.mark.parametrize("block", [1, 2, 100])
@pytest.mark.parametrize("n_iteration", [1, 5])
@pytest.mark.parametrize("circular", [True, False])
def test_block_bootstrap_values(block, n_iteration, circular):
    """
    Test that block bootstrapping produces different values along the
    sample and iteration dimensions
    """
    size = 100
    data = np.array([f"a{j}" for j in range(size)])
    nested_indexes = _n_nested_blocked_random_indices(
        dict(a=(100, block)), n_iteration, circular
    )
    indexes = _expand_n_nested_random_indices([nested_indexes["a"]])
    bootstrapped_data = data[indexes]
    assert not (bootstrapped_data[0, :] == bootstrapped_data).all()
    if (block == size) | (n_iteration == 1):
        assert (np.expand_dims(bootstrapped_data[:, 0], -1) == bootstrapped_data).all()
    else:
        assert not (
            np.expand_dims(bootstrapped_data[:, 0], -1) == bootstrapped_data
        ).all()


@pytest.mark.parametrize("block", [10, 3, 1])
@pytest.mark.parametrize("n_iteration", [1, 5])
def test_block_bootstrap_multi_arg(block, n_iteration):
    """Test block bootstrapping with multiple arguments"""
    shape = (10, 5)
    axes = ["a", "b"]
    data = np.zeros(shape, dtype="<U16")
    for i in itertools.product(*[range(i) for i in shape]):
        data[i] = "".join([f"{axes[j]}{i[j]}" for j in range(len(i))])
    x = xr.DataArray(
        data,
        coords={f"d{i}": range(shape[i]) for i in range(len(shape))},
    )
    y = xr.DataArray(data[:, 0], coords={"d0": range(shape[0])})
    x_bs, y_bs = block_bootstrap(x, y, blocks={"d0": block}, n_iteration=n_iteration)
    assert (
        x_bs.isel({f"d{i}": 0 for i in range(1, len(shape))}).values == y_bs.values
    ).all()


@pytest.mark.parametrize(
    "data",
    [np.zeros(shape=(10, 5)), dask.array.zeros((240, 240, 240), chunks=(-1, -1, -1))],
)
@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("n_iteration", [1, 2])
def test_block_bootstrap_output_type(data, blocks, n_iteration):
    """Test that output type is correct"""
    shape = data.shape
    x = xr.DataArray(
        data,
        coords={f"d{i}": range(shape[i]) for i in range(len(shape))},
    )
    out = block_bootstrap(x, blocks={"d0": blocks}, n_iteration=n_iteration)
    assert isinstance(out, xr.DataArray)
    out = block_bootstrap(x, x, blocks={"d0": blocks}, n_iteration=n_iteration)
    assert isinstance(out, tuple)
