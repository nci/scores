# pylint: disable=all

import pprint

import numpy as np
import xarray as xr

from scores.emerging.block_bootstrap.axis_info import AxisInfo, make_axis_info
from scores.emerging.block_bootstrap.block_bootstrap import (
    _construct_block_bootstrap_array,
)
from scores.emerging.block_bootstrap.helpers import (
    partial_linear_order_by_ref,
    reorder_dims,
)
from scores.emerging.block_bootstrap.methods import FitBlocksMethod


def _test_numpy_blk_bootstrap_single_iter():
    # generate test data
    axis_len = [13, 10, 8, 7]
    block_sizes = [4, 3, 2, 3]
    method = FitBlocksMethod.PARTIAL
    rng = np.random.default_rng(seed=42)
    # random dataset with integers so its easy to visualize
    input_arr = rng.integers(low=0, high=10, size=axis_len)

    print("--- axis info ---")
    axis_info = make_axis_info(input_arr, block_sizes, method)
    pprint.pp(axis_info)

    print("\n--- input array ---")
    pprint.pp(input_arr.shape)
    res = np.histogram(input_arr, bins=5)
    pprint.pp(res)

    print("\nBOOTSTRAPPING...")
    (output_arr, block_sample_idx) = _construct_block_bootstrap_array(
        input_arr,
        block_sizes,
        cyclic=True,
        fit_blocks_method=method,
    )

    print("\n--- sample axis block indices ---")
    pprint.pp(block_sample_idx)

    print("\n--- output array ---")
    pprint.pp(output_arr.shape)
    # print(output_arr)
    res = np.histogram(output_arr, bins=5)
    pprint.pp(res)


def _test_order_dims():
    arr_test = np.random.rand(10, 10, 12, 7, 5)
    da = xr.DataArray(
        data=arr_test,
        dims=["x", "y", "time", "lead_time", "height"],
    )

    print("\n--- partial order ---")
    da_ord = reorder_dims(da, ["y", "height", "x"])
    print(f"shape_in: {da.shape}, shape_out: {da_ord.shape}")
    print(f"dims_in: {da.dims}, dims_out: {da_ord.dims}")
    try:
        reorder_dims(da, ["y", "height", "x"], False)
    except ValueError as e:
        print(f"\n--- no auto ordering for unspecified dims ---")
        print(f"Caught expected value error: {e}")

    try:
        reorder_dims(da, ["x", "height", "alpha", "y"])
    except ValueError as e:
        print(f"\n--- gap in ordering, could not find alpha ---")
        print(f"Caught expected value error: {e}")


if __name__ == "__main__":
    _test_order_dims()
    _test_numpy_blk_bootstrap_single_iter()
