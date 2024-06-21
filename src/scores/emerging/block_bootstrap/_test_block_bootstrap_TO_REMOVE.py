# pylint: disable=all

import pprint

import numpy as np
import xarray as xr

from scores.emerging.block_bootstrap.axis_info import (
    AxisInfo,
    make_axis_info,
    make_axis_info_collection,
)
from scores.emerging.block_bootstrap.block_bootstrap import (
    _construct_block_bootstrap_array,
)
from scores.emerging.block_bootstrap.helpers import (
    partial_linear_order_by_ref,
    reorder_all_arr_dims,
    reorder_dims,
)
from scores.emerging.block_bootstrap.methods import FitBlocksMethod


def _test_numpy_blk_bootstrap_single_iter():
    print("\nBOOTSTRAPPING (single iteration, single array)...")
    # generate test data
    axis_len = [13, 10, 8, 7]
    block_sizes = [4, 3, 2, 3]
    method = FitBlocksMethod.PARTIAL
    rng = np.random.default_rng(seed=42)
    # random dataset with integers so its easy to visualize
    input_arr = rng.integers(low=0, high=10, size=axis_len)

    print("\n--- axis info ---")
    axis_info = make_axis_info(input_arr, block_sizes, method)
    pprint.pp(axis_info)

    print("\n--- input array ---")
    pprint.pp(input_arr.shape)
    res = np.histogram(input_arr, bins=5)
    pprint.pp(res)

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


def _test_make_axis_info_collection():
    print("\nAXIS COLLECTION...")

    arr_test_1 = np.random.rand(10, 10, 12, 7, 5)
    da_1 = xr.DataArray(
        data=arr_test_1,
        dims=["x", "y", "time", "lead_time", "height"],
    )
    arr_test_2 = np.random.rand(10, 10, 7, 5)
    da_2 = xr.DataArray(
        data=arr_test_2,
        dims=["y", "x", "lead_time", "height"],
    )
    arr_test_3 = np.random.rand(10, 10, 7)
    da_3 = xr.DataArray(
        data=arr_test_3,
        dims=["x", "y", "lead_time"],
    )
    axi_collection = make_axis_info_collection(
        [da_1, da_2, da_3],
        ["lead_time", "x", "y"],
        [2, 4, 5],
    )

    print("\n--- axis collection ---")
    pprint.pp(axi_collection)

    try:
        make_axis_info_collection(
            [da_1, da_2, da_3],
            ["lead_time", "x", "y"],
            [2, 4, 12],
        )
    except AssertionError:
        print(f"\n--- failure due to invalid block size ---")
        print(f"Caught expected assertion error")

    try:
        arr_test_fail = np.random.rand(10, 11, 7)
        da_fail = xr.DataArray(
            data=arr_test_fail,
            dims=["x", "y", "lead_time"],
        )
        make_axis_info_collection(
            [da_1, da_2, da_fail],
            ["lead_time", "x", "y"],
            [2, 4, 5],
        )
    except ValueError as e:
        print(f"\n--- failure due to inconsistent dim size ---")
        print(f"Caught expected value error: {e}")


def _test_reorder_all_dims():
    print("\nREORDER ALL ARRAY DIMS...")

    arr_test_1 = np.random.rand(10, 10, 12, 7, 5)
    da_1 = xr.DataArray(
        data=arr_test_1,
        dims=["x", "y", "time", "lead_time", "height"],
    )
    arr_test_2 = np.random.rand(10, 10, 7, 5)
    da_2 = xr.DataArray(
        data=arr_test_2,
        dims=["y", "x", "lead_time", "height"],
    )
    arr_test_3 = np.random.rand(10, 10, 7)
    da_3 = xr.DataArray(
        data=arr_test_3,
        dims=["x", "y", "lead_time"],
    )
    (arrs_ord, all_dims) = reorder_all_arr_dims([da_1, da_2, da_3], ["lead_time", "x", "y"])

    print("\n--- re-order all arrays ---")
    print(f"expected shape for leading dimensions = {(7, 10, 10)}")
    print(f"all dimensions ordered = {all_dims}")
    print("---")
    print(f"shape arr_1 original: {da_1.shape}")
    print(f"shape arr_1 reordered: {arrs_ord[0].shape}")
    print("---")
    print(f"shape arr_2 original: {da_2.shape}")
    print(f"shape arr_2 reordered: {arrs_ord[1].shape}")
    print("---")
    print(f"shape arr_3 original: {da_3.shape}")
    print(f"shape arr_3 reordered: {arrs_ord[2].shape}")
    print("---")

    try:
        reorder_all_arr_dims(
            [da_1, da_2, da_3],
            ["lead_time", "x", "y", "potato"],
        )
    except ValueError as e:
        print(f"\n--- failure due to missing dim ---")
        print(f"Caught expected value error: {e}")


def _test_reorder_dims():
    print("\nREORDER DIMS...")

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
    _test_reorder_dims()
    _test_reorder_all_dims()
    _test_make_axis_info_collection()
    _test_numpy_blk_bootstrap_single_iter()
