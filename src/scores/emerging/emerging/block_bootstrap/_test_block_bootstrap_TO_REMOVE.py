# pylint: disable=all

import numpy as np

from scores.emerging.block_bootstrap.axis_info import AxisInfo, make_axis_info
from scores.emerging.block_bootstrap.block_bootstrap import (
    construct_block_bootstrap_array,
)
from scores.emerging.block_bootstrap.methods import FitBlocksMethod


def _test_numpy_blk_bootstrap_single_iter():
    import pprint  # pylint: disable=import-outside-toplevel

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
    (output_arr, block_sample_idx) = construct_block_bootstrap_array(
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


if __name__ == "__main__":
    _test_numpy_blk_bootstrap_single_iter()
