"""
Tests scores.processing.test_matching
"""

import numpy as np
import pytest
import xarray as xr

from scores.processing import broadcast_and_match_nan
from tests.processing import test_data as xtd


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # DataArrays
        ([], tuple()),
        ([xtd.DA_2], (xtd.DA_2,)),
        ([xtd.DA_1, xtd.DA_2], xtd.EXPECTED_12),
        ([xtd.DA_2, xtd.DA_1], xtd.EXPECTED_21),
        ([xtd.DA_1, xtd.DA_2, xtd.DA_3], xtd.EXPECTED_123),
        ([xtd.DA_2, xtd.DA_4], xtd.EXPECTED_24),
        ([xtd.DA_1, xtd.DA_5], xtd.EXPECTED_15),
        ([xtd.DA_2, xtd.DA_6], xtd.EXPECTED_26),
        ([xtd.DA_3, xtd.DA_7], xtd.EXPECTED_37),
        # Datasets
        ([xtd.DS_12], xtd.EXPECTED_DS12),
        ([xtd.DS_123], xtd.EXPECTED_DS123),
        ([xtd.DS_12, xtd.DS_123], xtd.EXPECTED_DS12_DS123),
        ([xtd.DS_12, xtd.DS_3], xtd.EXPECTED_DS12_DS3),
        # Datasets and DataArrays
        ([xtd.DS_3, xtd.DA_7], xtd.EXPECTED_DS3_7),
        ([xtd.DS_7, xtd.DA_3], xtd.EXPECTED_DS7_3),
        ([xtd.DS_12, xtd.DA_3], xtd.EXPECTED_DS12_3),
    ],
)
def test_broadcast_and_match_nan(args, expected):
    """
    Tests that broadcast_and_match_nan calculates the correct result
    Args:
        args: a list of the args that will be *-ed into match_DataArray
        expected: a tuple, the expected output of match_DataArray
    """
    calculated = broadcast_and_match_nan(*args)
    for calculated_element, expected_element in zip(calculated, expected):
        assert calculated_element.equals(expected_element)


@pytest.mark.parametrize(
    ("args", "error_msg_snippet"),
    [
        ([xr.Dataset({"DA_1": xtd.DA_1}), xtd.DA_1, np.arange(4)], "Argument 2"),
        ([np.arange(5)], "Argument 0"),
    ],
)
def test_broadcast_and_match_nan_rasies(args, error_msg_snippet):
    """
    Tests that processing.broadcast_and_match_nan correctly raises an ValueError
    """
    with pytest.raises(ValueError) as excinfo:
        broadcast_and_match_nan(*args)
    assert error_msg_snippet in str(excinfo.value)
