"""Tests for processing"""
import numpy as np
import pytest
import xarray as xr

from scores.processing import (
    binary_discretise,
    broadcast_and_match_nan,
    check_binary,
    comparative_discretise,
)
from tests import test_processing_data as xtd


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


@pytest.mark.parametrize(
    ("data", "comparison", "mode", "abs_tolerance", "expected"),
    [
        ####################################################
        # Tests with 1-D comparison, testing abs_tolerance #
        ####################################################
        # TESTS FOR MODE='>='
        # 0: mode='>=', 0-D, value greater than threshold
        (xr.DataArray(0.5), xtd.THRESH_DA_0, ">=", None, xtd.EXP_CDIS_0),
        # 1: mode='>=', 0-D, equal value and threshold
        (xr.DataArray(0.3), xtd.THRESH_DA_0, ">=", None, xtd.EXP_CDIS_1),
        # 2: mode='>=', 0-D, value less than threshold
        (xr.DataArray(0.2), xtd.THRESH_DA_0, ">=", None, xtd.EXP_CDIS_2),
        # 3: mode='>=', 0-D, value less than threshold but within tolerance
        (xr.DataArray(0.299999999), xtd.THRESH_DA_0, ">=", 1e-8, xtd.EXP_CDIS_3),
        # 4: mode='>=', 0-D, value less than threshold and outside tolerance
        (xr.DataArray(0.299999999), xtd.THRESH_DA_0, ">=", 1e-10, xtd.EXP_CDIS_4),
        # 5: mode='>=', 0-D, threshold of length 3
        (xr.DataArray(0.3), xtd.THRESH_DA_1, ">=", 1e-10, xtd.EXP_CDIS_5),
        # 6: mode='>=', 0-D, integers, threshold of length 3
        (xr.DataArray(5), xtd.THRESH_DA_2, ">=", 0, xtd.EXP_CDIS_6),
        # 7: mode='>=', 0-D, NaN, threshold of length 3
        (xr.DataArray(np.nan), xtd.THRESH_DA_2, ">=", 1e-10, xtd.EXP_CDIS_7),
        # 8: mode='>=', 1-D, one threshold
        (
            xr.DataArray([0.4, 0.2, 0.7], dims=["day"], coords={"day": [0, 1, 2]}),
            xtd.THRESH_DA_4,
            ">=",
            None,
            xtd.EXP_CDIS_8,
        ),
        # 9: mode='>=', 1-D, two thresholds
        (
            xr.DataArray([0.4, 0.2, 0.7], dims=["day"], coords={"day": [0, 1, 2]}),
            xtd.THRESH_DA_3,
            ">=",
            None,
            xtd.EXP_CDIS_9,
        ),
        # 10: mode='>=', 1-D, two thresholds, tolerance=1e-8
        (xtd.DATA_4X1, xtd.THRESH_DA_3, ">=", 1e-8, xtd.EXP_CDIS_10),
        # 11: mode='>=', 1-D, with NaN, two thresholds, tolerance=1e-8
        (xtd.DATA_4X1_NAN, xtd.THRESH_DA_3, ">=", 1e-8, xtd.EXP_CDIS_11),
        # 12: mode='>=', 1-D, with NaNs, two thresholds, tolerance=1e-8
        (xtd.DATA_4X1_2NAN, xtd.THRESH_DA_3, ">=", 1e-8, xtd.EXP_CDIS_12),
        # 13: mode='>=', 1-D, with NaN, two thresholds, tolerance=1e-10
        (xtd.DATA_4X1_NAN, xtd.THRESH_DA_3, ">=", 1e-10, xtd.EXP_CDIS_13),
        # 14: mode='>=', 2-D, with NaN, two thresholds, tolerance=1e-8
        (xtd.DATA_4X2_POINT4_POINT5_NAN, xtd.THRESH_DA_3, ">=", 1e-8, xtd.EXP_CDIS_14),
        ############
        # Datasets #
        ############
        # 15: Data 1-D dataset, comparison 1-D
        (
            xr.Dataset({"zero": xtd.DATA_4X1, "one": xtd.DATA_4X1_NAN}),
            xtd.THRESH_DA_3,
            ">=",
            1e-8,
            xtd.EXP_CDIS_DS_1D_1D,
        ),
        # 16: Dataset input, mode='>=', 0-D data, 0_D comparison
        (
            xr.Dataset({"zero": xr.DataArray(0.5)}),
            xtd.THRESH_DA_0,
            ">=",
            None,
            xtd.EXP_CDIS_DS_0D_1D,
        ),
        #############################
        # Tests with n-D comparison #
        #############################
        # 17. 0-D data, 0-D comparison
        (
            xr.DataArray(0.3),
            xr.DataArray(2),
            ">",
            1e-8,
            xr.DataArray(
                0.0,
                attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
            ),
        ),
        # 18. 0-D data, 1-D comparison
        (xr.DataArray(0.3), xtd.COMP_1D, ">", None, xtd.EXP_CDIS_0D_1D),
        # 19. 1-D data, 0-D comparison
        (xtd.DA_1D, xr.DataArray(1.3), ">", None, xtd.EXP_CDIS_1D_0D),
        # 20. 1-D data, 1-D comparison with shared dimension
        (xtd.DA_1D, xtd.COMP_1DA, ">", None, xtd.EXP_CDIS_1D_1DA),
        # 21. 1-D data, 2-D comparison, one dimension shared
        (xtd.DA_1D, xtd.COMP_2DA, ">=", None, xtd.EXP_CDIS_1D_2DA),
        # 22. 1-D data, 2-D comparison, no shared dimension
        (xtd.DA_1D, xtd.COMP_2D, ">", None, xtd.EXP_CDIS_1D_2D),
        # 23. 2-D data, 0-D comparison
        (xtd.DA_2D, xr.DataArray(1.2), ">", None, xtd.EXP_CDIS_2D_0D),
        # 24. 2-D data, 1-D comparison with shared dimension
        (xtd.DA_2D, xtd.COMP_1DA, ">", None, xtd.EXP_CDIS_2D_1DA),
        # 25. 2-D data, 1-D comparison
        (xtd.DA_2D, xtd.COMP_1D, ">", None, xtd.EXP_CDIS_2D_1D),
        # 26. 3-D data, 1-D comparison withshared dimension
        (xtd.DA_3D, xtd.COMP_1DA, ">", None, xtd.EXP_CDIS_3D_1DA),
        # 27. 3-D data, 1-D comparison withshared dimension
        (xtd.DA_3D, xtd.DA_3D, ">", None, xtd.EXP_CDIS_3D_3D),
        ################################################################
        # SMOKE TESTS FOR ALL MODES ['<', '>', '>=', '<=', '++', '!='] #
        ################################################################
        # 28. mode='>=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), ">=", 1e-8, xtd.EXP_CDIS_GE0),
        # 29. mode='>=', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), ">=", None, xtd.EXP_CDIS_GE1),
        # 30. mode='>', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), ">", 1e-8, xtd.EXP_CDIS_GT0),
        # 31. mode='>', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), ">", None, xtd.EXP_CDIS_GT1),
        # 32. mode='<=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "<=", 1e-8, xtd.EXP_CDIS_LE0),
        # 33. mode='<=', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "<=", None, xtd.EXP_CDIS_LE1),
        # 34. mode='<', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "<", 1e-8, xtd.EXP_CDIS_LT0),
        # 35. mode='<', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "<", None, xtd.EXP_CDIS_LT1),
        # 36. mode='==', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "==", 1e-8, xtd.EXP_CDIS_EQ0),
        # 37. mode='==', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "==", None, xtd.EXP_CDIS_EQ1),
        # 38. mode='!=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "!=", 1e-8, xtd.EXP_CDIS_NE0),
        # 39. mode='!=', tolerance=0
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "!=", None, xtd.EXP_CDIS_NE1),
        ##################################
        # 0-D Integer & float comparison #
        ##################################
        # 40. Integer DataArray, float comparison
        (
            xr.DataArray([5], dims=["a"], coords={"a": [0]}),
            4.0,
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 41. Integer DataArray, integer comparison (output should still be float!)
        (
            xr.DataArray([5], dims=["a"], coords={"a": [0]}),
            4,
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 42. Float DataArray, integer comparison
        (
            xr.DataArray([5.0], dims=["a"], coords={"a": [0]}),
            4.0,
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 43. Float DataArray data, 0-D float comparison
        (
            xr.DataArray(2.3),
            2.0,
            ">",
            1e-8,
            xr.DataArray(
                1.0,
                attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
            ),
        ),
        # 44. NaN in comparison
        (
            xr.DataArray(2),
            np.nan,
            ">",
            0,
            xr.DataArray(
                np.nan,
                attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
            ),
        ),
        ##########################
        # Float & int DataArrays #
        ##########################
        # 45. Integer DataArray, float comparison
        (
            xr.DataArray([5], dims=["a"], coords={"a": [0]}),
            xr.DataArray([4.0], dims=["a"], coords={"a": [0]}),
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 46. Integer DataArray, integer comparison (output should still be float!)
        (
            xr.DataArray([5], dims=["a"], coords={"a": [0]}),
            xr.DataArray([4], dims=["a"], coords={"a": [0]}),
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 47. Float DataArray, integer comparison
        (
            xr.DataArray([5.0], dims=["a"], coords={"a": [0]}),
            xr.DataArray([4.0], dims=["a"], coords={"a": [0]}),
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
        # 48. Float DataArray data, 0-D float comparison
        (
            xr.DataArray(2.3),
            xr.DataArray([2.0], dims=["a"], coords={"a": [0]}),
            ">",
            1e-8,
            xtd.EXP_CDIS_ONE,
        ),
    ],
)
def test_comparative_discretise(data, comparison, mode, abs_tolerance, expected):
    """
    Tests comparative_discretise
    """
    calculated = comparative_discretise(data, comparison, mode, abs_tolerance=abs_tolerance)
    xr.testing.assert_equal(calculated, expected)


@pytest.mark.parametrize(
    ("data", "comparison", "mode", "abs_tolerance", "error_class", "error_msg_snippet"),
    [
        # invalid abs_tolerance
        (
            xr.DataArray(0.5),
            xr.DataArray(0.3),
            ">=",
            -1e-8,
            ValueError,
            "value -1e-08 of abs_tolerance is invalid, it must be a non-negative float",
        ),
        # invalid mode
        (
            xr.DataArray(0.5),
            xr.DataArray(0.3),
            "&",
            1e-8,
            ValueError,
            "'&' is not a valid mode. Available modes are: ['<', '<=', '>', '>=', '!=', '==']",
        ),
        # invalid comparison
        (
            xr.DataArray(0.5),
            np.array(0.5),
            ">",
            None,
            TypeError,
            "comparison must be a float, int or xarray.DataArray",
        ),
    ],
)
def test_comparative_discretise_raises(data, comparison, mode, abs_tolerance, error_class, error_msg_snippet):
    """
    Tests that .comparitive_discretise raises the correct error
    """
    with pytest.raises(error_class) as exc:
        comparative_discretise(data, comparison, mode, abs_tolerance=abs_tolerance)
    assert error_msg_snippet in str(exc.value)


@pytest.mark.parametrize(
    ("data", "thresholds", "mode", "abs_tolerance", "autosqueeze", "expected"),
    [
        # Test autosqueeze
        # 0. mode='>=', 0-D, value greater than threshold, autosqueeze=False
        (xr.DataArray(0.5), [0.3], ">=", None, False, xtd.EXP_DIS_0),
        # 1. mode='>=', 0-D, test autosqueeze
        (xr.DataArray(0.5), [0.3], ">=", None, True, xtd.EXP_DIS_1),
        # SMOKE TESTS FOR ALL MODES: ['<', '>', '>=', '<=', '==', '!=']
        # 2. mode='>=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], ">=", 1e-8, True, xtd.EXP_DIS_GE0),
        # 3. mode='>=', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], ">=", None, True, xtd.EXP_DIS_GE1),
        # 4. mode='>', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], ">", 1e-8, True, xtd.EXP_DIS_GT0),
        # 5. mode='>', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], ">", None, True, xtd.EXP_DIS_GT1),
        # 6. mode='<=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], "<=", 1e-8, True, xtd.EXP_DIS_LE0),
        # 7. mode='<=', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], "<=", None, True, xtd.EXP_DIS_LE1),
        # 8. mode='<', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], "<", 1e-8, True, xtd.EXP_DIS_LT0),
        # 9. mode='<', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], "<", None, True, xtd.EXP_DIS_LT1),
        # 10. mode='==', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], "==", 1e-8, True, xtd.EXP_DIS_EQ0),
        # 11. mode='==', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], "==", None, True, xtd.EXP_DIS_EQ1),
        # 12. mode='!=', tolerance=1e-8
        (xtd.DATA_5X1_POINT4, [0.4], "!=", 1e-8, True, xtd.EXP_DIS_NE0),
        # 13. mode='!=', tolerance=0
        (xtd.DATA_5X1_POINT4, [0.4], "!=", None, True, xtd.EXP_DIS_NE1),
        # Dataset input
        # 14. 1-D data,
        (
            xr.Dataset({"zero": xtd.DATA_4X1, "one": xtd.DATA_4X1_NAN}),
            [0.4, 0.5],
            ">=",
            1e-8,
            True,
            xtd.EXP_DIS_DS_1D,
        ),
        # 15. autosqueeze=True
        (
            xr.Dataset({"zero": xr.DataArray(0.5)}),
            [0.3],
            ">=",
            None,
            True,
            xtd.EXP_DIS_DS_0D,
        ),
        # 16: 2-D with NaN,
        (xtd.DATA_4X2_POINT4_POINT5_NAN, [0.4, 0.5], ">=", 1e-8, True, xtd.EXP_CDIS_14),
        # Checks with 0-D input
        # 17: float
        (xtd.DATA_5X1_POINT4, 0.4, ">=", 1e-8, True, xtd.EXP_DIS_GE0),
        # 18: np.array
        (xtd.DATA_5X1_POINT4, np.array(0.4), "<=", 1e-8, True, xtd.EXP_DIS_LE0),
        # 19: xr.DataArray
        (xtd.DATA_5X1_POINT4, xr.DataArray(0.4), "==", 1e-8, True, xtd.EXP_DIS_EQ0),
        # 20: ignore autosqueeze
        (xtd.DATA_5X1_POINT4, 0.4, ">=", 1e-8, False, xtd.EXP_DIS_GE0),
    ],
)
def test_binary_discretise(data, thresholds, mode, abs_tolerance, autosqueeze, expected):
    """
    Tests binary_discretise
    """
    calc = binary_discretise(data, thresholds, mode, abs_tolerance=abs_tolerance, autosqueeze=autosqueeze)
    xr.testing.assert_equal(calc, expected)


@pytest.mark.parametrize(
    (
        "data",
        "thresholds",
        "mode",
        "abs_tolerance",
        "autosqueeze",
        "error_class",
        "error_msg_snippet",
    ),
    [
        # invalid thresholds
        (
            xr.DataArray(0.5),
            [0.3, 0.2],
            ">=",
            None,
            False,
            ValueError,
            "Values in `thresholds` are not montonic increasing",
        ),
        # invalid abs_tolerance
        (
            xr.DataArray(0.5),
            [0.2, 0.5],
            ">=",
            -1e-8,
            False,
            ValueError,
            "value -1e-08 of abs_tolerance is invalid, it must be a non-negative float",
        ),
        # invalid mode
        (
            xr.DataArray(0.5),
            [0.2, 0.5],
            "&",
            1e-8,
            False,
            ValueError,
            "'&' is not a valid mode. Available modes are: ['<', '<=', '>', '>=', '!=', '==']",
        ),
        # 'threshold' in data.dims:
        (
            xr.DataArray([0.5], dims=["threshold"]),
            [0.3],
            ">=",
            None,
            False,
            ValueError,
            "'threshold' must not be in the supplied data object dimensions",
        ),
    ],
)
def test_binary_discretise_raises(data, thresholds, mode, abs_tolerance, autosqueeze, error_class, error_msg_snippet):
    """
    Tests that binary_discretise raises the correct error
    """
    with pytest.raises(error_class) as exc:
        binary_discretise(data, thresholds, mode, abs_tolerance=abs_tolerance, autosqueeze=autosqueeze)
    assert error_msg_snippet in str(exc.value)


@pytest.mark.parametrize(
    ("da"),
    [
        (xr.DataArray([0, 1, 2])),
        (xr.DataArray([0, 1, -1])),
        (xr.DataArray([0, 1, 0.5])),
        (xr.DataArray([[0, 1, 1.0000001], [0, 1, 1]])),
    ],
)
def test_check_binary_raises(da):
    """test check_binary raises"""
    with pytest.raises(ValueError) as exc:
        check_binary(da, "my name")
    assert "`my name` contains values that are not in the set {0, 1, np.nan}" in str(exc.value)


@pytest.mark.parametrize(
    ("da"),
    [
        (xr.DataArray([0, 1])),
        (xr.DataArray([0, 0])),
        (xr.DataArray([1, 1])),
        (xr.DataArray([0, 1, np.nan])),
        (xr.DataArray([[0, 1, np.nan], [0, 1, np.nan]])),
    ],
)
def test_check_binary_doesnt_raise(da):
    """test check_binary doesn't raise"""
    check_binary(da, "my name")
