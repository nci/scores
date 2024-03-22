"""Test data for test_processing"""

import numpy as np
import xarray as xr

DA_1 = xr.DataArray([[1, 2], [3, 4]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]})
DA_2 = xr.DataArray([[5, 6], [7, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]})
DA_3 = xr.DataArray([[np.nan, 8], [9, 10]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]})
DA_4 = xr.DataArray([11, 12], dims=["a"], coords={"a": [0, 1]})
DA_5 = xr.DataArray([np.nan, 12], dims=["a"], coords={"a": [0, 1]})
DA_6 = xr.DataArray([13], dims=["a"], coords={"a": [0]})
DA_7 = xr.DataArray([14, np.nan, 15], dims=["a"], coords={"a": [0, 1, 2]})

EXPECTED_12 = (
    xr.DataArray([[1, 2], [3, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[5, 6], [7, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)
EXPECTED_21 = (
    xr.DataArray([[5, 6], [7, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[1, 2], [3, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)
EXPECTED_123 = (
    xr.DataArray([[np.nan, 2], [3, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[np.nan, 6], [7, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[np.nan, 8], [9, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)
EXPECTED_24 = (
    xr.DataArray([[5, 6], [7, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[11, 11], [12, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)
EXPECTED_15 = (
    xr.DataArray([[np.nan, np.nan], [3, 4]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
    xr.DataArray([[np.nan, np.nan], [12, 12]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)
EXPECTED_26 = (
    xr.DataArray([[5, 6]], dims=["a", "b"], coords={"a": [0], "b": [2, 3]}),
    xr.DataArray([[13, 13]], dims=["a", "b"], coords={"a": [0], "b": [2, 3]}),
)
EXPECTED_37 = (
    xr.DataArray(
        [[np.nan, 8], [np.nan, np.nan]],
        dims=["a", "b"],
        coords={"a": [0, 1], "b": [2, 3]},
    ),
    xr.DataArray(
        [[np.nan, 14], [np.nan, np.nan]],
        dims=["a", "b"],
        coords={"a": [0, 1], "b": [2, 3]},
    ),
)

DS_12 = xr.Dataset({"DA_1": DA_1, "DA_2": DA_2})
DS_123 = xr.Dataset({"DA_1": DA_1, "DA_2": DA_2, "DA_3": DA_3})
DS_3 = xr.Dataset({"DA_3": DA_3})
DS_7 = xr.Dataset({"DA_7": DA_7})

EXPECTED_DS12 = (
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[1, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[5, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
)
EXPECTED_DS123 = (
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[np.nan, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[np.nan, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_3": xr.DataArray(
                [[np.nan, 8], [9, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
)
EXPECTED_DS12_DS123 = (
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[np.nan, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[np.nan, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[np.nan, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[np.nan, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_3": xr.DataArray(
                [[np.nan, 8], [9, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
)
EXPECTED_DS12_DS3 = (
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[np.nan, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[np.nan, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
    xr.Dataset(
        {
            "DA_3": xr.DataArray(
                [[np.nan, 8], [9, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            )
        }
    ),
)
EXPECTED_DS3_7 = (
    xr.Dataset(
        {
            "DA_3": xr.DataArray(
                [[np.nan, 8], [np.nan, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            )
        }
    ),
    xr.DataArray(
        [[np.nan, 14], [np.nan, np.nan]],
        dims=["a", "b"],
        coords={"a": [0, 1], "b": [2, 3]},
    ),
)
EXPECTED_DS7_3 = (
    xr.Dataset(
        {
            "DA_7": xr.DataArray(
                [[np.nan, 14], [np.nan, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            )
        }
    ),
    xr.DataArray(
        [[np.nan, 8], [np.nan, np.nan]],
        dims=["a", "b"],
        coords={"a": [0, 1], "b": [2, 3]},
    ),
)
EXPECTED_DS12_3 = (
    xr.Dataset(
        {
            "DA_1": xr.DataArray(
                [[np.nan, 2], [3, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
            "DA_2": xr.DataArray(
                [[np.nan, 6], [7, np.nan]],
                dims=["a", "b"],
                coords={"a": [0, 1], "b": [2, 3]},
            ),
        }
    ),
    xr.DataArray([[np.nan, 8], [9, np.nan]], dims=["a", "b"], coords={"a": [0, 1], "b": [2, 3]}),
)

"""
Test data for comparative_discretise
"""
# input data
DA_1D = xr.DataArray([1, 2.4, 1.2, np.nan], coords=[("a", [0, 1, 2, 3])])
DA_2D = xr.DataArray([[1, 2.4], [1.2, np.nan]], coords=[("a", [0, 1]), ("b", [2, 3])])
DA_3D = xr.DataArray(
    [[[1, 2.4], [1.2, np.nan]], [[1, 1.1], [np.nan, 1.3]]],
    coords=[("a", [0, 2]), ("b", [2, 3]), ("c", [np.pi, 0])],
)
# input data for testing abs_tolerance
DATA_4X1 = xr.DataArray([0.39, 0.399999999, 0.2, 0.7], coords=[("day", [0, 1, 2, 3])])
DATA_4X1_NAN = xr.DataArray([0.39, 0.399999999, 0.2, np.nan], coords=[("day", [0, 1, 2, 3])])
DATA_4X1_2NAN = xr.DataArray([np.nan, 0.399999999, 0.2, np.nan], coords=[("day", [0, 1, 2, 3])])
DATA_4X2_POINT4_POINT5_NAN = xr.DataArray(
    [[np.nan, -0.3], [0.399999999, 0.39], [0.499999999, 0.5]],
    coords=[("day", [0, 1, 2]), ("colour", [0, 1])],
)
DATA_5X1_POINT4 = xr.DataArray([0.39, 0.399999999, 0.4, 0.40000000001, 0.41], coords=[("day", [0, 1, 2, 3, 4])])

# 1-D comparison data with thresholds
THRESH_DA_0 = xr.DataArray([0.3], dims=["threshold"], coords={"threshold": [0.3]})
THRESH_DA_1 = xr.DataArray([0.2, 0.3, 0.4], dims=["threshold"], coords={"threshold": [0.2, 0.3, 0.4]})
THRESH_DA_2 = xr.DataArray([1, 5, 7], dims=["threshold"], coords={"threshold": [1, 5, 7]})
THRESH_DA_3 = xr.DataArray([0.4, 0.5], dims=["threshold"], coords={"threshold": [0.4, 0.5]})
THRESH_DA_4 = xr.DataArray([0.4], dims=["threshold"], coords={"threshold": [0.4]})
THRESH_DA_5 = xr.DataArray([0.3, 0.8], dims=["threshold"], coords={"threshold": [0.3, 0.8]})

# n-D comparison data
COMP_1D = xr.DataArray([1.3, 1.1], dims=["banana"], coords={"banana": [2, 1]})
COMP_1DA = xr.DataArray([1.1, 2.5, np.nan], dims=["a"], coords={"a": [2, 1, 0]})
COMP_2D = xr.DataArray([[0, 4, 1], [2, np.nan, 1]], coords=[("banana", [2, 1]), ("apple", [np.pi, 3, 0])])
COMP_2DA = xr.DataArray([[1.3, 1.1], [2.4, np.nan], [1, 4]], coords=[("a", [2, 1, 3]), ("banana", [2, 1])])

####################################################
# Tests with 1-D comparison, testing abs_tolerance #
####################################################
EXP_CDIS_0 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_1 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_2 = xr.DataArray(
    [0.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_3 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_4 = xr.DataArray(
    [0.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 1e-10, "discretisation_mode": ">="},
)
EXP_CDIS_5 = xr.DataArray(
    [1, 1.0, 0],
    dims=["threshold"],
    coords={"threshold": [0.2, 0.3, 0.4]},
    attrs={"discretisation_tolerance": 1e-10, "discretisation_mode": ">="},
)
EXP_CDIS_6 = xr.DataArray(
    [1.0, 1.0, 0.0],
    dims=["threshold"],
    coords={"threshold": [1, 5, 7]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_7 = xr.DataArray(
    [np.nan, np.nan, np.nan],
    dims=["threshold"],
    coords={"threshold": [1, 5, 7]},
    attrs={"discretisation_tolerance": 1e-10, "discretisation_mode": ">="},
)
EXP_CDIS_8 = xr.DataArray(
    [[1.0], [0.0], [1.0]],
    coords=[("day", [0, 1, 2]), ("threshold", [0.4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_9 = xr.DataArray(
    [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
    coords=[("day", [0, 1, 2]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_10 = xr.DataArray(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
    coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_11 = xr.DataArray(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
    coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_12 = xr.DataArray(
    [[np.nan, np.nan], [1.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
    coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_13 = xr.DataArray(
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
    coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-10, "discretisation_mode": ">="},
)
EXP_CDIS_14 = xr.DataArray(
    [[[np.nan, np.nan], [0, 0]], [[1, 0], [0, 0]], [[1, 1], [1, 1]]],
    coords=[("day", [0, 1, 2]), ("colour", [0, 1]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)

############
# Datasets #
############
EXP_CDIS_DS_1D_1D = xr.Dataset(
    {
        "zero": xr.DataArray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
        ),
        "one": xr.DataArray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
            coords=[("day", [0, 1, 2, 3]), ("threshold", [0.4, 0.5])],
        ),
    },
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_DS_0D_1D = xr.Dataset(
    {"zero": xr.DataArray([1.0], dims=["threshold"], coords={"threshold": [0.3]})},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)

#############################
# Tests with n-D comparison #
#############################
EXP_CDIS_0D_1D = xr.DataArray(
    [0.0, 0.0],
    coords=[("banana", [2, 1])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_1D_1DA = xr.DataArray(
    [np.nan, 0, 1],
    coords=[("a", [0, 1, 2])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_1D_0D = xr.DataArray(
    [0, 1, 0, np.nan],
    dims=["a"],
    coords={"a": [0, 1, 2, 3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_1D_2DA = xr.DataArray(
    [[1, np.nan], [0, 1], [np.nan, np.nan]],
    coords=[("a", [1, 2, 3]), ("banana", [2, 1])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_1D_2D = xr.DataArray(
    [
        [[1, 0, 0], [0, np.nan, 0]],
        [[1, 0, 1], [1, np.nan, 1]],
        [[1, 0, 1], [0, np.nan, 1]],
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    ],
    coords=[("a", [0, 1, 2, 3]), ("banana", [2, 1]), ("apple", [np.pi, 3, 0])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_2D_0D = xr.DataArray(
    [[0, 1], [0, np.nan]],
    coords=[("a", [0, 1]), ("b", [2, 3])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_2D_1DA = xr.DataArray(
    [[np.nan, np.nan], [0, np.nan]],
    coords=[("a", [0, 1]), ("b", [2, 3])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_2D_1D = xr.DataArray(
    [[[0, 0], [1, 1]], [[0, 1], [np.nan, np.nan]]],
    coords=[("a", [0, 1]), ("b", [2, 3]), ("banana", [2, 1])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_3D_1DA = xr.DataArray(
    [[[np.nan, np.nan], [np.nan, np.nan]], [[0, 0], [np.nan, 1]]],
    coords=[("a", [0, 2]), ("b", [2, 3]), ("c", [np.pi, 0])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_3D_3D = xr.DataArray(
    [[[0.0, 0.0], [0.0, np.nan]], [[0.0, 0.0], [np.nan, 0.0]]],
    coords=[("a", [0, 2]), ("b", [2, 3]), ("c", [np.pi, 0])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
################################################################
# SMOKE TESTS FOR ALL MODES ['<', '>', '>=', '<=', '++', '!='] #
################################################################
EXP_CDIS_GE0 = xr.DataArray(
    [0.0, 1.0, 1.0, 1.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_CDIS_GE1 = xr.DataArray(
    [0.0, 0.0, 1.0, 1.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_CDIS_GT0 = xr.DataArray(
    [0.0, 0.0, 0.0, 0.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
)
EXP_CDIS_GT1 = xr.DataArray(
    [0.0, 0.0, 0.0, 1.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_CDIS_LE0 = xr.DataArray(
    [1.0, 1.0, 1.0, 1.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<="},
)
EXP_CDIS_LE1 = xr.DataArray(
    [1.0, 1.0, 1.0, 0.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "<="},
)
EXP_CDIS_LT0 = xr.DataArray(
    [1.0, 0.0, 0.0, 0.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<"},
)
EXP_CDIS_LT1 = xr.DataArray(
    [1.0, 1.0, 0.0, 0.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "<"},
)
EXP_CDIS_EQ0 = xr.DataArray(
    [0.0, 1.0, 1.0, 1.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "=="},
)
EXP_CDIS_EQ1 = xr.DataArray(
    [0.0, 0.0, 1.0, 0.0, 0.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "=="},
)
EXP_CDIS_NE0 = xr.DataArray(
    [1.0, 0.0, 0.0, 0.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "!="},
)
EXP_CDIS_NE1 = xr.DataArray(
    [1.0, 1.0, 0.0, 1.0, 1.0],
    dims=["day"],
    coords=[("day", [0, 1, 2, 3, 4])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "!="},
)

##################################
# 0-D Integer & float comparison #
##################################
EXP_CDIS_ONE = xr.DataArray(
    [1.0],
    coords=[("a", [0])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
)

"""
Testing data for binary_discretise
"""
EXP_DIS_0 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_DIS_1 = xr.DataArray(
    1.0,
    coords={"threshold": 0.3},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_DIS_GE0 = xr.DataArray(
    [0.0, 1.0, 1.0, 1.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_DIS_GE1 = xr.DataArray(
    [0.0, 0.0, 1.0, 1.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_DIS_GT0 = xr.DataArray(
    [0.0, 0.0, 0.0, 0.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
)
EXP_DIS_GT1 = xr.DataArray(
    [0.0, 0.0, 0.0, 1.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">"},
)
EXP_DIS_LE0 = xr.DataArray(
    [1.0, 1.0, 1.0, 1.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<="},
)
EXP_DIS_LE1 = xr.DataArray(
    [1.0, 1.0, 1.0, 0.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "<="},
)
EXP_DIS_LT0 = xr.DataArray(
    [1.0, 0.0, 0.0, 0.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<"},
)
EXP_DIS_LT1 = xr.DataArray(
    [1.0, 1.0, 0.0, 0.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "<"},
)
EXP_DIS_EQ0 = xr.DataArray(
    [0.0, 1.0, 1.0, 1.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "=="},
)
EXP_DIS_EQ1 = xr.DataArray(
    [0.0, 0.0, 1.0, 0.0, 0.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "=="},
)
EXP_DIS_NE0 = xr.DataArray(
    [1.0, 0.0, 0.0, 0.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "!="},
)
EXP_DIS_NE1 = xr.DataArray(
    [1.0, 1.0, 0.0, 1.0, 1.0],
    dims=["day"],
    coords={"day": [0, 1, 2, 3, 4], "threshold": 0.4},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": "!="},
)
EXP_DIS_DS_1D = xr.Dataset(
    {
        "zero": xr.DataArray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            dims=["day", "threshold"],
            coords={"day": [0, 1, 2, 3], "threshold": [0.4, 0.5]},
        ),
        "one": xr.DataArray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
            dims=["day", "threshold"],
            coords={"day": [0, 1, 2, 3], "threshold": [0.4, 0.5]},
        ),
    },
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_DIS_DS_0D = xr.Dataset(
    {"zero": xr.DataArray(1.0, coords={"threshold": 0.3})},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)

"""
Test data for _binary_discretise_proportion
"""

EXP_BDP_0 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_BDP_1 = xr.DataArray(
    [[[np.nan, np.nan], [0, 0]], [[1, 0], [0, 0]], [[1, 1], [1, 1]]],
    coords=[("day", [0, 1, 2]), ("colour", [0, 1]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_BDP_2 = xr.DataArray(
    [[1, 0.5], [1 / 3, 1 / 3]],
    coords=[("colour", [0, 1]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_BDP_3 = xr.DataArray(
    [0.6, 0.4],
    dims=["threshold"],
    coords=[("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)
EXP_BDP_4 = xr.DataArray(
    0.2,
    coords={"threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">"},
)
EXP_BDP_5 = xr.DataArray(
    0.8,
    coords={"threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<="},
)
EXP_BDP_6 = xr.DataArray(
    0.2,
    coords={"threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "<"},
)
EXP_BDP_7 = xr.DataArray(
    0.6,
    coords={"threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "=="},
)
EXP_BDP_8 = xr.DataArray(
    0.4,
    coords={"threshold": 0.4},
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": "!="},
)

EXP_BDP_9 = xr.Dataset(
    {
        "zero": xr.DataArray([0.5, 0.25], coords=[("threshold", [0.4, 0.5])]),
        "one": xr.DataArray([1 / 3, 0], coords=[("threshold", [0.4, 0.5])]),
    },
    attrs={"discretisation_tolerance": 1e-8, "discretisation_mode": ">="},
)

# Data for testing proportion_exceeding
EXP_PE_0 = xr.DataArray(
    [1.0],
    dims=["threshold"],
    coords={"threshold": [0.3]},
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_PE_1 = xr.DataArray(
    [[[np.nan, np.nan], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [1, 1]]],
    coords=[("day", [0, 1, 2]), ("colour", [0, 1]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_PE_2 = xr.DataArray(
    [[0.5, 0], [1 / 3, 1 / 3]],
    coords=[("colour", [0, 1]), ("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_PE_3 = xr.DataArray(
    [0.4, 0.2],
    coords=[("threshold", [0.4, 0.5])],
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
EXP_PE_4 = xr.Dataset(
    {
        "zero": xr.DataArray([0.75, 0.25, 0.25], coords=[("threshold", [0.39, 0.4, 0.5])]),
        "one": xr.DataArray([2 / 3, 0.0, 0.0], coords=[("threshold", [0.39, 0.4, 0.5])]),
    },
    attrs={"discretisation_tolerance": 0, "discretisation_mode": ">="},
)
