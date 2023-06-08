"""
Test data for scores.utils
"""
import numpy as np
import pandas as pd
import xarray as xr

# from XXXXX import _BinaryContingencyTable, _ContingencyTable

# Test data for scores.utils

LEAD_HOURS_CT = [15, 39]
VALID_STARTS_CT = pd.date_range("2000-01-01 15:00:00", periods=2)
STATION_NUMBERS_CT = [10000, 10001]

# BINARY_CONTINGENCY_TABLE = _BinaryContingencyTable(
#     (
#         [
#             [[[[0.0]], [[1.0]]], [[[0.0]], [[1.0]]]],
#             [[[[0.0]], [[1.0]]], [[[0.0]], [[1.0]]]],
#         ]
#     ),
#     coords=[
#         ("lead_hour", LEAD_HOURS_CT),
#         ("valid_start", VALID_STARTS_CT),
#         ("station_number", STATION_NUMBERS_CT),
#         ("fcst_category", [0]),
#         ("obs_category", [0]),
#     ],
#     name="ctable",
#     attrs={"fcst_categories": {"0": 0}, "obs_categories": {"0": 0}},
# )

# CONTINGENCY_TABLE = _ContingencyTable(
#     (
#         [
#             [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
#             [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
#         ]
#     ),
#     coords=[
#         ("lead_hour", LEAD_HOURS_CT),
#         ("fcst_category", [0, 1, 2]),
#         ("obs_category", [0, 1, 2]),
#     ],
#     name="ctable",
#     attrs={
#         "fcst_categories": {"1": 1, "0": 0, "2": 2},
#         "obs_categories": {"1": 1, "0": 0, "2": 2},
#     },
# )
"""
Test data for scores.utils (broadcast_and_match_nan)
"""
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
Test data for scores.utils.check_dims
"""
DA_R = xr.DataArray(np.array(1).reshape((1,)), dims=["red"])
DA_G = xr.DataArray(np.array(1).reshape((1,)), dims=["green"])
DA_B = xr.DataArray(np.array(1).reshape((1,)), dims=["blue"])

DA_RG = xr.DataArray(np.array(1).reshape((1, 1)), dims=["red", "green"])
DA_GB = xr.DataArray(np.array(1).reshape((1, 1)), dims=["green", "blue"])

DA_RGB = xr.DataArray(np.array(1).reshape((1, 1, 1)), dims=["red", "green", "blue"])

DS_R = xr.Dataset({"DA_R": DA_R})
DS_R_2 = xr.Dataset({"DA_R": DA_R, "DA_R_2": DA_R})
DS_G = xr.Dataset({"DA_G": DA_G})

DS_RG = xr.Dataset({"DA_RG": DA_RG})
DS_GB = xr.Dataset({"DA_GB": DA_GB})

DS_RG_RG = xr.Dataset({"DA_RG": DA_RG, "DA_RG_2": DA_RG})
DS_RG_R = xr.Dataset({"DA_RG": DA_RG, "DA_R": DA_R})

DS_RGB_GB = xr.Dataset({"DA_RGB": DA_RGB, "DA_GB": DA_GB})

DS_RGB = xr.Dataset({"DA_RGB": DA_RGB})


# Test data for round_values
DA_ROUND = xr.DataArray(
    data=[[0.2, 1.323, 10.412], [-3.914, 0.001, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

EXP_ROUND1 = DA_ROUND.copy()

EXP_ROUND2 = xr.DataArray(  # round to nearest .2
    data=[[0.2, 1.4, 10.4], [-4.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

EXP_ROUND3 = xr.DataArray(  # round to nearest 5
    data=[[0.0, 0.0, 10.0], [-5.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords={"x": [0, 1, 2], "station": [1001, 1002]},
)

# test data for `custom_round`
ROUND_VALUES = [10.0, np.nan, 2, 0, 50]
DA_TO_ROUND = xr.DataArray(
    data=[[-100.0, 1, np.nan], [30, 68, 8]],
    dims=["stn", "date"],
    coords={"stn": [1001, 1002], "date": ["01", "02", "03"]},
)
EXP_ROUND = xr.DataArray(
    data=[[0.0, 0, np.nan], [10, 50, 10]],
    dims=["stn", "date"],
    coords={"stn": [1001, 1002], "date": ["01", "02", "03"]},
)
