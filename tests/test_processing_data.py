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
