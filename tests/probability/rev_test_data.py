"""
Contains test data for test_rev.py
"""

from collections import OrderedDict

import numpy as np
import xarray as xr

# Module-level test data ... Where relevant carried over exactly from Jive
# In some cases the data have been transposed from what was in Jive.
SCALAR_DA = xr.DataArray(np.array([0.5]), dims=("x",), coords={"x": [0]})
BINARY_DA = xr.DataArray([0, 1, 0], dims=["time"])
PROB_FCST_DA = xr.DataArray([0.2, 0.8, 0.1], dims=["time"])

DISCRETE_FCST_2X3X2_WITH_NAN_MISALIGNED = xr.DataArray(
    [
        [[1, np.nan, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0]],
    ],
    dims=["lead_day", "letter", "pet"],
    coords=OrderedDict(
        [
            ("lead_day", [0, 1]),
            ("letter", ["a", "CAT", "b", "c"]),
            ("pet", ["muffin", "balthazar", "morpheus"]),
        ]
    ),
)

DISCRETE_FCST_2X3X2_WITH_NAN = xr.DataArray(
    [[[1, np.nan, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [0, 0, 0], [1, 0, 0]]],
    dims=["lead_day", "letter", "pet"],
    coords=OrderedDict(
        [
            ("lead_day", [0, 1]),
            ("letter", ["a", "b", "c"]),
            ("pet", ["muffin", "balthazar", "morpheus"]),
        ]
    ),
)

DISCRETE_FCST_2X3X2X3_WITH_NAN = xr.DataArray(
    [
        [
            [[1, 1, 0], [np.nan, np.nan, np.nan], [1, 0, 0]],
            [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
            [[1, 0, 0], [1, 1, 1], [1, 0, 0]],
        ],
        [
            [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
        ],
    ],
    dims=["lead_day", "letter", "pet", "binary_threshold"],
    coords=OrderedDict(
        [
            ("lead_day", [0, 1]),
            ("letter", ["a", "b", "c"]),
            ("pet", ["muffin", "balthazar", "morpheus"]),
            ("binary_threshold", [0, 0.3, 1]),
        ]
    ),
)

DISCRETE_FCST_3X5_INT = xr.DataArray(
    [[1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [0, 0, 1, 0, 0]],
    dims=["letter", "pet"],
    coords=OrderedDict(
        [
            ("letter", ["a", "b", "c"]),
            ("pet", ["muffin", "balthazar", "morpheus", "rick", "dainty"]),
        ]
    ),
)


EXP_PREV_CASE0 = xr.Dataset(
    {
        "maximum": xr.DataArray(
            [[np.nan, 0, 1 / 3, 0.25, np.nan], [np.nan, 0, 0, 0, np.nan]],
            coords=[("lead_day", [0, 1]), ("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])],
        ),
        "threshold_0": xr.DataArray(
            [[np.nan, 0, 0, -2, np.nan], [np.nan, 0, 0, -3, np.nan]],
            coords=[("lead_day", [0, 1]), ("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])],
        ),
        "threshold_0_3": xr.DataArray(
            [
                [np.nan, -2 / 3, 1 / 3, -0.25, np.nan],
                [np.nan, -2.75, -0.5, -2.75, np.nan],
            ],
            coords=[("lead_day", [0, 1]), ("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])],
        ),
        "threshold_1": xr.DataArray(
            [[np.nan, -3, 0, 0.25, np.nan], [np.nan, -3, 0, 0, np.nan]],
            coords=[("lead_day", [0, 1]), ("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])],
        ),
    },
)

EXP_PREV_CASE2 = xr.Dataset(
    {
        "maximum": xr.DataArray(
            [np.nan, 0, 0, 0.125, np.nan],
            coords=[("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])],
        )
    },
)
EXP_PREV_CASE3 = xr.Dataset(
    {"threshold_0_3": xr.DataArray([[1 / 3], [-0.5]], coords=[("lead_day", [0, 1]), ("cost_loss_ratio", [0.5])])},
)

# Case 0
EXP_REV_CASE0 = xr.DataArray(
    [
        # lead_day = 0
        [
            [np.nan, 0.0, 0.0, -2.0, np.nan],  # binary_threshold = 0.0
            [np.nan, -2 / 3, 1 / 3, -0.25, np.nan],  # binary_threshold = 0.3
            [np.nan, -3.0, 0.0, 0.25, np.nan],  # binary_threshold = 1.0
        ],
        # lead_day = 1
        [
            [np.nan, 0.0, 0.0, -3.0, np.nan],
            [np.nan, -2.75, -0.5, -2.75, np.nan],
            [np.nan, -3.0, 0.0, 0.0, np.nan],
        ],
    ],
    dims=["lead_day", "binary_threshold", "cost_loss_ratio"],
    coords={
        "lead_day": [0, 1],
        "binary_threshold": [0.0, 0.3, 1.0],
        "cost_loss_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    },
)


# Case 2 (singleton lead_day = 0)
EXP_REV_CASE2 = xr.DataArray(
    [
        [np.nan, 0.0, 0.0, -2.5, np.nan],
        [np.nan, -13 / 7, -1 / 7, -1.5, np.nan],
        [np.nan, -3.0, 0.0, 0.125, np.nan],
    ],
    dims=["binary_threshold", "cost_loss_ratio"],
    coords={
        "binary_threshold": [0.0, 0.3, 1.0],
        "cost_loss_ratio": [0.0, 0.2, 0.5, 0.8, 1.0],
    },
)


EXP_REV_CASE3 = xr.DataArray(
    [np.nan, -13 / 7, -1 / 7, -1.5, np.nan],
    dims=["cost_loss_ratio"],
    coords=OrderedDict([("cost_loss_ratio", [0, 0.2, 0.5, 0.8, 1])]),
)

HIT_RATE_REV_LEADDAY = xr.DataArray(
    [[1, 1], [0.75, 0.25], [0.25, 0]],
    dims=["binary_threshold", "lead_day"],
    coords=OrderedDict([("binary_threshold", [0, 0.3, 1]), ("lead_day", [0, 1])]),
)

FALSE_ALARM_RATE_REV_LEADDAY = xr.DataArray(
    [[1, 1], [1 / 3, 0.75], [0, 0]],
    dims=["binary_threshold", "lead_day"],
    coords=OrderedDict([("binary_threshold", [0, 0.3, 1]), ("lead_day", [0, 1])]),
)

OBAR_REV_LEADDAY = xr.DataArray([4 / 7, 0.5], dims=["lead_day"], coords={"lead_day": [0, 1]})

HIT_RATE_REV_NONE = xr.DataArray(
    [1, 0.5, 1 / 8],
    dims=["binary_threshold"],
    coords={"binary_threshold": [0, 0.3, 1]},
)

FALSE_ALARM_RATE_REV_NONE = xr.DataArray(
    [1, 4 / 7, 0],
    dims=["binary_threshold"],
    coords={"binary_threshold": [0, 0.3, 1]},
)

OBAR_REV_NONE = xr.DataArray(8 / 15)

FCST_2X3X2_WITH_NAN = xr.DataArray(
    [
        [[0.4, np.nan, 0.2], [0.7, 0.4, 0.3], [0.2, 1, 0]],
        [[0.3, 0.8, 0.9], [0.1, 0.1, 0.1], [0.8, 0.2, 0.1]],
    ],
    coords=[
        ("lead_day", [0, 1]),
        ("letter", ["a", "b", "c"]),
        ("pet", ["muffin", "balthazar", "morpheus"]),
    ],
)

OBS_3X3_WITH_NAN = xr.DataArray(
    [[1, 0, 0], [0, 1, np.nan], [0, 1, 1]],
    coords=[("letter", ["a", "b", "c"]), ("pet", ["muffin", "balthazar", "morpheus"])],
)

FCST_2X3X2_WITH_NAN_MISALIGNED = xr.DataArray(
    [
        [[0.4, np.nan, 0.2], [0.2, 1, 0], [0.7, 0.4, 0.3], [0.2, 1, 0]],
        [[0.3, 0.8, 0.9], [0.2, 1, 0], [0.1, 0.1, 0.1], [0.8, 0.2, 0.1]],
    ],
    coords=[
        ("lead_day", [0, 1]),
        ("letter", ["a", "CAT", "b", "c"]),
        ("pet", ["muffin", "balthazar", "morpheus"]),
    ],
)

OBS_3X3_WITH_NAN_MISALIGNED = xr.DataArray(
    [[1, 0, 0, 1], [0, 1, np.nan, 0], [0, 1, 1, np.nan]],
    coords=[
        ("letter", ["a", "b", "c"]),
        ("pet", ["muffin", "balthazar", "morpheus", "rick"]),
    ],
)

OBS_3X5_INT = xr.DataArray(
    [[1, 0, 0, 1, 0], [1, 1, 1, 0, 0], [0, 1, 0, 1, 1]],
    dims=["letter", "pet"],
    coords=OrderedDict(
        [
            ("letter", ["a", "b", "c"]),
            ("pet", ["muffin", "balthazar", "morpheus", "rick", "dainty"]),
        ]
    ),
)
