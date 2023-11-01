"""Data for ROC tests"""
import numpy as np
import xarray as xr

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

DISCRETE_FCST_2X3X2_WITH_NAN = xr.DataArray(
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
    coords=[
        ("lead_day", [0, 1]),
        ("letter", ["a", "b", "c"]),
        ("pet", ["muffin", "balthazar", "morpheus"]),
        ("threshold", [0, 0.3, 1]),
    ],
)
OBS_3X3_WITH_NAN = xr.DataArray(
    [[1, 0, 0], [0, 1, np.nan], [0, 1, 1]],
    coords=[("letter", ["a", "b", "c"]), ("pet", ["muffin", "balthazar", "morpheus"])],
)
OBS_3X3_WITH_NAN_MISALIGNED = xr.DataArray(
    [[1, 0, 0, 1], [0, 1, np.nan, 0], [0, 1, 1, np.nan]],
    coords=[
        ("letter", ["a", "b", "c"]),
        ("pet", ["muffin", "balthazar", "morpheus", "rick"]),
    ],
)

EXP_ROC_LEADDAY = xr.Dataset(
    {
        "POD": xr.DataArray(
            [[1, 0.75, 0.25], [1, 0.25, 0]],
            coords=[("lead_day", [0, 1]), ("threshold", [0, 0.3, 1])],
        ),
        "POFD": xr.DataArray(
            [[1, 1 / 3, 0], [1, 0.75, 0]],
            coords=[("lead_day", [0, 1]), ("threshold", [0, 0.3, 1])],
        ),
        "AUC": xr.DataArray([0.75, 0.25], coords=[("lead_day", [0, 1])]),
    }
)
LEAD_DAY_WEIGHTS = xr.DataArray([1, 2], dims="lead_day", coords={"lead_day": [0, 1]})

EXP_ROC_MULTI_DIMS = xr.Dataset(
    {
        "POD": xr.DataArray(
            [
                [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.5, 0.5]],
                [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            coords=[
                ("lead_day", [0, 1]),
                ("letter", ["a", "b", "c"]),
                ("threshold", [0, 0.3, 1]),
            ],
        ),
        "POFD": xr.DataArray(
            [
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            coords=[
                ("lead_day", [0, 1]),
                ("letter", ["a", "b", "c"]),
                ("threshold", [0, 0.3, 1]),
            ],
        ),
        "AUC": xr.DataArray(
            [[1, 0.5, 0.75], [0.5, 0.5, 0]],
            coords=[("lead_day", [0, 1]), ("letter", ["a", "b", "c"])],
        ),
    }
)
EXP_ROC_NONE = xr.Dataset(
    {
        "POD": xr.DataArray([1, 0.5, 1 / 8], dims=["threshold"], coords={"threshold": [0, 0.3, 1]}),
        "POFD": xr.DataArray([1, 4 / 7, 0], dims=["threshold"], coords={"threshold": [0, 0.3, 1]}),
        "AUC": xr.DataArray([0.5]).squeeze(),
    }
)

EXP_ROC_NONE_WEIGHTED = xr.Dataset(
    {
        "POD": xr.DataArray([1, 5 / 12, 1 / 12], dims=["threshold"], coords={"threshold": [0, 0.3, 1]}),
        "POFD": xr.DataArray([1, 0.6363636363636364, 0], dims=["threshold"], coords={"threshold": [0, 0.3, 1]}),
        "AUC": xr.DataArray([0.41666666666666663]).squeeze(),
    }
)
