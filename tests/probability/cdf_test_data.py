"""
Generation of the test data used to test scores.probability.functions.cdfs
"""

import xarray as xr
from numpy import nan

###
# Test data for cdfvalues_from_probs
###
DA_CDF_FROM_PROBS = xr.DataArray(  # nonexceedance probabilities
    data=[0.4, 0.9, 0.6, 0.8, nan], dims=["x"], coords={"x": [0, 10, 5, 6, 11]}
)

EXP_CDF_FROM_PROBS = xr.DataArray(
    data=[0.4, 0.6, 0.8, 0.9, nan],
    dims=["x"],
    coords={"x": [0, 5, 6, 10, 11]},
    name="cdf",
)


""""
Test data for cdfvalues_from_quantiles
"""
DA_CDF_FROM_QUANTS1 = xr.DataArray(  # example with no point mass
    data=[[8, 11, 15, nan], [1, 3, 13, nan]],
    dims=["station", "q_level"],
    coords={"station": [1001, 1002], "q_level": [0.2, 0.5, 0.9, 0.95]},
)

EXP_CDF_FROM_QUANTS1 = xr.DataArray(
    data=[
        [nan, nan, 0.2, 0.5, nan, 0.9],
        [0.2, 0.5, nan, nan, 0.9, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [1, 3, 8, 11, 13, 15]},
    name="cdf",
)

DA_CDF_FROM_QUANTS2 = xr.DataArray(  # point mass at x=13
    data=[[8, 8, 15], [1, 3, 13]],
    dims=["station", "q_level"],
    coords={"station": [1001, 1002], "q_level": [0.2, 0.5, 0.9]},
)

EXP_CDF_FROM_QUANTS2 = xr.DataArray(
    data=[[nan, nan, 0.35, nan, 0.9], [0.2, 0.5, nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [1, 3, 8, 13, 15]},
    name="cdf",
)

DA_CDF_FROM_QUANTS3 = xr.DataArray(  # quantile levels outside interval (0,1)
    data=[[8, 8, 15], [1, 3, 13]],
    dims=["station", "q_level"],
    coords={"station": [1001, 1002], "q_level": [-0.2, 0.5, 0.9]},
)

DA_CDF_FROM_QUANTS4 = xr.DataArray(  # quantile levels outside interval (0,1)
    data=[[8, 8, 15], [1, 3, 13]],
    dims=["station", "q_level"],
    coords={"station": [1001, 1002], "q_level": [0.2, 0.5, 1.0]},
)

DA_WITHIN_BOUNDS1 = xr.DataArray(  # all nans
    data=[[nan, nan, nan], [nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0.2, 0.5, 1.0]},
)

DA_WITHIN_BOUNDS2 = xr.DataArray(
    data=[[nan, 0, 0.9], [0.5, nan, 1]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0.2, 0.5, 1.0]},
)

DA_WITHIN_BOUNDS3 = xr.DataArray(
    data=[[nan, 0, -0.9], [0.5, nan, 1]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [0.2, 0.5, 1.0]},
)


DA_FILL_CDF1 = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.4, nan, nan, nan, nan, nan],
        [nan, 0.2, nan, 0.8, 0.9, nan, nan],
        [0, 0.1, 0.6, 0.8, 1, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

EXP_FILL_CDF1A = xr.DataArray(  # linear fill
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan, nan, nan],
        [0, 0.2, 0.5, 0.8, 0.9, 1, 1],
        [0, 0.1, 0.6, 0.8, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

EXP_FILL_CDF1B = xr.DataArray(  # step fill
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        [0, 0.2, 0.2, 0.8, 0.9, 0.9, 0.9],
        [0, 0.1, 0.6, 0.8, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

EXP_FILL_CDF1C = xr.DataArray(  # step fill, min_nonnan=2
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan, nan, nan],
        [0, 0.2, 0.2, 0.8, 0.9, 0.9, 0.9],
        [0, 0.1, 0.6, 0.8, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

EXP_FILL_CDF1D = xr.DataArray(  # forward fill
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        [0.2, 0.2, 0.2, 0.8, 0.9, 0.9, 0.9],
        [0, 0.1, 0.6, 0.8, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

EXP_FILL_CDF1E = xr.DataArray(  # backward fill
    data=[
        [nan, nan, nan, nan, nan, nan, nan],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        [0.2, 0.2, 0.8, 0.8, 0.9, 0.9, 0.9],
        [0, 0.1, 0.6, 0.8, 1, 1, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4, 5, 6]},
)

DA_CDFFPF_PROBS = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan],
        [nan, 0.5, nan, 0.8, nan],
        [0, 0.2, 0.2, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003], "x": [0, 1, 2, 3, 4]},
)

DA_CDFFPF_PROBS_REORDER = xr.DataArray(  # same as above but reorderd coords
    data=[
        [nan, nan, nan, nan, nan],
        [nan, 0.5, nan, nan, 0.8],
        [0, 0.2, 0.2, 0.9, 0.8],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003], "x": [0, 1, 2, 4, 3]},
)

DA_CDFFPF_QUANTS = xr.DataArray(
    data=[
        [nan, nan, nan],
        [0.2, 1, 2.773],
        [2.667, 2.8, 2.901],
    ],
    dims=["station", "level"],
    coords={"station": [1001, 1002, 1003], "level": [0.25, 0.5, 0.75]},
)

DA_CDFFPF_QUANTS_REORDER = xr.DataArray(  # same as above but reorderd coords
    data=[
        [nan, nan, nan],
        [1, 0.2, 2.773],
        [2.8, 2.667, 2.901],
    ],
    dims=["station", "level"],
    coords={"station": [1001, 1002, 1003], "level": [0.5, 0.25, 0.75]},
)

EXP_CDFFPF_NOFILL_NOROUND = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.25, 0.5, nan, nan, 0.75, nan, nan, 0.8, nan],
        [0, nan, 0.2, 0.2, 0.25, nan, 0.5, 0.75, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 2.901, 3, 4],
    },
    name="cdf",
)

EXP_CDFFPF_NOFILL_ROUND = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.25, 0.5, nan, nan, 0.75, 0.8, nan],
        [0, nan, 0.2, 0.2, 0.25, 0.5, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.6, 2.8, 3, 4],
    },
    name="cdf",
)

DA_CDFFPF_QUANTS1 = xr.DataArray(
    data=[
        [nan, nan, nan],
        [0.2, 1, 2.773],
        [2.667, 2.8, 5],  # last entry differs from DA_CDFFPF_QUANTS
    ],
    dims=["station", "level"],
    coords={"station": [1001, 1002, 1003], "level": [0.25, 0.5, 0.75]},
)

EXP_CDFFPF_STEPFILL_NOROUND1 = xr.DataArray(  # minnan = 2
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [0, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.8, 0.8],
        [0, 0, 0.2, 0.2, 0.25, 0.25, 0.5, 0.75, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 2.901, 3, 4],
    },
    name="cdf",
)

EXP_CDFFPF_STEPFILL_NOROUND2 = xr.DataArray(  # minnan = 5
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [0, 0, 0.2, 0.2, 0.25, 0.25, 0.5, 0.75, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 2.901, 3, 4],
    },
    name="cdf",
)

EXP_CDFFPF_PRIORITISE_PROBS1 = xr.DataArray(  # inconsistent but not conflicting
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.25, 0.5, nan, nan, 0.75, nan, 0.8, nan, nan],
        [0, nan, 0.2, 0.2, 0.25, nan, 0.5, 0.8, 0.9, 0.75],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 3, 4, 5],
    },
    name="cdf",
)

DA_CDFFPF_QUANTS2 = xr.DataArray(
    data=[
        [nan, nan, nan],
        [0.2, 1, 2.773],
        [2.667, 2.8, 4],  # last entry differs from DA_CDFFPF_QUANTS
    ],
    dims=["station", "level"],
    coords={"station": [1001, 1002, 1003], "level": [0.25, 0.5, 0.75]},
)

EXP_CDFFPF_PRIORITISE_PROBS2 = xr.DataArray(  # conflicting, last entry of quants
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.25, 0.5, nan, nan, 0.75, nan, 0.8, nan],
        [0, nan, 0.2, 0.2, 0.25, nan, 0.5, 0.8, 0.9],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 3, 4],
    },
    name="cdf",
)

EXP_CDFFPF_PRIORITISE_QUANTS2 = xr.DataArray(  # conflicting, last entry of quants
    data=[
        [nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, 0.25, 0.5, nan, nan, 0.75, nan, 0.8, nan],
        [0, nan, 0.2, 0.2, 0.25, nan, 0.5, 0.8, 0.75],
    ],
    dims=["station", "x"],
    coords={
        "station": [1001, 1002, 1003],
        "x": [0, 0.2, 1, 2, 2.667, 2.773, 2.8, 3, 4],
    },
    name="cdf",
)

DA_CDF_ENVELOPE1 = xr.DataArray(
    data=[
        [0.2, 0.4, 0.5, 0.8, 1],
        [0.2, 0.1, 0.7, 0.4, 0.9],
        [0, 0.6, nan, 0.7, 0.5],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4]},
)

EXP_CDF_ENVELOPE1 = xr.DataArray(
    data=[
        [
            [0.2, 0.4, 0.5, 0.8, 1],
            [0.2, 0.1, 0.7, 0.4, 0.9],
            [0, 0.6, nan, 0.7, 0.5],
            [nan, nan, nan, nan, nan],
        ],
        [
            [0.2, 0.4, 0.5, 0.8, 1],
            [0.2, 0.2, 0.7, 0.7, 0.9],
            [0, 0.6, nan, 0.7, 0.7],
            [nan, nan, nan, nan, nan],
        ],
        [
            [0.2, 0.4, 0.5, 0.8, 1],
            [0.1, 0.1, 0.4, 0.4, 0.9],
            [0, 0.5, nan, 0.5, 0.5],
            [nan, nan, nan, nan, nan],
        ],
    ],
    dims=["cdf_type", "station", "x"],
    coords={
        "cdf_type": ["original", "upper", "lower"],
        "station": [1001, 1002, 1003, 1004],
        "x": [0, 1, 2, 3, 4],
    },
)

DA_CDF_ENVELOPE2 = xr.DataArray(  # same as DA_CDF_ENVELOPE1, but with coords unordered
    data=[
        [0.2, 0.4, 0.5, 1, 0.8],
        [0.2, 0.1, 0.7, 0.9, 0.4],
        [0, 0.6, nan, 0.5, 0.7],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 4, 3]},
)

DA_PROPNAN = xr.DataArray(
    data=[
        [0, 0, nan, 1.0],
        [0, 1.0, 1.0, nan],
        [0, 0, 1.0, 1.0],
    ],
    dims=["x", "y"],
    coords={"x": [1001, 1002, 1003], "y": [0, 1, 2, 4]},
)

EXP_PROPNAN_X = xr.DataArray(
    data=[
        [0, 0, nan, nan],
        [0, 1.0, nan, nan],
        [0, 0, nan, nan],
    ],
    dims=["x", "y"],
    coords={"x": [1001, 1002, 1003], "y": [0, 1, 2, 4]},
)

EXP_PROPNAN_Y = xr.DataArray(
    data=[
        [nan, nan, nan, nan],
        [nan, nan, nan, nan],
        [0, 0, 1.0, 1.0],
    ],
    dims=["x", "y"],
    coords={"x": [1001, 1002, 1003], "y": [0, 1, 2, 4]},
)

DA_DECREASING_CDFS1 = xr.DataArray(
    data=[
        [0.2, 0.4, 0.5, 0.8, 1],
        [0.2, 0.1, 0.7, 0.4, 0.9],
        [0, 0.6, 0.59, 0.7, 0.69],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4]},
)

EXP_DECREASING_CDFS1A = xr.DataArray(
    data=[False, True, True, False],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004]},
)

EXP_DECREASING_CDFS1B = xr.DataArray(
    data=[False, True, False, False],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004]},
)

DA_NAN_DECREASING_CDFS = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan],
        [0.2, 0.4, 0.5, 0.8, 1],
        [0, 0.6, 0.59, 0.7, 0.69],
        [0, 0.6, 0.59, 0.7, 0.68],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4]},
)

EXP_NAN_DECREASING_CDFS = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan],
        [0.2, 0.4, 0.5, 0.8, 1],
        [0, 0.6, 0.59, 0.7, 0.69],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004], "x": [0, 1, 2, 3, 4]},
)

DA_NAN_DECREASING_CDFS2 = xr.DataArray(
    data=[[0.2, 0.4, 0.5, 0.8, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [0, 1, 20, 3, 4]},
)

DA_OBSERVED_CDF = xr.DataArray(
    data=[[0, 0.15, 10], [0.24, 0, nan]],
    dims=["station", "date"],
    coords={"station": [1001, 1002], "date": [1, 2, 3]},
)

EXP_OBSERVED_CDF1 = xr.DataArray(  # no rounding
    data=[
        [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]],
        [[0, 0, 1, 1], [1, 1, 1, 1], [nan, nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={"x": [0, 0.15, 0.24, 10], "station": [1001, 1002], "date": [1, 2, 3]},
)

EXP_OBSERVED_CDF2 = xr.DataArray(  # round to nearest 0.2
    data=[
        [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
        [[0, 1, 1], [1, 1, 1], [nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={"x": [0, 0.2, 10], "station": [1001, 1002], "date": [1, 2, 3]},
)

EXP_OBSERVED_CDF3 = xr.DataArray(  # include [5] in thresholds
    data=[
        [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [nan, nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={"x": [0, 0.2, 5, 10], "station": [1001, 1002], "date": [1, 2, 3]},
)

EXP_OBSERVED_CDF4 = xr.DataArray(  # include [0, 5, 15] in thresholds, no obs
    data=[
        [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
        [[0, 1, 1], [1, 1, 1], [nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={"x": [0, 5, 15], "station": [1001, 1002], "date": [1, 2, 3]},
)

DA_OBSERVED_CDF2 = xr.DataArray(
    data=[[nan, nan, nan], [nan, nan, nan]],
    dims=["station", "date"],
    coords={"station": [1001, 1002], "date": [1, 2, 3]},
)

DA_ADD_THRESHOLDS = xr.DataArray(
    data=[[[0.2, 0.4, 1, 1], [0, 0, 0.6, 1]]],
    dims=["date", "station", "x"],
    coords={"station": [1001, 1002], "date": ["2020-01-01"], "x": [0, 0.2, 0.5, 1]},
)

EXP_ADD_THRESHOLDS1 = xr.DataArray(
    data=[[[0.2, 0.4, 1, 1, 1], [0, 0, 0.6, 0.8, 1]]],
    dims=["date", "station", "x"],
    coords={
        "station": [1001, 1002],
        "date": ["2020-01-01"],
        "x": [0, 0.2, 0.5, 0.75, 1],
    },
)

EXP_ADD_THRESHOLDS2 = xr.DataArray(
    data=[[[0.2, 0.4, 1, nan, 1], [0, 0, 0.6, nan, 1]]],
    dims=["date", "station", "x"],
    coords={
        "station": [1001, 1002],
        "date": ["2020-01-01"],
        "x": [0, 0.2, 0.5, 0.75, 1],
    },
)

EXP_ECDF = xr.DataArray(
    data=[0.4, 0.6, 0.7, 0.8, 0.9, 1],
    dims=["x"],
    coords={"x": [0, 0.2, 0.3, 0.7, 1.5, 4.2]},
).rename("cdf")


DA_CDF_EXPECTEDVALUE = xr.DataArray(
    data=[
        [0, 0.5, 1, 1],
        [nan, 0.5, 1, 1],
        [0, 0, 0.5, 1],
        [0, 0, nan, 1],
        [nan, nan, nan, nan],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005], "x": [-1, 0, 1, 2]},
)

EXP_EXPECTEDVALUE = xr.DataArray(
    data=[0.25, 0.25, 1.0, nan, nan],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005]},
)

DA_SUPPORT1 = xr.DataArray(
    data=[[0, 0, 0.2, 0.5, 1, 1], [0, 0, nan, 0.5, 1, 1]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [-2, -1, 0, 1, 2, 3]},
)

DA_SUPPORT2 = xr.DataArray(
    data=[[0, 0, nan, 0.2, 0.5, 1, nan, 1], [nan, nan, nan, nan, nan, nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001, 1002], "x": [-2, -1, 0, 1, 2, 3, 4, 5]},
)

DA_SUPPORT3 = xr.DataArray(
    data=[[nan, nan, nan, nan, nan, nan, nan, nan]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [-2, -1, 0, 1, 2, 3, 4, 5]},
)


DA_CDF_CHECK_SUPP = xr.DataArray(
    data=[[0.2, 0.5, 1, 1]],
    dims=["station", "x"],
    coords={"station": [1001], "x": [-1, 0, 1, 2]},
)

DA_FUNCVALS = xr.DataArray(
    data=[
        [0, 0.25, 0.5, 1],
        [0.5, 0.5, 0.5, 1],
        [0.5, 0.5, nan, 1],
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003], "x": [0, 0.5, 1, 2]},
)

EXP_VARCDF = xr.DataArray(
    data=[2 / 3, 1 / 4 + 2 - 4 / 3 - 1 / 2 + 1 / 6, nan],
    dims=["station"],
    coords={"station": [1001, 1002, 1003]},
)

DA_CDF_VARIANCE = xr.DataArray(
    data=[
        [0, 0, 0.5, 1, 1],  # std uniform dist
        [0, 0, 0.25, 0.5, 1],  # uniform dist on [0,2]
        [0, 0, 0, 1, 1],  # uniform dist on [0.5,1]
        [nan, 0, 0.5, 1, 1],  # NaN outside support of CDF
        [0, 0, nan, 1, 1],  # NaN inside support of CDF
        [nan, nan, nan, nan, nan],  # all NaN
    ],
    dims=["station", "x"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006], "x": [-3, 0, 0.5, 1, 2]},
)

EXP_VARIANCE = xr.DataArray(
    data=[1 / 12, 4 / 12, 1 / (4 * 12), 1 / 12, nan, nan],
    dims=["station"],
    coords={"station": [1001, 1002, 1003, 1004, 1005, 1006]},
)

DA_CDF_QUANTILES = xr.DataArray(
    data=[
        [0, 0.1, 0.3, 0.4, 0.7, 1.0, 1],
        [nan, nan, nan, nan, nan, nan, nan],
        [0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0],  # quantile not unique
        [0.0, 0.3, 0.9, 1.0, 1.0, 1.0, 1.0],  # nearest not unique
        [0.0, 0.5, nan, 0.65, 0.8, 0.9, 0.95],
    ],
    dims=["date", "x"],
    coords={
        "date": ["01", "02", "03", "04", "05"],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
)

EXP_QUANTILES1 = xr.DataArray(
    data=[4.0, nan, 1.0, 1.0, 3.0],
    dims=["date"],
    coords={"date": ["01", "02", "03", "04", "05"]},
    attrs={"quantile_level": 0.6},
).rename("quantile")

EXP_QUANTILES2 = xr.DataArray(
    data=[3.5, nan, 1.0, 1.5, 2.5],
    dims=["date"],
    coords={"date": ["01", "02", "03", "04", "05"]},
    attrs={"quantile_level": 0.6},
).rename("quantile")


DA_CDF_PIW = xr.DataArray(
    data=[
        [
            [0, 0.1, 0.2, 0.4, 0.7, 0.8, 1],  # exact answer
            [nan, nan, nan, nan, nan, nan, nan],
        ],
        [
            [0, 0.2, 0.2, 0.4, 0.7, 0.71, 0.9],  # approximated
            [0, 0, 0, 0.8, 0.2, 1, 1],  # decreasing cdf (clips)
        ],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [101, 102],
        "date": ["1", "2"],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
)

EXP_PIW1 = xr.DataArray(
    data=[[5.0 - 2.0, nan], [5.0 - 1.0, 0.0]],
    dims=["station", "date"],
    coords={"station": [101, 102], "date": ["1", "2"]},
    attrs={"lower_quantile_level": 0.2, "upper_quantile_level": 0.8},
).rename("prediction_interval_width")

EXP_PIW2 = xr.DataArray(
    data=[3.0, 2.0],
    dims=["station"],
    coords={"station": [101, 102]},
    attrs={"lower_quantile_level": 0.2, "upper_quantile_level": 0.8},
).rename("prediction_interval_width")

EXP_PIW3 = xr.DataArray(
    data=(3.0 + 4.0 + 0.0) / 3,
    attrs={"lower_quantile_level": 0.2, "upper_quantile_level": 0.8},
).rename("prediction_interval_width")

EXP_PIW4 = xr.DataArray(
    data=[[5.0 - 2.0, nan], [5.5 - 1.0, 0.0]],
    dims=["station", "date"],
    coords={"station": [101, 102], "date": ["1", "2"]},
    attrs={"lower_quantile_level": 0.2, "upper_quantile_level": 0.8},
).rename("prediction_interval_width")

DA_CDF_BROACAST1 = xr.DataArray(
    data=[[0.0, 0.5, 1.0, 1.0], [nan, nan, nan, nan], [0.0, nan, 0.0, 1.0]],
    dims=["station", "x"],
    coords={"station": [101, 102, 103], "x": [0.0, 1.0, 2.0, 3.0]},
)

DA_CDF_BROACAST2 = xr.DataArray(
    data=[
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        [[0.0, 0.5, 1.0, 1.0], [0.0, 0.2, 1.0, 1.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 1.0, 1.0]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [100, 101, 102], "date": ["1", "2"], "x": [0.0, 1.0, 2.0, 3.0]},
)

DA_CDF_BROACAST3 = xr.DataArray(
    data=[
        [[0.0, 0.5, 1.0, 1.0], [0.0, 0.2, nan, 0.4]],
    ],
    dims=["station", "date", "x"],
    coords={"station": [101], "date": ["1", "2"], "x": [0.0, 1.0, 1.5, 2.0]},
)

EXP_BROADCAST_T1 = xr.DataArray(
    data=[
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[0.0, 0.5, 0.75, 1.0, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [100, 101, 102, 103],
        "date": ["1", "2"],
        "x": [0.0, 1.0, 1.5, 2.0, 3.0],
    },
)

EXP_BROADCAST_T2 = xr.DataArray(
    data=[
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[0.0, 0.5, 0.75, 1.0, 1.0], [0.0, 0.2, 0.6, 1.0, 1.0]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [100, 101, 102, 103],
        "date": ["1", "2"],
        "x": [0.0, 1.0, 1.5, 2.0, 3.0],
    },
)

EXP_BROADCAST_T3 = xr.DataArray(
    data=[
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[0.0, 0.5, 1.0, 1.0, 1.0], [0.0, 0.2, 0.3, 0.4, 0.6]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
        [[nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan]],
    ],
    dims=["station", "date", "x"],
    coords={
        "station": [100, 101, 102, 103],
        "date": ["1", "2"],
        "x": [0.0, 1.0, 1.5, 2.0, 3.0],
    },
)


DA_SAMPLE = xr.DataArray(
    data=[
        [0.0, 0.0, 2.0, 5.0, 7.0],
        [0.0, 1.0, 1.0, nan, 5.0],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "member"],
    coords={"station": [100, 101, 102], "member": ["1", "2", "3", "4", "5"]},
)

EXP_ECDF1 = xr.DataArray(  # preserve station, thresholds=None
    data=[
        [2 / 5, 2 / 5, 3 / 5, 4 / 5, 5 / 5],
        [1 / 4, 3 / 4, 3 / 4, 4 / 4, 4 / 4],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "threshold"],
    coords={"station": [100, 101, 102], "threshold": [0.0, 1.0, 2.0, 5.0, 7.0]},
).rename("cdf")

EXP_ECDF2 = xr.DataArray(  # dims=None, thresholds=None
    data=[3 / 9, 5 / 9, 6 / 9, 8 / 9, 9 / 9],
    dims=["threshold"],
    coords={"threshold": [0.0, 1.0, 2.0, 5.0, 7.0]},
).rename("cdf")

EXP_ECDF3 = xr.DataArray(  # preserve station, thresholds=[2, 5, 10]
    data=[
        [3 / 5, 4 / 5, 5 / 5],
        [3 / 4, 4 / 4, 4 / 4],
        [nan, nan, nan],
    ],
    dims=["station", "threshold"],
    coords={"station": [100, 101, 102], "threshold": [2.0, 5.0, 10.0]},
).rename("cdf")

DA_NAN_SAMPLE = xr.DataArray(
    data=[
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
    ],
    dims=["station", "member"],
    coords={"station": [100, 101, 102], "member": ["1", "2", "3", "4", "5"]},
)

EXP_ECDF4 = xr.DataArray(  # preserve station
    data=[[nan], [nan], [nan]],
    dims=["station", "threshold"],
    coords={"station": [100, 101, 102], "threshold": [0.0]},
).rename("cdf")
