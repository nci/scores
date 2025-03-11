"""
Test data for scores.utils
"""

import numpy as np
import xarray as xr

#####
# Test data for scores.utils.check_dims

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

DA_WEIGHTS_GOOD = xr.DataArray([1, 2, 3, 4])
DA_WEIGHTS_GOOD_SOME_NAN = xr.DataArray([np.nan, 1, np.nan, 3])
DA_WEIGHTS_GOOD_SOME_ZERO = xr.DataArray([0, 2, 0, 4])
DA_WEIGHTS_BAD_ALL_NAN = xr.DataArray([np.nan, np.nan, np.nan, np.nan])
DA_WEIGHTS_BAD_ALL_ZERO = xr.DataArray([0, 0, 0, 0])
DA_WEIGHTS_BAD_ANY_NEG = xr.DataArray([1, 2, -1, 4])

DS_WEIGHTS_GOOD = xr.Dataset(dict(good1=DA_WEIGHTS_GOOD, good2=DA_WEIGHTS_GOOD))
DS_WEIGHTS_GOOD_SOME_NAN = xr.Dataset(dict(good1=DA_WEIGHTS_GOOD_SOME_NAN, good2=DA_WEIGHTS_GOOD))
DS_WEIGHTS_GOOD_SOME_ZERO = xr.Dataset(dict(good1=DA_WEIGHTS_GOOD_SOME_ZERO, good2=DA_WEIGHTS_GOOD))
DS_WEIGHTS_BAD_ALL_NAN = xr.Dataset(dict(bad=DA_WEIGHTS_BAD_ALL_NAN, good=DA_WEIGHTS_GOOD))
DS_WEIGHTS_BAD_ALL_ZERO = xr.Dataset(dict(bad=DA_WEIGHTS_BAD_ALL_ZERO, good=DA_WEIGHTS_GOOD))
DS_WEIGHTS_BAD_ANY_NEG = xr.Dataset(dict(bad=DA_WEIGHTS_BAD_ANY_NEG, good=DA_WEIGHTS_GOOD))
