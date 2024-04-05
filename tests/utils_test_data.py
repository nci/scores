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
