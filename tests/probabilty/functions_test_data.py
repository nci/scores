import numpy as np
import xarray as xr

DA_ROUND = xr.DataArray(
    data=[[0.2, 1.323, 10.412], [-3.914, 0.001, np.nan]],
    dims=["station", "x"],
    coords=dict(x=[0, 1, 2], station=[1001, 1002]),
)

EXP_ROUND1 = DA_ROUND.copy()

EXP_ROUND2 = xr.DataArray(  # round to nearest .2
    data=[[0.2, 1.4, 10.4], [-4.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords=dict(x=[0, 1, 2], station=[1001, 1002]),
)

EXP_ROUND3 = xr.DataArray(  # round to nearest 5
    data=[[0.0, 0.0, 10.0], [-5.0, 0.0, np.nan]],
    dims=["station", "x"],
    coords=dict(x=[0, 1, 2], station=[1001, 1002]),
)
