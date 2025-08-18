"""
Methods for the probability integral transform (PIT) class.

Reserved dimension names:
- 'uniform_endpoint', 'pit_x_values'
"""

import numpy as np
import xarray as xr

from scores.typing import XarrayLike


def _pit_values_for_ensemble(fcst: XarrayLike, obs: XarrayLike, ens_member_dim: str) -> XarrayLike:
    """
    For each forecast case in the form of an ensemble, the PIT value of the ensemble for
    the corresponding observation is a uniform distribution over the closed interval
    [lower,upper], where
        lower = (count of ensemble members strictly less than the observation) / n
        upper = (cont of ensemble members not exceeding the observation) / n
        n = size of the ensemble.

    Returns an array of [lower,upper] values in the dimension 'uniform_endpoint'.

    Args:
        fcst: array of forecast values, including dimension `ens_member_dim`
        obs: array of forecast values, excluding `ens_member_dim`
        ens_member_dim: name of the ensemble member dimension in `fcst`

    Returns:
        array of PIT values in the form [lower,upper], with dimensions
        'uniform_endpoint', all dimensions in `obs` and all dimensions in `fcst`
        excluding `ens_member_dim`.
    """
    ensemble_size = fcst.count(ens_member_dim).where(obs.notnull())
    pit_lower = (fcst < obs).sum(ens_member_dim) / ensemble_size
    pit_upper = (fcst <= obs).sum(ens_member_dim) / ensemble_size

    pit_lower = pit_lower.assign_coords(uniform_endpoint="lower").expand_dims("uniform_endpoint")
    pit_upper = pit_upper.assign_coords(uniform_endpoint="upper").expand_dims("uniform_endpoint")

    return xr.concat([pit_lower, pit_upper], "uniform_endpoint")


def _get_pit_x_values(pit_values: XarrayLike) -> xr.DataArray:
    """
    Returns a data array of consisting of exactly those x-axis values needed for
    constructing a unifornm PIT probability plot for the given array of `pit_values`.

    Args:
        pit_values: output from `_pit_values_for_ensemble`

    Returns:
        xarray data array of x-axis values, indexed by the same values on the dimension
        'pit_x_values'
    """
    if isinstance(pit_values, xr.Dataset):
        pit_values = pit_values.to_dataarray()

    x_values = np.unique(pit_values)
    x_values = np.unique(np.concatenate((np.array([0, 1]), x_values)))
    x_values = x_values[~np.isnan(x_values)]
    x_values = xr.DataArray(data=x_values, dims=["pit_x_values"], coords={"pit_x_values": x_values})
    return x_values
