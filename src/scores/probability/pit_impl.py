"""
Methods for the probability integral transform (PIT) class
"""

from scores.typing import XarrayLike

import xarray as xr

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

    pit_lower = pit_lower.assign_coords(uniform_endpoint='lower').expand_dims('uniform_endpoint')
    pit_upper = pit_upper.assign_coords(uniform_endpoint='upper').expand_dims('uniform_endpoint')

    return xr.concat([pit_lower, pit_upper], 'uniform_endpoint')
