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


def _pit_cdfvalues_for_jumps(pit_values: XarrayLike, x_values: xr.DataArray) -> dict:
    """
    Gives the values F(x) where F is the CDF for a pit value,
    and x comes from `x_values`, given that the CDF jumps at x. This occurs precisely
    when upper == lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    It is assumed that `x_values` contains all the values in `pit_values`.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`
        x_values: xr.DataArray of x values, with dimension 'pit_x_values', output from
            `_get_pit_x_values`

    Returns:
        dictionary of cdf values, with keys 'left' and 'right',
        representing the left and right hand limits of the cdf values F(x).
        Each value in the dictionary is an xarray object representing limits of F(x),
        where x is represented by values in the 'pit_x_values' dimension.
    """
    lower_values = pit_values.sel(uniform_endpoint="lower").drop_vars("uniform_endpoint")
    upper_values = pit_values.sel(uniform_endpoint="upper").drop_vars("uniform_endpoint")
    # get the cases where jumps occur
    pit_jumps = upper_values.where(upper_values == lower_values)
    # pit_jumps, xs = xr.broadcast(pit_jumps, xs)
    cdf_left = xr.zeros_like(pit_jumps).where(x_values <= pit_jumps, 1).where(pit_jumps.notnull())
    cdf_right = xr.zeros_like(pit_jumps).where(x_values < pit_jumps, 1).where(pit_jumps.notnull())
    return {"left": cdf_left, "right": cdf_right}


def _pit_cdfvalues_for_unif(pit_values: XarrayLike, x_values: xr.DataArray) -> dict:
    """
    Gives the values F(x) where F is the CDF for a pit value,
    and x comes from `x_values`, given that the CDF is uniform. This occurs precisely
    when upper > lower in the [lower, upper] representation of the PIT value.
    If this condition fails, NaNs are returned.

    Left-hand and right-hand limits of F(x) are equal in this case.

    It is assumed that `x_values` containes all the values in `pit_values`.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`
        x_values: xr.DataArray of x values, with dimension 'pit_x_values', output from
            `_get_pit_x_values`

    Returns:
        An xarray object representing values of F(x), where x is represented by values
        in the 'pit_x_values' dimension.
    """
    lower_values = pit_values.sel(uniform_endpoint="lower").drop_vars("uniform_endpoint")
    upper_values = pit_values.sel(uniform_endpoint="upper").drop_vars("uniform_endpoint")
    # get the cases where the cdf is a uniform distribution over [a,b], a < b
    unif_cases = upper_values > lower_values
    # get the upper and lower values: these correspond to a, b
    pit_unif_upper = upper_values.where(unif_cases)  # the b
    pit_unif_lower = lower_values.where(unif_cases)  # the a
    # calculate the cdf of Unif(a,b) at the values in x
    pit_unif = (
        xr.zeros_like(pit_unif_upper)
        .where(x_values <= pit_unif_lower)  # 0 for x <= a
        .where(x_values < pit_unif_upper, 1)  # 1 for x >= b
        .interpolate_na("pit_x_values")  # interpolate in between
        .where(unif_cases)  # preserve nans where appropriate
    )
    return pit_unif


def _pit_cdfvalues(pit_values):
    """
    Calculates F(x) for each CDF F representing the PIT value for a forecast
    case. The x values used are all values where there is a non-linear change
    at least one F from the collection of Fs. Intermediate values of F can be recovered
    using linear interpolation.

    The values are output as a dictionary of two xarray object representing left hand
    and right hand limits of F at the point x.

    Args:
        pit_values: xarray object output from `_pit_values_for_ensemble`

    Returns:
        dictionary of cdf values, with keys 'left' and 'right',
        representing the left and right hand limits of the cdf values F(x).
        Each value in the dictionary is an xarray object representing limits of F(x),
        where x is represented by values in the 'pit_x_values' dimension.
    """
    # get the x values
    x_values = _get_pit_x_values(pit_values)

    # get cdf values where the pit cdf jumps
    cdf_at_jump_cases = _pit_cdfvalues_for_jumps(pit_values, x_values)
    # get the cdf values where the pit is uniform
    cdf_at_unif_cases = _pit_cdfvalues_for_unif(pit_values, x_values)
    # combine
    cdf_right = cdf_at_jump_cases["right"].combine_first(cdf_at_unif_cases)
    cdf_left = cdf_at_jump_cases["left"].combine_first(cdf_at_unif_cases)

    return {"left": cdf_left, "right": cdf_right}
