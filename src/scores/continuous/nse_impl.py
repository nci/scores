import functools
import warnings
from collections.abc import Hashable

import xarray as xr

import scores.continuous
from scores.processing import broadcast_and_match_nan
from scores.typing import (
    FlexibleDimensionTypes,
    XarrayLike,
)
from scores.utils import gather_dimensions

from .utils import validate_inputs_outputs


def merge_sizes(*ds) -> dict[Hashable, int]:
    """
    Merges the maps that contain the size (value) for each dimension (key)
    in the given ``xr.Dataset`` object(s).

    Args:
        *ds: Variadic argument of each of type ``xr.Dataset``
    """
    assert len(ds) > 0

    def _merge_single(
        acc_sizes: dict[Hashable, int],
        curr_ds: xr.Dataset | None,
    ) -> dict[Hashable, int]:
        """
        merges ``sizes`` attribute from each dataset. ``sizes`` is a
        mapping: dimension name -> length. In theory this should also be
        compatible with data arrays.
        """
        if curr_ds is None:
            return acc_sizes
        merged_sizes: dict[Hashable, int] = acc_sizes | dict(curr_ds.sizes)
        return merged_sizes

    ret_sizes: dict[Hashable, int] = functools.reduce(_merge_single, ds, {})

    return ret_sizes


@validate_inputs_outputs()
def nse(
    fcst: XarrayLike,
    obs: XarrayLike,
    *,
    reduce_dims: FlexibleDimensionTypes | None = None,
    preserve_dims: FlexibleDimensionTypes | None = None,
    weights: XarrayLike | None = None,
    is_angular: bool = False,
):
    fcst, obs = broadcast_and_match_nan(fcst, obs)

    gathered_dims: FlexibleDimensionTypes = gather_dimensions(
        fcst_dims=fcst.dims,
        obs_dims=obs.dims,
        weights_dims=weights.dims if weights is not None else None,
        reduce_dims=reduce_dims,
        preserve_dims=preserve_dims,
    )

    if len(list(gathered_dims)) == 0:
        raise ValueError("No dimensions to reduce. NSE requires at least one dimension to reduce")
    dim_sizes = merge_sizes(obs, fcst, weights)
    dim_has_more_than_one_obs = any(dim_sizes[k] > 1 for k in gathered_dims)

    if not dim_has_more_than_one_obs:
        raise ValueError("Do not have more than one observation.")

    fcst_error = scores.continuous.mse(
        fcst,
        obs,
        reduce_dims=gathered_dims,
        weights=weights,
        is_angular=is_angular,
    )
    obs_mean = obs.mean(dim=gathered_dims)
    obs_variance = scores.continuous.mse(
        obs_mean,
        obs,
        reduce_dims=gathered_dims,
        weights=weights,
        is_angular=is_angular,
    )

    is_zero = obs_variance == 0
    if isinstance(is_zero, xr.Dataset):
        has_zero = any(bool(v.any()) for v in is_zero.data_vars.values())
    else:
        has_zero = bool(is_zero.any())
    if has_zero:
        warnings.warn(
            "divide by zero encountered in NSE calculation",
            RuntimeWarning,
            stacklevel=2,
        )

    nse = 1.0 - (fcst_error / obs_variance)
    nse.name = "NSE"

    return nse
