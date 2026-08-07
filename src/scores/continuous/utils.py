from functools import wraps
from inspect import signature

import numpy as np
import xarray as xr

from scores.utils import check_weights, gather_dimensions


def _check_isinstance(*args, classes):
    return all(isinstance(arg, classes) for arg in args if arg is not None)


def assert_is_xarraylike(*args, same_types):
    if not same_types:
        if not _check_isinstance(*args, classes=(xr.Dataset, xr.DataArray)):
            raise TypeError(
                "Inputs are not of type `xr.Dataset` or `xr.DataArray`. "
                "Check the inputted `fcst`, `obs` and `weights` (if applicable)."
            )
    else:
        if not (_check_isinstance(*args, classes=xr.Dataset) or _check_isinstance(*args, classes=xr.DataArray)):
            raise TypeError(
                "Inputs must all be of type `xr.Dataset`, or must all be of type `xr.DataArray`. "
                "Check the inputted `fcst`, `obs` and `weights` (if applicable)."
            )


def _check_dims_exist_da(da, dims):
    return np.intersect1d(da.dims, tuple(dims)).size > 0


def _check_dims_exist_ds(ds, dims):
    return all(_check_dims_exist_da(ds[var], dims) for var in ds.data_vars)


def check_dims_exist(*args, dims):
    if len(list(dims)) == 0:
        return True

    checks = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, xr.Dataset):
            checks.append(_check_dims_exist_ds(arg, dims))
        else:
            checks.append(_check_dims_exist_da(arg, dims))

    return all(checks)


def get_full_signature(func, *args, **kwargs):
    sig = signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def validate_inputs_outputs(same_input_types=False, same_input_and_output_type=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inspect the function signature and bind all arguments
            all_args = get_full_signature(func, *args, **kwargs)

            fcst = all_args.pop("fcst", None)
            obs = all_args.pop("obs", None)
            weights = all_args.pop("weights", None)

            is_angular = all_args.pop("is_angular", False)
            include_components = all_args.pop("include_components", False)

            reduce_dims = all_args.pop("reduce_dims", None)
            preserve_dims = all_args.pop("preserve_dims", None)

            assert_is_xarraylike(fcst, obs, weights, same_types=same_input_types)
            if not isinstance(is_angular, bool):
                raise TypeError("`is_angular` must be boolean.")
            if not isinstance(include_components, bool):
                raise TypeError("`include_components` must be boolean.")

            check_weights(weights) if weights is not None else None

            gathered_dims = gather_dimensions(
                fcst_dims=fcst.dims,
                obs_dims=obs.dims,
                weights_dims=weights.dims if weights is not None else None,
                reduce_dims=reduce_dims,
                preserve_dims=preserve_dims,
            )

            assert check_dims_exist(fcst, obs, weights, dims=gathered_dims)

            out = func(*args, **kwargs)

            if same_input_and_output_type:
                assert type(fcst) is type(out)

            return out

        return wrapper

    return decorator
