"""
Import the functions from the implementations into the public API
"""

from functools import wraps

from scores import continuous
from scores.pandas.typing import PandasType
from scores.typing import FlexibleDimensionTypes


@wraps(continuous.mse)
def mse(
    fcst: PandasType,
    obs: PandasType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    angular: bool = False,
):
    return continuous.mse(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, angular=angular)


@wraps(continuous.rmse)
def rmse(
    fcst: PandasType,
    obs: PandasType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    angular: bool = False,
):
    return continuous.rmse(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, angular=angular)


@wraps(continuous.mae)
def mae(
    fcst: PandasType,
    obs: PandasType,
    reduce_dims: FlexibleDimensionTypes = None,
    preserve_dims: FlexibleDimensionTypes = None,
    angular: bool = False,
):
    return continuous.mae(fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims, angular=angular)
