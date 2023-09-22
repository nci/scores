'''
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
'''

import typing
import pandas
import xarray as xr

# These type hint values are used for various standard arguments across the 
# scores repository to guaranteee consistency. These types *should* be used
# by developers and *should not* be deviated from in standard arguments

FlexibleDimensionTypes = typing.Union[str, typing.List[str], None]
XarrayLike = typing.Union[xr.DataArray, xr.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in

FlexibleArrayType = typing.Union[XarrayLike, pandas.Series]