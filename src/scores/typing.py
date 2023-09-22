'''
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
'''

import typing
import pandas
import xarray as xr

# Flexible Dimension Types should be used for preserve_dims and reduce_dims in all
# cases across the repository
FlexibleDimensionTypes = typing.Optional[typing.Union[str, typing.List[str], typing.Set[str]]]

# Xarraylike data types should be used for all forecast, observed and weights
# However currently some are specified as DataArray only
XarrayLike = typing.Union[xr.DataArray, xr.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in

FlexibleArrayType = typing.Union[XarrayLike, pandas.Series]