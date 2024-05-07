"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""
from collections.abc import Hashable, Iterable
from typing import Union

import pandas as pd
import xarray as xr

# Flexible Dimension Types should be used for preserve_dims and reduce_dims in all
# cases across the repository
FlexibleDimensionTypes = Iterable[Hashable]

# Xarraylike data types should be used for all forecast, observed and weights
# However currently some are specified as DataArray only
XarrayLike = Union[xr.DataArray, xr.Dataset]

# These type hint values *may* be used for various arguments across the
# scores repository but are not establishing a standard or expectation beyond
# the function they are used in

FlexibleArrayType = Union[XarrayLike, pd.Series]
