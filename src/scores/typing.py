"""
This module contains various compound or union types which can be used across the codebase to ensure
a consistent approach to typing is handled.
"""

from collections.abc import Hashable, Iterable
from typing import Any, TypeAlias, TypeGuard, Union, cast

import pandas as pd
import xarray

# Dimension name. `xarray` recommends `str` for dimension names, however, it doesn't impose
# restrictions on generic hashables, so we also need to support hashables.
DimName: TypeAlias = Hashable

# `FlexibleDimensionTypes` should be used for `preserve_dims` and `reduce_dims` in all cases across
# the repository. NOTE: may be replaced by `DimNameCollection` down the track.
FlexibleDimensionTypes: TypeAlias = Iterable[DimName]

# `XarrayLike` data types should be used for all forecast, observed and weights. However, currently
# some are specified as DataArray only
XarrayLike: TypeAlias = Union[xarray.DataArray, xarray.Dataset]

# `FlexibleArrayType` *may* be used for various arguments across the scores repository, but are not
# establishing a standard or expectation beyond the function they are used in.
FlexibleArrayType: TypeAlias = Union[XarrayLike, pd.Series]

# Generic collection that supports both `DimName`, a single hashable and `FlexibleDimensionTypes`.
# A iterable collection of dimension names. Useful for applying functions that can support both
# single names or collections e.g. `gather_dimensions` which currently does these checks internally.
#
# NOTE: may replace `FlexibleDimensionTypes` down the line - this has more utility functions.
DimNameCollection: TypeAlias = Union[DimName, Iterable[DimName]]

# --- FIXME: add back typing helpers from test/__TOBEDELETED_nse_refactor_test.py ---

