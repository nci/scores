"""
This module contains the types allowed in the dataframe API.
"""

from typing import Union

from narwhals.typing import IntoSeries

# Any data series supported by Narwhals (e.g. pandas, Polars, PyArrow), or a scalar to
# compare a series against. Scalars are accepted for parity with `scores.pandas`.
SeriesType = Union[IntoSeries, float]
