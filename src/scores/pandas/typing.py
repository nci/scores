"""
This module contains the types allowed in the pandas api.
"""

from typing import Type, Union

import pandas as pd

# Pandas Available Types
PandasType = Type[pd.Series]

PandasFlexibleSeries = Union[PandasType, str]
PossibleDataFrame = Union[None, pd.DataFrame]
