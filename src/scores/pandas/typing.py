"""
This module contains the types allowed in the pandas api.
"""

from typing import Union

import numpy as np
import pandas as pd

# Pandas Available Types
PandasType = Union[pd.Series, pd.DataFrame]
PandasReturnType = Union[pd.Series, np.floating, np.integer]
