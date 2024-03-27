"""
scores pandas utils
"""
from typing import Union

import pandas as pd

from scores.pandas.typing import PandasType


def split_dataframe(
    dataframe: Union[pd.DataFrame, None], *column_names: Union[PandasType, str]
) -> tuple[PandasType, ...]:
    """
    Split a `pd.DataFrame` into series's if provided, with columns as informed by the args.

    If `dataframe` not given, return `column_names` directly.

    Can support a mix of str or other elements in `column_names`.

    Examples:
        >>> fcst_series = pd.Series([1, 2])
        >>> obs_series  = pd.Series([2, 2])
        >>> split_dataframe(None, fcst_series, obs_series)
        (pd.Series([1, 2]), pd.Series([2, 2]))
        >>> pd_dataframe = pd.DataFrame({"fcst": fcst_series, "obs": obs_series})
        >>> split_dataframe(pd_dataframe, 'fcst', 'obs')
        (pd.Series([1, 2]), pd.Series([2, 2]))
        >>> split_dataframe(pd_dataframe, fcst_series, 'obs')

    """

    if dataframe is None:
        return column_names

    if not all(x in dataframe.columns if isinstance(x, str) else True for x in column_names):
        raise KeyError(f"Columns were missing from the `pd.DataFrame`. Could not find all of {column_names}.")

    return (dataframe[x] if isinstance(x, str) else x for x in column_names)
