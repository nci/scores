"""
Pandas scores decorators
"""

import functools
from typing import Union

import pandas as pd

DATAFRAME_DOCS = """
    Supports:
        df: pd.DataFrame with column names specified in other keywords.
            Set `{keywords}` to column names.
            E.g.
            >>> scores.pandas.METRIC_NAME(df, fcst = 'ForecastColumn', obs = 'ObsColumn')
"""


def split_dataframe(*series_keywords: str):
    """
    Allow for a `pd.DataFrame` to be provided, with `pd.Series` retrieved and passed to the underlying metric.

    Searches the keyword arguments given to the metric for `series_keywords` to retrieve columns.

    Expects the `pd.DataFrame` to either be the only argument, or under the `df` keyword.

    Examples:
        >>> @split_dataframe('fcst', 'obs')
        >>> def mean_error(fcst, obs):
                return (fcst - obs).mean()
        >>> fcst_series = pd.Series([1, 2])
        >>> obs_series  = pd.Series([2, 2])
        >>> mean_error(fcst_series, obs_series)
        -0.5
        >>> pd_dataframe = pd.DataFrame({"fcst": fcst_series, "obs": obs_series})
        >>> mean_error(pd_dataframe, fcst = 'fcst', obs = 'obs')
        -0.5
        >>> mean_error(fcst = 'fcst', obs = 'obs', df = pd_dataframe)
        -0.5


        >>> @split_dataframe('model', 'expected') # Atypical argument names
        >>> def difference(model, expected):
                return (model - expected)
        >>> difference(model = 'fcst', expected = 'obs', df = pd_dataframe)

    """

    def internal_function(func):
        func.__doc__ = str(getattr(func, "__doc__", "")) + DATAFRAME_DOCS.format(keywords="`, `".join(series_keywords))

        @functools.wraps(func)
        def decorated_function(*args, df: Union[pd.DataFrame, None] = None, **kwargs):
            # Check if DataFrame given
            if df is None:
                # Allow for only arg to be the DataFrame
                if len(args) == 1 and isinstance(args[0], pd.DataFrame):
                    df = args[0]
                elif len(args) > 0 and isinstance(args[0], pd.DataFrame):
                    raise KeyError(
                        f"A `pd.DataFrame` was supplied first, but a following arg was also given, provide {series_keywords} as keyword arguments."
                    )

                else:
                    return func(*args, **kwargs)

            # Check if all column keywords given
            if not all(x in kwargs for x in series_keywords):
                raise KeyError(
                    "A `pd.DataFrame` was passed but not all column keywords present. "
                    f"Ensure {series_keywords} are specified."
                )

            # Check if all columns included
            column_names = list(kwargs[key] for key in series_keywords)

            if not all(x in df.columns for x in column_names):
                raise KeyError(f"Columns were missing from the `pd.DataFrame`. Could not find all of {column_names}.")

            column_kwargs = {key: df[kwargs.pop(key)] for key in series_keywords}
            return func(**column_kwargs, **kwargs)

        return decorated_function

    return internal_function
