"""
Murphy score
"""
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import xarray as xr

from scores.processing import broadcast_and_match_nan
from scores.typing import FlexibleArrayType, FlexibleDimensionTypes
from scores.utils import gather_dimensions

QUANTILE = "quantile"
HUBER = "huber"
EXPECTILE = "expectile"
VALID_SCORING_FUNC_NAMES = [QUANTILE, HUBER, EXPECTILE]
SCORING_FUNC_DOCSTRING_PARAMS = {fun.upper(): fun for fun in VALID_SCORING_FUNC_NAMES}


def murphy_score(  # pylint: disable=R0914
    fcst: xr.DataArray,
    obs: xr.DataArray,
    thetas: Union[Sequence[float], xr.DataArray],
    *,  # Force keywords arguments to be keyword-only
    functional: Literal["quantile", "huber", "expectile"],
    alpha: float,
    huber_a: Optional[float] = None,
    decomposition: bool = False,
    reduce_dims: Optional[FlexibleDimensionTypes] = None,
    preserve_dims: Optional[FlexibleDimensionTypes] = None,
) -> xr.Dataset:
    """Returns the mean elementary score (Ehm et. al. 2016), also known as Murphy score,
    evaluated at decision thresholds specified by thetas. Optionally returns a decomposition
    of the score in terms of penalties for over- and under-forecasting.

    Select ``functional="quantile"`` and ``alpha=0.5`` for the median functional.
    Select ``functional="expectile"`` and ``alpha=0.5`` for the mean (i.e., expectation) functional.

    Consider using :py:func:`murphy_thetas` to generate thetas. If utilising dask, you may want
    to store ``thetas`` as a netCDF on disk and pass it in as an xarray object. Otherwise,
    very large objects may be created when ``fcst``, ``obs`` and ``thetas`` are broadcast
    together.


    Args:
        fcst: Forecast numerical values.
        obs: Observed numerical values.
        thetas: Theta thresholds.
        functional: The type of elementary scoring function, one of "quantile",
            "huber", or "expectile".
        alpha: Risk parameter (i.e. quantile or expectile level) for the functional. Must be between 0 and 1.
        huber_a: Huber transition parameter, used for "huber" functional only.
            Must be strictly greater than 0.
        decomposition: True to return penalty values due to under- and over-fcst
            as well as the total score, False to return total score only.
        reduce_dims: Optionally specify which dimensions to reduce when
            calculating the Murphy score. All other dimensions will be preserved. As a
            special case, 'all' will allow all dimensions to be reduced. Only one
            of ``reduce_dims`` and ``preserve_dims`` can be supplied. The default behaviour
            if neither are supplied is to reduce all dims.
        preserve_dims: Optionally specify which dimensions to preserve
            when calculating the Murphy score. All other dimensions will be reduced.
            As a special case, 'all' will allow all dimensions to be
            preserved. In this case, the result will be in the same
            shape/dimensionality as the forecast, and the errors will be
            the FIRM score at each point (i.e. single-value comparison
            against observed), and the forecast and observed dimensions
            must match precisely. Only one of ``reduce_dims`` and ``preserve_dims`` can be
            supplied. The default behaviour if neither are supplied is to reduce all dims.

    Returns:

        An xr.Dataset with dimensions based on the ``preserve_dims`` or ``reduce_dims`` arg
        as well as a "theta" dimension with values matching ``thetas`` input.

        If ``decomposition`` is False, the dataset's variables will contain 1 element
        "total".

        If ``decomposition`` is True, in addition to "total", it will have
        "underforecast" and "overforecast" data_vars.

    Raises:
        ValueError: If ``functional`` is not one of the expected functions.
        ValueError: If ``alpha`` is not strictly between 0 and 1.
        ValueError: If ``huber_a`` is not strictly greater than 0.

    References:

        For mean elementary score definitions, see:
            - Theorem 1 of Ehm, W., Gneiting, T., Jordan, A., & Krüger, F. (2016).
              Of quantiles and expectiles: Consistent scoring functions, Choquet
              representations and forecast rankings.
              *Journal of the Royal Statistical Society Series B: Statistical Methodology*,
              78(3), 505–562. https://doi.org/10.1111/rssb.12154
            - Theorem 5.3 of Taggart, R. J. (2022). Point forecasting and forecast evaluation
              with generalized Huber loss.
              *Electronic Journal of Statistics*, 16(1), 201-231.
              https://doi.org/10.1214/21-EJS1957

    """
    functional_lower = functional.lower()
    _check_murphy_inputs(alpha=alpha, functional=functional_lower, huber_a=huber_a)
    if isinstance(thetas, xr.DataArray):
        theta1 = thetas
    else:
        theta1 = xr.DataArray(data=thetas, dims=["theta"], coords={"theta": thetas})
    theta1, fcst1, obs1 = broadcast_and_match_nan(theta1, fcst, obs)  # type: ignore

    over, under = exposed_functions()[f"_{functional_lower}_elementary_score"](
        fcst1, obs1, theta1, alpha, huber_a=huber_a
    )
    # Align dimensions, this is required in cases such as when the station numbers
    # are not in the same order in `obs` and `fcst` to prevent an exception on the next
    # line that combines the scores
    over, under, fcst1, *_ = xr.align(over, under, fcst1)
    score = over.combine_first(under).fillna(0).where(~np.isnan(fcst1), np.nan)

    sources = [score]
    names = ["total"]
    if decomposition:
        over = over.fillna(0).where(~np.isnan(fcst1), np.nan)
        under = under.fillna(0).where(~np.isnan(fcst1), np.nan)
        sources += [under, over]
        names += ["underforecast", "overforecast"]
    for source, name in zip(sources, names):
        source.name = name
    result = xr.merge(sources)
    reduce_dims = gather_dimensions(fcst.dims, obs.dims, reduce_dims=reduce_dims, preserve_dims=preserve_dims)
    result = result.mean(dim=reduce_dims)
    return result


def _quantile_elementary_score(fcst: FlexibleArrayType, obs: FlexibleArrayType, theta, alpha, **_):
    """Return over and under forecast vs obs penalties relative to theta for {QUANTILE}."""
    zero_array = fcst * 0.0
    over = (zero_array + 1 - alpha).where((fcst > theta) & (obs <= theta))
    under = (zero_array + alpha).where((fcst <= theta) & (obs > theta))
    return over, under


def _huber_elementary_score(fcst: FlexibleArrayType, obs: FlexibleArrayType, theta, alpha, *, huber_a):
    """Return over and under forecast vs obs penalties relative to theta for {HUBER}."""
    zero_array = fcst * 0.0
    over = (1 - alpha) * np.minimum(theta - obs, zero_array + huber_a).where((fcst > theta) & (obs <= theta))
    under = alpha * np.minimum(obs - theta, zero_array + huber_a).where((fcst <= theta) & (obs > theta))
    return over, under


def _expectile_elementary_score(fcst: FlexibleArrayType, obs: FlexibleArrayType, theta, alpha, **_):
    """Return over and under forecast vs obs penalties relative to theta for {EXPECTILE}."""
    over = (1 - alpha) * np.abs(obs - theta).where((fcst > theta) & (obs <= theta))
    under = alpha * np.abs(obs - theta).where((fcst <= theta) & (obs > theta))
    return over, under


def _check_murphy_inputs(*, alpha=None, functional=None, huber_a=None, left_limit_delta=None):
    """Raise ValueError if the arguments have unexpected values."""
    if (alpha is not None) and not (0 < alpha < 1):  # pylint: disable=C0325
        err = f"alpha (={alpha}) argument for Murphy scoring function should be strictly " "between 0 and 1."
        raise ValueError(err)
    if (functional is not None) and (functional not in VALID_SCORING_FUNC_NAMES):
        err = (
            f"Functional option '{functional}' for Murphy scoring function is "
            f"unknown, should be one of {VALID_SCORING_FUNC_NAMES}."
        )
        raise ValueError(err)
    if (functional == HUBER) and ((huber_a is None) or (huber_a <= 0)):
        err = f"huber_a (={huber_a}) argument should be > 0 when functional='{HUBER}'."
        raise ValueError(err)
    if (left_limit_delta is not None) and (left_limit_delta < 0):
        err = f"left_limit_delta (={left_limit_delta}) argument should be >= 0."
        raise ValueError(err)


def murphy_thetas(
    forecasts: list[xr.DataArray],
    obs: xr.DataArray,
    functional: Literal["quantile", "huber", "expectile"],
    *,  # Force keywords arguments to be keyword-only
    huber_a: Optional[float] = None,
    left_limit_delta: Optional[float] = None,
) -> list[float]:
    """Return the decision thresholds (theta values) at which to evaluate
    elementary scores for the construction of Murphy diagrams.

    This function may generate a very large number of theta thresholds if the forecast
    and obs data arrays are large and their values have high precision, which may lead
    to long computational times for Murphy diagrams. To reduce the number of thetas,
    users can first round forecast and obs values to an appropriate resolution.

    Args:
        forecasts: Forecast values, one array per source.
        obs: Observed values.
        functional: The type of elementary scoring function, one of "quantile",
            "huber", or "expectile".
        huber_a: Huber transition parameter, used for "huber" functional only.
            Must be strictly greater than 0.
        left_limit_delta: Approximation of the left hand limit, used for "huber"
            and "expectile" functionals. Must be greater than or equal to 0.
            None will be treated as 0. Ideally, left_limit_delta should be
            small relative to the fcst and obs precision, and not greater than
            that precision.

    Returns:
        List[float]: theta thresholds to be used to compute murphy scores.

    Raises:
        ValueError: If ``functional`` is not one of the expected functions.
        ValueError: If ``huber_a`` is not strictly greater than 0.
        ValueError: If ``left_limit_delta`` is not greater than or equal to 0.

    Notes:
        For theta values at which to evaluate elementary scores, see

        - Corollary 2 (p.521) of Ehm, W., Gneiting, T., Jordan, A., & Krüger, F. (2016).
          Of quantiles and expectiles: Consistent scoring functions, Choquet
          representations and forecast rankings.
          *Journal of the Royal Statistical Society Series B: Statistical Methodology*,
          78(3), 505–562. https://doi.org/10.1111/rssb.12154
        - Corollary 5.6 of Taggart, R. J. (2022). Point forecasting and forecast evaluation
          with generalized Huber loss. *Electronic Journal of Statistics*, 16(1), 201-231.
          https://doi.org/10.1214/21-EJS1957

    """
    _check_murphy_inputs(functional=functional, huber_a=huber_a, left_limit_delta=left_limit_delta)
    if (left_limit_delta is None) and (functional in ["huber", "expectile"]):
        left_limit_delta = 0

    func = exposed_functions()[f"_{functional}_thetas"]
    result = func(
        forecasts=forecasts,
        obs=obs,
        huber_a=huber_a,
        left_limit_delta=left_limit_delta,
    )
    return result  # type: ignore


def _quantile_thetas(forecasts, obs, **_):
    """Return thetas for {QUANTILE} elementary scoring function."""
    ufcasts_and_uobs = np.unique(np.concatenate([*[x.values.flatten() for x in forecasts], obs.values.flatten()]))
    result = ufcasts_and_uobs[~np.isnan(ufcasts_and_uobs)]
    return list(result)


def _huber_thetas(forecasts, obs, *, huber_a, left_limit_delta, **_):
    """Return thetas for {HUBER} elementary scoring function."""
    uobs = np.unique(obs)
    uobs_minus_a = uobs - huber_a
    uobs_plus_a = uobs + huber_a
    ufcasts = np.unique(forecasts)
    left_limit_points = ufcasts - left_limit_delta
    ufcasts_and_uobs = np.unique(np.concatenate([ufcasts, left_limit_points, uobs, uobs_minus_a, uobs_plus_a]))
    result = ufcasts_and_uobs[~np.isnan(ufcasts_and_uobs)]
    return list(result)


def _expectile_thetas(forecasts, obs, *, left_limit_delta, **_):
    """Return thetas for {EXPECTILE} elementary scoring function."""
    ufcasts = np.unique(forecasts)
    left_limit_points = ufcasts - left_limit_delta
    ufcasts_and_uobs = np.unique(np.concatenate([ufcasts, left_limit_points, obs.values.flatten()]))
    result = ufcasts_and_uobs[~np.isnan(ufcasts_and_uobs)]
    return list(result)


def exposed_functions() -> dict[str, Callable[..., Any]]:
    """Expose functions used in calculation for easy usage upon user arguments."""
    return {
        "_quantile_elementary_score": _quantile_elementary_score,
        "_huber_elementary_score": _huber_elementary_score,
        "_expectile_elementary_score": _expectile_elementary_score,
        "_quantile_thetas": _quantile_thetas,
        "_huber_thetas": _huber_thetas,
        "_expectile_thetas": _expectile_thetas,
    }
