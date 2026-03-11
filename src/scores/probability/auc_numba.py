"""
Numba-accelerated AUC computation.

Provides a gufunc-based weighted Mann-Whitney U implementation that can be
called via ``xr.apply_ufunc`` with ``dask="parallelized"``.  The module is
imported lazily inside ``auc_impl.roc_auc`` so that the package works
without numba installed.
"""

import numpy as np
from numba import float64, guvectorize


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(n),(n),(n)->()",
) # pragma: no cover
def _roc_auc_mann_whitney_weighted_gufunc(
    fcst_flat: np.ndarray,
    obs_flat: np.ndarray,
    weights_flat: np.ndarray,
    result: np.ndarray,
) -> None:
    """Weighted ROC AUC via the Mann-Whitney U statistic (numba gufunc).

    Handles NaN masking, sorting, tie-group sweeping and returns a scalar
    AUC for each set of 1-D input vectors.
    """
    # NaN masking
    n_total = fcst_flat.shape[0]
    n_valid = 0
    for k in range(n_total):
        if not (np.isnan(fcst_flat[k]) or np.isnan(obs_flat[k]) or np.isnan(weights_flat[k])):
            n_valid += 1

    if n_valid == 0:
        result[0] = np.nan
        return

    fcst_v = np.empty(n_valid, dtype=np.float64)
    obs_v = np.empty(n_valid, dtype=np.float64)
    w_v = np.empty(n_valid, dtype=np.float64)
    idx = 0
    for k in range(n_total):
        if not (np.isnan(fcst_flat[k]) or np.isnan(obs_flat[k]) or np.isnan(weights_flat[k])):
            fcst_v[idx] = fcst_flat[k]
            obs_v[idx] = obs_flat[k]
            w_v[idx] = weights_flat[k]
            idx += 1

    # Get total positive / negative weights
    w_pos = 0.0  # weight of the positive class (obs == 1)
    w_neg = 0.0  # weight of the negative class (obs == 0)
    for k in range(n_valid):
        if obs_v[k] == 1.0:
            w_pos += w_v[k]
        else:
            w_neg += w_v[k]

    if w_pos == 0.0 or w_neg == 0.0:
        result[0] = np.nan
        return

    # Sort by forecast value
    order = np.argsort(fcst_v)
    fcst_s = fcst_v[order]
    obs_s = obs_v[order]
    w_s = w_v[order]

    # Sweep with cumulative negative weight
    u_weighted = 0.0
    cum_neg = 0.0
    i = 0
    n = n_valid

    while i < n:
        # Find the end of the current tie group
        j = i + 1
        while j < n and fcst_s[j] == fcst_s[i]:
            j += 1

        # Accumulate tie-group negative weight and positive U contribution
        tie_neg = 0.0
        tie_pos_w = 0.0
        for k in range(i, j):
            if obs_s[k] == 0.0:
                tie_neg += w_s[k]
            else:
                tie_pos_w += w_s[k]

        u_weighted += tie_pos_w * (cum_neg + 0.5 * tie_neg)
        cum_neg += tie_neg
        i = j

    result[0] = u_weighted / (w_pos * w_neg)
